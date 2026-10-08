"""通用分割训练器：所有模型共用同一套训练 / 验证 / checkpoint / 早停流程。

训练指标（loss / dice / lr / 逐类 Dice / 超参）写入 TensorBoard（``runs/<name>/tensorboard/``）；
控制台与 ``train.log`` 只保留精简的每 epoch 摘要。
"""

from __future__ import annotations

import random
import time
from itertools import islice
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import build_class_sampler, build_dataset, build_loader
from .labels import LabelSpace
from .losses import build_loss
from .metrics import DiceAccumulator
from .models import build_model, count_parameters, model_in_channels
from .optim import build_optimizer, build_scheduler, step_scheduler
from .recorder import RunRecorder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | None) -> torch.device:
    if requested and requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] 请求 {requested} 但 CUDA 不可用，改用 cpu")
        return torch.device("cpu")
    return torch.device(requested or ("cuda" if torch.cuda.is_available() else "cpu"))


def checkpoint_state(model: nn.Module, cfg: Mapping[str, Any], epoch: int, val_dice: float,
                     optimizer: torch.optim.Optimizer | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "val_dice": val_dice,
        "config": dict(cfg),
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    return state


def _on_off(flag: bool) -> str:
    return "on" if flag else "off"


class Trainer:
    def __init__(self, cfg: dict[str, Any], recorder: RunRecorder, device: torch.device):
        self.cfg = cfg
        self.recorder = recorder
        self.device = device
        self.train_cfg = cfg["train"]
        self.hp = cfg["hyperparameters"]
        self.num_epochs = int(self.hp["num_epochs"])
        self.limit_batches = int(self.train_cfg.get("limit_batches", 0)) or None
        self.log = recorder.logger()

        set_seed(int(self.train_cfg.get("seed", 42)))
        self.space = LabelSpace.from_config(cfg)
        pin = device.type == "cuda"
        self.train_set = build_dataset(cfg, "train", train=True)
        self.val_set = build_dataset(cfg, "val")
        batch_size = int(cfg["dataset"]["batch_size"])
        self.guaranteed_sampling = bool(self.train_cfg.get("guaranteed_sampling", False))
        self.sampler = (
            build_class_sampler(self.train_set, self.train_cfg, batch_size, self.space.class_name)
            if self.guaranteed_sampling else None
        )
        self.train_loader = build_loader(self.train_set, cfg, shuffle=True, sampler=self.sampler, pin_memory=pin)
        self.val_loader = build_loader(self.val_set, cfg, pin_memory=pin)

        self.model = build_model(cfg, self.space.num_classes).to(device)
        self.criterion = build_loss(self.train_cfg, self.space.num_classes).to(device)
        self.optimizer = build_optimizer(self.hp, self.model)
        self.scheduler = build_scheduler(self.hp, self.optimizer, self.num_epochs)
        self.use_amp = bool(self.train_cfg.get("amp", False)) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.grad_clip = float(self.train_cfg.get("grad_clip_norm", 0.0))

        self.class_metrics_every = int(self.train_cfg.get("class_metrics_every", 0))
        self.class_metrics_splits = tuple(self.train_cfg.get("class_metrics_splits", ["val"]))

        # 早停开关：开启时 patience 必须为正数，否则无法判断"未提升"的次数
        self.early_stopping = bool(self.train_cfg.get("early_stopping", False))
        self.patience = int(self.train_cfg.get("early_stopping_patience", 0))
        self.min_delta = float(self.train_cfg.get("early_stopping_min_delta", 0.0))
        if self.early_stopping and self.patience <= 0:
            raise ValueError("early_stopping=true 时需要 early_stopping_patience > 0")

        self.writer = recorder.tensorboard_writer()

    def _batches(self, loader: DataLoader):
        return islice(loader, self.limit_batches) if self.limit_batches else loader

    def _log_setup(self) -> None:
        cfg, log = self.cfg, self.log
        model_cfg, ds, tr = cfg["model"], cfg["dataset"], self.train_cfg
        log.info(f"实验 {self.recorder.name} | 模型 {model_cfg['display_name']} ({model_cfg['arch']}) | 设备 {self.device}")
        log.info(
            f"数据 train={len(self.train_set)} / val={len(self.val_set)} 切片 | "
            f"image_size={ds['image_size']} batch_size={ds['batch_size']}"
        )
        log.info(
            f"参数量 {count_parameters(self.model) / 1e6:.2f}M | in_channels={model_in_channels(model_cfg)} | "
            f"classes={self.space.num_classes}"
        )
        log.info(f"损失 {self.criterion.describe()}")
        log.info(
            f"优化器 {self.hp.get('optimizer')} lr={self.hp['learning_rate']} "
            f"wd={self.hp.get('weight_decay', 0)} | 调度 {self.hp.get('lr_scheduler') or 'none'} | "
            f"epochs={self.num_epochs}"
        )
        log.info(
            "开关 | "
            f"AMP={_on_off(self.use_amp)} | 梯度裁剪={_on_off(self.grad_clip > 0)} | "
            f"数据增强={_on_off(bool(ds.get('augment', False)))} | "
            f"类别权重={_on_off(bool(tr.get('use_class_weights', False)))} | "
            f"跳过缺失类={_on_off(bool(tr.get('skip_absent_classes', True)))} | "
            f"保底采样={_on_off(self.guaranteed_sampling)} | "
            f"早停={_on_off(self.early_stopping)}"
            + (f"(patience={self.patience}, min_delta={self.min_delta})" if self.early_stopping else "")
        )
        if self.sampler is not None:
            pools = ", ".join(f"{n}={s}" for n, s in zip(self.sampler.group_names, self.sampler.pool_sizes))
            log.info(f"保底采样池: {pools} | 每组每 epoch 至少 {self.sampler.min_samples_per_group} 个")
        if self.limit_batches:
            log.info(f"[quick] 每个 epoch 仅跑 {self.limit_batches} 个 batch")

    def _run_epoch(self, loader: DataLoader, *, train: bool) -> tuple[float, DiceAccumulator]:
        self.model.train(train)
        acc = DiceAccumulator(self.space)
        total_loss, count = 0.0, 0
        with torch.set_grad_enabled(train):
            for images, labels, _ in self._batches(loader):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                with torch.autocast(self.device.type, enabled=self.use_amp):
                    logits = self.model(images)
                    loss = self.criterion(logits.float(), labels)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                total_loss += loss.item()
                count += 1
                acc.update(logits, labels)
        return total_loss / max(count, 1), acc

    def _should_log_classes(self, epoch: int) -> bool:
        every = self.class_metrics_every
        return every > 0 and (epoch == 1 or epoch == self.num_epochs or epoch % every == 0)

    def _log_epoch(self, m: Mapping[str, Any]) -> None:
        self.log.info(
            f"[E{m['epoch']:03d}/{self.num_epochs}] "
            f"loss tr {m['train_loss']:.4f} va {m['val_loss']:.4f} | "
            f"dice tr {m['train_dice']:.4f} va {m['val_dice']:.4f} | "
            f"lr {m['lr']:.2e} | {m['seconds']:.1f}s"
        )

    def _write_epoch_scalars(self, m: Mapping[str, Any]) -> None:
        epoch, w = m["epoch"], self.writer
        w.add_scalar("loss/train", m["train_loss"], epoch)
        w.add_scalar("loss/val", m["val_loss"], epoch)
        w.add_scalar("dice/train", float(m["train_dice"]), epoch)
        w.add_scalar("dice/val", float(m["val_dice"]), epoch)
        w.add_scalar("lr", m["lr"], epoch)
        w.add_scalar("time/epoch_seconds", m["seconds"], epoch)

    def _log_class_dice(self, epoch: int, accumulators: Mapping[str, DiceAccumulator]) -> None:
        for split in self.class_metrics_splits:
            acc = accumulators.get(split)
            if acc is None:
                continue
            parts = []
            for class_id, dice in acc.per_class().items():
                name = self.space.class_name(class_id)
                if dice is None:
                    parts.append(f"{name}=N/A")
                    continue
                self.writer.add_scalar(f"class_dice/{split}/{name}", float(dice), epoch)
                parts.append(f"{name}={dice:.4f}")
            self.log.info(f"  class dice {split}: " + " | ".join(parts))

    def _write_config_text(self) -> None:
        """把合并后的完整配置以文本形式写入 TensorBoard 的 Text 页。"""
        text = self.recorder.config_path.read_text(encoding="utf-8")
        self.writer.add_text("config", f"```yaml\n{text}\n```", 0)

    def _write_hparams(self, best_dice: float) -> None:
        """写入 TensorBoard 的 HParams 页，便于比较不同实验的超参与结果。"""
        tr, ds = self.train_cfg, self.cfg["dataset"]
        hparams = {
            "arch": str(self.cfg["model"]["arch"]),
            "optimizer": str(self.hp.get("optimizer")),
            "learning_rate": float(self.hp["learning_rate"]),
            "lr_scheduler": str(self.hp.get("lr_scheduler") or "none"),
            "batch_size": int(ds["batch_size"]),
            "dice_loss_weight": float(tr.get("dice_loss_weight", 1.0)),
            "ce_loss_weight": float(tr.get("ce_loss_weight", 1.0)),
            "use_class_weights": bool(tr.get("use_class_weights", False)),
            "skip_absent_classes": bool(tr.get("skip_absent_classes", True)),
            "guaranteed_sampling": self.guaranteed_sampling,
            "early_stopping": self.early_stopping,
            "amp": self.use_amp,
        }
        self.writer.add_hparams(hparams, {"hparam/best_val_dice": float(best_dice)})

    def fit(self) -> dict[str, Any]:
        self._log_setup()
        self._write_config_text()
        best_dice, best_epoch, completed, stale = -1.0, None, 0, 0
        status = "completed"
        last: dict[str, Any] = {}
        try:
            for epoch in range(1, self.num_epochs + 1):
                if self.sampler is not None:
                    self.sampler.set_epoch(epoch)
                start = time.time()
                lr = self.optimizer.param_groups[0]["lr"]
                train_loss, train_acc = self._run_epoch(self.train_loader, train=True)
                val_loss, val_acc = self._run_epoch(self.val_loader, train=False)
                train_dice, val_dice = train_acc.mean(), val_acc.mean()
                elapsed = time.time() - start
                completed = epoch

                last = {
                    "epoch": epoch, "lr": lr,
                    "train_loss": train_loss, "train_dice": float(train_dice),
                    "val_loss": val_loss, "val_dice": float(val_dice),
                    "seconds": round(elapsed, 2),
                }
                self._log_epoch(last)
                self._write_epoch_scalars(last)
                if self._should_log_classes(epoch):
                    self._log_class_dice(epoch, {"train": train_acc, "val": val_acc})

                state = checkpoint_state(self.model, self.cfg, epoch, val_dice, self.optimizer)
                self.recorder.save_checkpoint("last", state)
                if val_dice > best_dice + self.min_delta:
                    best_dice, best_epoch, stale = val_dice, epoch, 0
                    self.recorder.save_checkpoint("best", state)
                    self.log.info(f"  * best val_dice={val_dice:.4f} -> checkpoints/best.pth")
                else:
                    stale += 1
                step_scheduler(self.scheduler, val_dice)
                self.writer.flush()

                if self.early_stopping and stale >= self.patience:
                    status = "early_stopped"
                    self.log.info(
                        f"早停: val_dice 连续 {stale} epoch 未提升（patience={self.patience}），"
                        f"在 epoch {epoch} 停止"
                    )
                    break
        except KeyboardInterrupt:
            status = "interrupted"
            self.log.info("训练被手动中断，已保存的 checkpoint 仍可用于 analyze")
        except Exception:
            status = "failed"
            raise
        finally:
            if best_epoch is not None:
                self._write_hparams(best_dice)
            self.writer.close()
            self.recorder.finish_training(
                status=status,
                best_val_dice=best_dice if best_epoch is not None else None,
                best_epoch=best_epoch,
                completed_epochs=completed,
                last_metrics=last or None,
            )
            self.log.info(
                f"结束 status={status} | best val_dice={best_dice:.4f} @ epoch {best_epoch} | "
                f"TensorBoard: {self.recorder.tensorboard_dir}"
            )
        return {"status": status, "best_val_dice": best_dice, "best_epoch": best_epoch}
