"""通用分割训练器：所有模型共用同一套训练 / 验证 / checkpoint / 早停流程。"""

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
        self.sampler = (
            build_class_sampler(self.train_set, self.train_cfg, batch_size, self.space.class_name)
            if self.train_cfg.get("guaranteed_sampling", False) else None
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
        self.patience = int(self.train_cfg.get("early_stopping_patience", 0))
        self.min_delta = float(self.train_cfg.get("early_stopping_min_delta", 0.0))

    def _batches(self, loader: DataLoader):
        return islice(loader, self.limit_batches) if self.limit_batches else loader

    def _log_setup(self) -> None:
        cfg, log = self.cfg, self.log
        model_cfg = cfg["model"]
        log.info(f"Experiment: {self.recorder.name}")
        log.info(f"Run dir: {self.recorder.run_dir}")
        log.info(f"Device: {self.device} | AMP: {self.use_amp}")
        log.info(
            f"Data: {cfg['dataset']['data_dir']} | train slices={len(self.train_set)}, "
            f"val slices={len(self.val_set)} | image_size={cfg['dataset']['image_size']}, "
            f"batch_size={cfg['dataset']['batch_size']}, augment={cfg['dataset'].get('augment', False)}"
        )
        log.info(f"Classes: {self.space.num_classes} | metric classes={list(self.space.metric_class_ids)}")
        log.info(
            f"Model: {model_cfg['display_name']} (arch={model_cfg['arch']}) | "
            f"in_channels={model_in_channels(model_cfg)} | params={count_parameters(self.model) / 1e6:.2f}M | "
            f"{model_cfg.get('params', {})}"
        )
        log.info(f"Loss: {self.criterion.describe()}")
        log.info(
            f"Optimizer: {self.hp.get('optimizer')} lr={self.hp['learning_rate']} "
            f"wd={self.hp.get('weight_decay', 0)} | scheduler={self.hp.get('lr_scheduler') or 'none'} | "
            f"epochs={self.num_epochs}"
        )
        if self.sampler is not None:
            pools = ", ".join(f"{n}={s}" for n, s in zip(self.sampler.group_names, self.sampler.pool_sizes))
            log.info(f"Guaranteed sampling: {pools} | min per group/epoch={self.sampler.min_samples_per_group}")
        if self.patience > 0:
            log.info(f"Early stopping: patience={self.patience}, min_delta={self.min_delta}")
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

    def _log_class_dice(self, epoch: int, accumulators: Mapping[str, DiceAccumulator], writer) -> None:
        for split in self.class_metrics_splits:
            acc = accumulators.get(split)
            if acc is None:
                continue
            rows, parts = [], []
            for class_id, dice in acc.per_class().items():
                name = self.space.class_name(class_id)
                rows.append({"epoch": epoch, "split": split, "class_id": class_id, "class_name": name, "dice": dice})
                parts.append(f"{name}={dice:.4f}" if dice is not None else f"{name}=N/A")
            writer.write_rows(rows)
            self.log.info(f"Epoch [{epoch:03d}] class dice [{split}]: " + ", ".join(parts))

    def fit(self) -> dict[str, Any]:
        self._log_setup()
        best_dice, best_epoch, completed, stale = -1.0, None, 0, 0
        status = "completed"
        history = self.recorder.history_writer()
        class_writer = self.recorder.class_metrics_writer()
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

                history.write({
                    "epoch": epoch, "lr": lr,
                    "train_loss": train_loss, "train_dice": train_dice,
                    "val_loss": val_loss, "val_dice": val_dice,
                    "seconds": round(elapsed, 2),
                })
                self.log.info(
                    f"Epoch [{epoch:03d}/{self.num_epochs}] train_loss={train_loss:.4f} train_dice={train_dice:.4f} "
                    f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} lr={lr:.6f} time={elapsed:.1f}s"
                )
                if self._should_log_classes(epoch):
                    self._log_class_dice(epoch, {"train": train_acc, "val": val_acc}, class_writer)

                state = checkpoint_state(self.model, self.cfg, epoch, val_dice, self.optimizer)
                self.recorder.save_checkpoint("last", state)
                if val_dice > best_dice + self.min_delta:
                    best_dice, best_epoch, stale = val_dice, epoch, 0
                    self.recorder.save_checkpoint("best", state)
                    self.log.info(f"  -> new best val_dice={val_dice:.4f}")
                else:
                    stale += 1
                step_scheduler(self.scheduler, val_dice)

                if self.patience > 0 and stale >= self.patience:
                    status = "early_stopped"
                    self.log.info(f"Early stopping at epoch {epoch}: val_dice 连续 {stale} epoch 未提升")
                    break
        except KeyboardInterrupt:
            status = "interrupted"
            self.log.info("训练被手动中断，已保存的 checkpoint 仍可用于 analyze")
        except Exception:
            status = "failed"
            raise
        finally:
            history.close()
            class_writer.close()
            self.recorder.finish_training(
                status=status,
                best_val_dice=best_dice if best_epoch is not None else None,
                best_epoch=best_epoch,
                completed_epochs=completed,
            )
            self.log.info(f"Status: {status} | best val_dice={best_dice:.4f} (epoch {best_epoch})")
            self.log.info(f"Best checkpoint: {self.recorder.checkpoint_path('best')}")
        return {"status": status, "best_val_dice": best_dice, "best_epoch": best_epoch}
