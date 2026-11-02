"""YOLOv5 训练器：训练 / 每 epoch 验证（mAP）/ EMA / checkpoint / 早停。

训练指标写入 TensorBoard（``runs/<name>/tensorboard/``）；控制台与 ``train.log`` 只保留每 epoch 摘要。
"""

from __future__ import annotations

import random
import time
from itertools import islice
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import ROOT
from .criterion import build_criterion
from .data import YoloDetDataset, collate_fn, load_dataset_meta
from .ema import ModelEMA
from .evaluator import evaluate
from .models import build_model, count_parameters
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


def _on_off(flag: bool) -> str:
    return "on" if flag else "off"


class Trainer:
    def __init__(self, cfg: Mapping[str, Any], recorder: RunRecorder, device: torch.device) -> None:
        self.cfg = cfg
        self.recorder = recorder
        self.device = device
        self.log = recorder.logger()

        ds, hp, tr, loss_cfg, test_cfg = (
            cfg["dataset"], cfg["hyperparameters"], cfg["train"], cfg["loss"], cfg["test"],
        )
        self.hp, self.train_cfg = hp, tr
        self.num_epochs = int(hp["num_epochs"])
        self.img_size = int(ds["image_size"])
        self.limit_batches = int(tr.get("limit_batches", 0))
        self.grad_clip = float(tr.get("grad_clip_norm", 0.0))
        self.use_amp = bool(tr.get("amp", False)) and device.type == "cuda"
        self.conf_thres = float(test_cfg.get("conf_thres", 0.001))
        self.iou_thres = float(test_cfg.get("iou_thres", 0.6))
        self.max_det = int(test_cfg.get("max_det", 300))

        set_seed(int(tr.get("seed", 42)))
        data_dir = self._resolve_data_dir(ds["data_dir"])
        self.nc, self.names = load_dataset_meta(data_dir)

        augment = bool(ds.get("augment", False))
        mosaic_prob = float(ds.get("mosaic_prob", 0.0)) if augment else 0.0
        fliplr = float(ds.get("fliplr", 0.0)) if augment else 0.0
        self.train_set = YoloDetDataset(
            data_dir / "images" / "train", data_dir / "labels" / "train", self.img_size,
            train=True, mosaic_prob=mosaic_prob, fliplr=fliplr,
        )
        self.val_set = YoloDetDataset(
            data_dir / "images" / "val", data_dir / "labels" / "val", self.img_size, train=False,
        )
        batch = int(ds["batch_size"])
        workers = int(ds.get("num_workers", 0))
        pin = device.type == "cuda"
        self.train_loader = DataLoader(self.train_set, batch_size=batch, shuffle=True, num_workers=workers,
                                       collate_fn=collate_fn, pin_memory=pin)
        self.val_loader = DataLoader(self.val_set, batch_size=batch, shuffle=False, num_workers=workers,
                                     collate_fn=collate_fn, pin_memory=pin)

        self.model = build_model(cfg, self.nc).to(device)
        self.criterion = build_criterion(cfg, self.model, self.img_size).to(device)
        self.loss_names = tuple(getattr(self.criterion, "item_names", ("box", "obj", "cls")))
        self.optimizer = build_optimizer(hp, self.model)
        self.scheduler = build_scheduler(hp, self.optimizer, self.num_epochs)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.use_ema = bool(tr.get("ema", False))
        self.ema = ModelEMA(self.model, float(tr.get("ema_decay", 0.9999)), float(tr.get("ema_tau", 2000))) \
            if self.use_ema else None
        self.eval_model = self.ema.ema if self.ema is not None else self.model

        # 官方风格的 warmup：前 warmup_epochs 线性升温，bias 组 lr 从 warmup_bias_lr 降到目标值，动量从 warmup_momentum 升到目标值
        self.warmup_epochs = float(hp.get("warmup_epochs", 0.0))
        self.warmup_bias_lr = float(hp.get("warmup_bias_lr", 0.1))
        self.warmup_momentum = float(hp.get("warmup_momentum", 0.8))
        self.momentum = float(hp.get("momentum", 0.937))
        self.nb = len(self.train_loader) if not self.limit_batches else min(self.limit_batches, len(self.train_loader))
        self.nw = max(round(self.warmup_epochs * self.nb), 100) if self.warmup_epochs > 0 else 0

        self.early_stopping = bool(tr.get("early_stopping", False))
        self.patience = int(tr.get("early_stopping_patience", 0))
        self.min_delta = float(tr.get("early_stopping_min_delta", 0.0))
        if self.early_stopping and self.patience <= 0:
            raise ValueError("early_stopping=true 时需要 early_stopping_patience > 0")

        self.writer = recorder.tensorboard_writer()

    @staticmethod
    def _resolve_data_dir(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (ROOT / path).resolve()

    def _batches(self, loader: DataLoader):
        return islice(loader, self.limit_batches) if self.limit_batches else loader

    def _log_setup(self) -> None:
        cfg, log = self.cfg, self.log
        model_cfg, ds = cfg["model"], cfg["dataset"]
        log.info(f"实验 {self.recorder.name} | 模型 {model_cfg['display_name']} ({model_cfg['arch']}"
                 f"-{model_cfg.get('params', {}).get('variant', '')}) | 设备 {self.device}")
        log.info(f"数据 train={len(self.train_set)} / val={len(self.val_set)} 图片 | "
                 f"image_size={self.img_size} batch_size={ds['batch_size']} | 类别 nc={self.nc} {self.names}")
        log.info(f"参数量 {count_parameters(self.model) / 1e6:.2f}M")
        if hasattr(self.criterion, "describe"):
            log.info(self.criterion.describe())
        log.info(f"优化器 {self.hp.get('optimizer')} lr={self.hp['learning_rate']} "
                 f"momentum={self.hp.get('momentum')} wd={self.hp.get('weight_decay')} | "
                 f"调度 {self.hp.get('lr_scheduler')} | epochs={self.num_epochs}")
        log.info(f"开关 | AMP={_on_off(self.use_amp)} | EMA={_on_off(self.use_ema)} | "
                 f"数据增强={_on_off(bool(ds.get('augment', False)))} "
                 f"(mosaic_prob={ds.get('mosaic_prob')}, fliplr={ds.get('fliplr')}) | "
                 f"梯度裁剪={_on_off(self.grad_clip > 0)} | "
                 f"早停={_on_off(self.early_stopping)}"
                 + (f"(patience={self.patience})" if self.early_stopping else ""))
        if self.limit_batches:
            log.info(f"[quick] 每个 epoch 仅跑 {self.limit_batches} 个 batch")

    def _warmup(self, ni: int, epoch_lrs: list[float]) -> None:
        """迭代级 warmup：ni 为全局迭代步，超过 nw 后不再干预（交给 epoch 级调度器）。"""
        if ni > self.nw:
            return
        for j, g in enumerate(self.optimizer.param_groups):
            start = self.warmup_bias_lr if j == 2 else 0.0  # 参数分组顺序：[BN, 卷积权重, bias]
            g["lr"] = float(np.interp(ni, [0, self.nw], [start, epoch_lrs[j]]))
            if "momentum" in g:
                g["momentum"] = float(np.interp(ni, [0, self.nw], [self.warmup_momentum, self.momentum]))

    def _train_epoch(self, epoch: int) -> list[float]:
        self.model.train()
        n_items = len(self.loss_names)
        sums = torch.zeros(n_items)
        n = 0
        epoch_lrs = [g["lr"] for g in self.optimizer.param_groups]
        for i, (imgs, targets) in enumerate(self._batches(self.train_loader)):
            imgs = imgs.to(self.device, non_blocking=True).float() / 255.0
            targets = targets.to(self.device, non_blocking=True)
            self._warmup((epoch - 1) * self.nb + i, epoch_lrs)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(self.device.type, enabled=self.use_amp):
                preds = self.model(imgs)
            # YOLOv5: list[Tensor]；YOLOv8: dict
            if isinstance(preds, (list, tuple)):
                loss, items = self.criterion([p.float() for p in preds], targets)
            else:
                loss, items = self.criterion(preds, targets)
            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.ema is not None:
                self.ema.update(self.model)
            sums += items.detach().float().cpu()
            n += 1
        return (sums / max(n, 1)).tolist()

    def _log_epoch(self, m: Mapping[str, Any]) -> None:
        loss_txt = " ".join(f"{name} {m[name]:.4f}" for name in self.loss_names)
        self.log.info(
            f"[E{m['epoch']:03d}/{self.num_epochs}] "
            f"loss {loss_txt} | val {m['val_loss']:.4f} | "
            f"mAP50 {m['map50']:.4f} mAP50-95 {m['map']:.4f} P {m['precision']:.4f} R {m['recall']:.4f} | "
            f"lr {m['lr']:.2e} | {m['seconds']:.1f}s"
        )

    def _write_epoch_scalars(self, m: Mapping[str, Any], per_class: Mapping[str, float]) -> None:
        epoch, w = m["epoch"], self.writer
        for name in self.loss_names:
            w.add_scalar(f"loss/{name}", m[name], epoch)
        w.add_scalar("loss/val", m["val_loss"], epoch)
        w.add_scalar("metrics/precision", m["precision"], epoch)
        w.add_scalar("metrics/recall", m["recall"], epoch)
        w.add_scalar("metrics/mAP50", m["map50"], epoch)
        w.add_scalar("metrics/mAP50-95", m["map"], epoch)
        w.add_scalar("lr", m["lr"], epoch)
        w.add_scalar("time/epoch_seconds", m["seconds"], epoch)
        for name, ap in per_class.items():
            w.add_scalar(f"class_ap50/{name}", ap, epoch)

    def _write_config_text(self) -> None:
        text = self.recorder.config_path.read_text(encoding="utf-8")
        self.writer.add_text("config", f"```yaml\n{text}\n```", 0)

    def _state(self, epoch: int, val: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "model": self.eval_model.state_dict(),
            "epoch": epoch,
            "nc": self.nc,
            "names": self.names,
            "img_size": self.img_size,
            "metrics": {k: val[k] for k in ("map50", "map", "precision", "recall") if k in val},
        }

    def fit(self) -> dict[str, Any]:
        self._log_setup()
        self._write_config_text()
        best, best_loss, best_epoch, stale, completed = -1.0, float("inf"), None, 0, 0
        status = "completed"
        last: dict[str, Any] = {}
        try:
            for epoch in range(1, self.num_epochs + 1):
                start = time.time()
                lr = self.optimizer.param_groups[0]["lr"]
                item_vals = self._train_epoch(epoch)
                val = evaluate(
                    self.eval_model, self.val_loader, self.device, img_size=self.img_size, names=self.names,
                    criterion=self.criterion, conf_thres=self.conf_thres, iou_thres=self.iou_thres,
                    max_det=self.max_det, use_amp=self.use_amp,
                )
                completed = epoch
                last = {
                    "epoch": epoch, "lr": lr,
                    **dict(zip(self.loss_names, item_vals)),
                    "val_loss": val.get("val_loss", 0.0), "map50": val["map50"], "map": val["map"],
                    "precision": val["precision"], "recall": val["recall"],
                    "seconds": round(time.time() - start, 2),
                }
                self._log_epoch(last)
                self._write_epoch_scalars(last, val["per_class_ap50"])

                state = self._state(epoch, val)
                self.recorder.save_checkpoint("last", state)
                # best 依据 mAP50-95；mAP 相同（如早期全为 0）时以验证损失更低者为准
                map_improved = val["map"] > best + self.min_delta
                loss_better = val["map"] >= best and val["val_loss"] < best_loss
                if map_improved or loss_better:
                    best, best_loss, best_epoch = val["map"], val["val_loss"], epoch
                    stale = 0 if map_improved else stale + 1
                    self.recorder.save_checkpoint("best", state)
                    self.log.info(f"  * best mAP50-95={val['map']:.4f} val={val['val_loss']:.4f} "
                                  f"-> checkpoints/best.pth")
                else:
                    stale += 1
                step_scheduler(self.scheduler)
                self.writer.flush()

                if self.early_stopping and stale >= self.patience:
                    status = "early_stopped"
                    self.log.info(f"早停: mAP50-95 连续 {stale} epoch 未提升（patience={self.patience}），"
                                  f"在 epoch {epoch} 停止")
                    break
        except KeyboardInterrupt:
            status = "interrupted"
            self.log.info("训练被手动中断，已保存的 checkpoint 仍可用于 eval")
        except Exception:
            status = "failed"
            raise
        finally:
            self.writer.close()
            self.recorder.finish_training(
                status=status,
                best_map50_95=best if best_epoch is not None else None,
                best_epoch=best_epoch,
                completed_epochs=completed,
                last_metrics=last or None,
            )
            self.log.info(f"结束 status={status} | best mAP50-95={max(best, 0.0):.4f} @ epoch {best_epoch} | "
                          f"TensorBoard: {self.recorder.tensorboard_dir}")
        return {"status": status, "best_map50_95": best, "best_epoch": best_epoch}
