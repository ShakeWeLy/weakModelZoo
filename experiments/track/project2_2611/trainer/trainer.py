"""SiamSPN++ 训练器：孪生对训练 / 可选 val loss / 主干冻结 / checkpoint。"""

from __future__ import annotations

import random
import time
from itertools import islice
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from .criterion import build_criterion
from .data import build_dataloader, build_pair_dataset, try_build_pair_dataset
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


class Trainer:
    def __init__(self, cfg: Mapping[str, Any], recorder: RunRecorder, device: torch.device) -> None:
        self.cfg = cfg
        self.recorder = recorder
        self.device = device
        self.log = recorder.logger()

        ds, hp, tr = cfg["dataset"], cfg["hyperparameters"], cfg["train"]
        self.hp, self.train_cfg = hp, tr
        self.num_epochs = int(hp["num_epochs"])
        self.freeze_backbone_epochs = int(hp.get("freeze_backbone_epochs", 0))
        self.limit_batches = int(tr.get("limit_batches", 0))
        self.grad_clip = float(hp.get("grad_clip_norm", tr.get("grad_clip_norm", 0.0)))
        self.print_freq = int(tr.get("print_freq", 50))
        self.use_amp = bool(tr.get("amp", False)) and device.type == "cuda"

        set_seed(int(tr.get("seed", 42)))

        self.train_set = build_pair_dataset(cfg, splits_key="splits")
        self.val_set = try_build_pair_dataset(cfg, splits_key="val_splits")
        self.train_loader = build_dataloader(self.train_set, cfg, shuffle=True, device_type=device.type)
        self.val_loader: DataLoader | None = None
        if self.val_set is not None:
            self.val_loader = build_dataloader(self.val_set, cfg, shuffle=False, device_type=device.type)

        self.model = build_model(cfg).to(device)
        self.criterion = build_criterion(cfg).to(device)
        self.loss_names = tuple(getattr(self.criterion, "item_names", ("cls", "loc")))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.early_stopping = bool(tr.get("early_stopping", False))
        self.patience = int(tr.get("early_stopping_patience", 0))
        self.min_delta = float(tr.get("early_stopping_min_delta", 0.0))
        if self.early_stopping and self.patience <= 0:
            raise ValueError("early_stopping=true 时需要 early_stopping_patience > 0")

        self.writer = recorder.tensorboard_writer()
        self._train_backbone: bool | None = None
        self.optimizer = None
        self.scheduler = None
        self._sync_trainable(epoch=1)

    def _sync_trainable(self, epoch: int) -> None:
        train_bb = epoch > self.freeze_backbone_epochs
        if self._train_backbone is train_bb and self.optimizer is not None:
            return
        self._train_backbone = train_bb
        self.model.freeze_backbone(not train_bb)
        self.optimizer = build_optimizer(self.hp, self.model, train_backbone=train_bb)
        self.scheduler = build_scheduler(self.hp, self.optimizer, self.num_epochs)
        self.log.info(
            "epoch>=%d backbone=%s | trainable=%.2fM",
            epoch,
            "on" if train_bb else "freeze",
            sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6,
        )

    def _log_setup(self) -> None:
        m = self.cfg["model"]
        ds = self.cfg["dataset"]
        self.log.info(
            "model=%s arch=%s params=%.2fM | data=%s modality=%s seq=%d pairs/epoch=%d batch=%d",
            m.get("display_name"),
            m.get("arch"),
            count_parameters(self.model) / 1e6,
            ds.get("data_dir"),
            ds.get("modality"),
            len(self.train_set.sequences),
            len(self.train_set),
            int(ds["batch_size"]),
        )
        self.log.info(
            "epochs=%d freeze_bb=%d lr=%s amp=%s device=%s | %s",
            self.num_epochs,
            self.freeze_backbone_epochs,
            self.hp.get("learning_rate"),
            self.use_amp,
            self.device,
            self.criterion.describe() if hasattr(self.criterion, "describe") else "",
        )
        if self.val_loader is None:
            self.log.info("val: disabled (无可用 val_splits)")

    def _move_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        out = {
            "template": batch["template"].to(self.device, non_blocking=True),
            "search": batch["search"].to(self.device, non_blocking=True),
            "label_cls": batch["label_cls"].to(self.device, non_blocking=True),
            "label_loc": batch["label_loc"].to(self.device, non_blocking=True),
            "label_loc_weight": batch["label_loc_weight"].to(self.device, non_blocking=True),
        }
        return out

    def _forward_loss(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        out = self.model(batch["template"], batch["search"])
        losses = self.criterion(
            out["cls"],
            out["loc"],
            batch["label_cls"],
            batch["label_loc"],
            batch["label_loc_weight"],
        )
        items = {name: float(losses[name]) for name in self.loss_names if name in losses}
        items["loss"] = float(losses["loss"])
        return losses["loss"], items

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self._sync_trainable(epoch)
        self.model.train()
        if not self._train_backbone:
            self.model.backbone.eval()

        loader = self.train_loader
        n_total = len(loader) if not self.limit_batches else min(self.limit_batches, len(loader))
        it = loader if not self.limit_batches else islice(loader, self.limit_batches)

        sums = {k: 0.0 for k in ("loss", *self.loss_names)}
        steps = 0
        for step, raw in enumerate(it, start=1):
            batch = self._move_batch(raw)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                loss, items = self._forward_loss(batch)
            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for k, v in items.items():
                sums[k] = sums.get(k, 0.0) + v
            steps += 1
            if step % self.print_freq == 0 or step == 1 or step == n_total:
                avg_loss = sums["loss"] / steps
                self.log.info(
                    "epoch %d/%d step %d/%d loss=%.4f cls=%.4f loc=%.4f bb=%s",
                    epoch,
                    self.num_epochs,
                    step,
                    n_total,
                    avg_loss,
                    sums.get("cls", 0.0) / steps,
                    sums.get("loc", 0.0) / steps,
                    "on" if self._train_backbone else "freeze",
                )

        return {k: v / max(steps, 1) for k, v in sums.items()}

    @torch.no_grad()
    def _validate(self) -> dict[str, float] | None:
        if self.val_loader is None:
            return None
        self.model.eval()
        sums = {k: 0.0 for k in ("loss", *self.loss_names)}
        steps = 0
        it = self.val_loader if not self.limit_batches else islice(self.val_loader, self.limit_batches)
        for raw in it:
            batch = self._move_batch(raw)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                _, items = self._forward_loss(batch)
            for k, v in items.items():
                sums[k] = sums.get(k, 0.0) + v
            steps += 1
        return {k: v / max(steps, 1) for k, v in sums.items()}

    def fit(self) -> dict[str, Any]:
        self._log_setup()
        best, best_epoch, stale, completed = float("inf"), None, 0, 0
        status = "completed"
        last: dict[str, Any] = {}
        try:
            for epoch in range(1, self.num_epochs + 1):
                t0 = time.time()
                train_m = self._train_epoch(epoch)
                val_m = self._validate()
                monitor = float((val_m or train_m)["loss"])
                lr = self.optimizer.param_groups[0]["lr"]

                self.writer.add_scalar("loss/train", train_m["loss"], epoch)
                for name in self.loss_names:
                    if name in train_m:
                        self.writer.add_scalar(f"loss/train_{name}", train_m[name], epoch)
                if val_m is not None:
                    self.writer.add_scalar("loss/val", val_m["loss"], epoch)
                self.writer.add_scalar("lr", lr, epoch)

                msg = (
                    f"epoch {epoch}/{self.num_epochs} train_loss={train_m['loss']:.4f} "
                    f"lr={lr:.6g} time={time.time()-t0:.1f}s"
                )
                if val_m is not None:
                    msg += f" val_loss={val_m['loss']:.4f}"
                self.log.info(msg)

                ckpt = {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "metrics": {"train": train_m, "val": val_m},
                    "cfg": dict(self.cfg),
                }
                self.recorder.save_checkpoint("last", ckpt)
                improved = monitor < best - self.min_delta
                if improved:
                    best, best_epoch, stale = monitor, epoch, 0
                    self.recorder.save_checkpoint("best", ckpt)
                else:
                    stale += 1

                last = {"train": train_m, "val": val_m, "lr": lr, "monitor_loss": monitor}
                completed = epoch
                self.recorder.update_summary(
                    completed_epochs=completed,
                    best_loss=best if best < float("inf") else None,
                    best_epoch=best_epoch,
                )
                step_scheduler(self.scheduler)

                if self.early_stopping and stale >= self.patience:
                    status = "early_stop"
                    self.log.info("early stop at epoch %d (best=%.4f @ %s)", epoch, best, best_epoch)
                    break
        except KeyboardInterrupt:
            status = "interrupted"
            self.log.info("interrupted")
        finally:
            self.writer.close()
            self.recorder.finish_training(
                status=status,
                best_loss=best if best < float("inf") else None,
                best_epoch=best_epoch,
                completed_epochs=completed,
                last_metrics=last,
            )
        return {"status": status, "best_loss": best, "best_epoch": best_epoch}
