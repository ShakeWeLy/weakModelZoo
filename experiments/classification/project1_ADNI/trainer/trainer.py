"""五折交叉验证训练器：所有分类模型共用同一套流程。

每折：训练集（其余 4 折）训练，留出折作为验证集；结论取最后一个 epoch 的指标（不按验证集挑选 epoch，避免乐观偏差）。
"""

from __future__ import annotations

import random
import time
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn

from .config import ROOT  # noqa: F401  (保证 ROOT 在 sys.path 中)
from .data import load_data, make_loader
from .metrics import METRIC_KEYS, binary_metrics, summarize_folds
from .models import build_model, count_parameters
from .optim import build_optimizer, build_scheduler
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


class CVTrainer:
    def __init__(self, cfg: Mapping[str, Any], recorder: RunRecorder, device: torch.device):
        self.cfg = cfg
        self.recorder = recorder
        self.device = device
        self.log = recorder.logger()
        self.ds = cfg["dataset"]
        self.hp = cfg["hyperparameters"]
        self.tr = cfg["train"]
        self.seed = int(self.tr.get("seed", 42))
        self.threshold = float(self.tr.get("threshold", 0.5))
        self.use_tensorboard = bool(self.tr.get("tensorboard", True))
        self.num_epochs = int(self.hp["num_epochs"])
        self.num_classes = 2

    def fit(self) -> dict[str, Any]:
        X, y, source = load_data(self.ds, self.seed, self.recorder.run_dir)
        self.log.info(f"数据: {source} | X {tuple(X.shape)} | 正类比例 {y.mean():.3f} | device {self.device}")
        self.log.info(f"模型参数: {self.cfg['model']['params']}")

        X_t, y_t = torch.from_numpy(X), torch.from_numpy(y)
        skf = StratifiedKFold(n_splits=int(self.ds["num_folds"]), shuffle=True, random_state=self.seed)

        fold_results: list[dict[str, float]] = []
        status = "completed"
        try:
            for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
                self.log.info(f"===== Fold {fold}/{skf.get_n_splits()} | train {len(tr_idx)} | val {len(va_idx)} =====")
                tr_t, va_t = torch.from_numpy(tr_idx), torch.from_numpy(va_idx)
                metrics = self._run_fold(
                    fold, X_t[tr_t], y_t[tr_t], X_t[va_t], y_t[va_t],
                )
                fold_results.append(metrics)
                self.log.info(f"[fold {fold}] 最终: " + " ".join(f"{k.upper()} {metrics[k]:.4f}" for k in METRIC_KEYS))
        except KeyboardInterrupt:
            status = "interrupted"
            self.log.info("训练被手动中断，已完成的折仍会写入结果")
        except Exception:
            status = "failed"
            raise
        finally:
            summary = summarize_folds(fold_results) if fold_results else {}
            self.recorder.finish(status=status, fold_results=fold_results, summary=summary)
            if summary:
                self.log.info("五折结果 (mean ± std): " + " | ".join(
                    f"{k.upper()} {summary[k]['mean']:.4f}±{summary[k]['std']:.4f}" for k in METRIC_KEYS))
            self.log.info(f"结束 status={status} | 结果目录: {self.recorder.run_dir}")
            self.recorder.close_logger()
        return {"status": status, "folds": fold_results}

    def _run_fold(self, fold: int, X_tr: torch.Tensor, y_tr: torch.Tensor,
                  X_va: torch.Tensor, y_va: torch.Tensor) -> dict[str, float]:
        set_seed(self.seed + fold)
        model = build_model(self.cfg, num_nodes=X_tr.shape[-1], num_classes=self.num_classes).to(self.device)
        if fold == 1:
            self.log.info(f"可训练参数量: {count_parameters(model):,}")
        optimizer = build_optimizer(self.hp, model)
        scheduler = build_scheduler(self.hp, optimizer, self.num_epochs)
        criterion = nn.CrossEntropyLoss()

        batch_size = int(self.ds["batch_size"])
        train_loader = make_loader(X_tr, y_tr, batch_size, shuffle=True, seed=self.seed + fold)
        val_loader = make_loader(X_va, y_va, batch_size, shuffle=False)
        writer = self.recorder.tensorboard_writer(fold) if self.use_tensorboard else None

        history: list[dict[str, Any]] = []
        try:
            for epoch in range(1, self.num_epochs + 1):
                t0 = time.time()
                lr = optimizer.param_groups[0]["lr"]
                train_loss = self._train_epoch(model, train_loader, optimizer, criterion)
                val_loss, val_prob = self._eval_epoch(model, val_loader, criterion)
                m = binary_metrics(y_va.numpy(), val_prob, self.threshold)
                row = {"epoch": epoch, "lr": lr, "train_loss": train_loss, "val_loss": val_loss, **m}
                history.append(row)
                self.log.info(
                    f"[fold {fold}] epoch {epoch:02d}/{self.num_epochs} | lr {lr:.2e} | "
                    f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                    f"AUC {m['auc']:.4f} ACC {m['acc']:.4f} SEN {m['sen']:.4f} SPE {m['spe']:.4f} BAC {m['bac']:.4f} "
                    f"| {time.time() - t0:.1f}s"
                )
                if writer is not None:
                    writer.add_scalar("loss/train", train_loss, epoch)
                    writer.add_scalar("loss/val", val_loss, epoch)
                    for k in METRIC_KEYS:
                        writer.add_scalar(f"val/{k}", m[k], epoch)
                    writer.add_scalar("lr", lr, epoch)
                if scheduler is not None:
                    scheduler.step()
        finally:
            if writer is not None:
                writer.close()

        self.recorder.save_fold_history(fold, history)
        self.recorder.save_checkpoint(fold, {
            "model_state_dict": model.state_dict(),
            "epoch": self.num_epochs,
            "metrics": {k: history[-1][k] for k in METRIC_KEYS},
            "config": dict(self.cfg),
        })
        return {k: history[-1][k] for k in METRIC_KEYS}

    def _train_epoch(self, model, loader, optimizer, criterion) -> float:
        model.train()
        total, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        return total / max(1, n)

    @torch.no_grad()
    def _eval_epoch(self, model, loader, criterion) -> tuple[float, np.ndarray]:
        model.eval()
        total, n, probs = 0.0, 0, []
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = model(x)
            total += criterion(logits, y).item() * x.size(0)
            n += x.size(0)
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        return total / max(1, n), np.concatenate(probs)
