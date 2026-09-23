"""在 Synapse 2D 切片上训练 Attention U-Net，使用 Dice Loss。

用法（在项目根目录执行）：
    python experiments/segment/project1_2609/AttnUNet/train_attn_unet.py
    python experiments/segment/project1_2609/AttnUNet/train_attn_unet.py --config experiments/segment/project1_2609/AttnUNet/config.toml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[3]
sys.path.insert(0, str(ROOT))

from src.models.segment.AttentionGateUnet.AttentionGateUnet import AttentionGateUnet
from src.utils.data.synapse.labels import EVAL_CLASS_IDS, LABEL_NAMES
from src.utils.logger import (
    CheckpointManager,
    ClassMetricsLogger,
    CsvMetricsLogger,
    create_experiment_run,
    resolve_runs_root,
    setup_file_logger,
    update_training_summary,
)

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore


class SynapseSliceDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, image_size: int, num_classes: int):
        self.image_dir = data_dir / split / "images"
        self.label_dir = data_dir / split / "labels"
        self.image_size = image_size
        self.num_classes = num_classes
        self.samples = sorted(self.image_dir.glob("*.npy"))
        if not self.samples:
            raise FileNotFoundError(f"{self.image_dir} 下未找到 .npy 切片")
        self._validate_labels()

    def _validate_labels(self) -> None:
        import numpy as np

        max_label = 0
        min_label = 0
        for image_path in self.samples:
            label = np.load(self.label_dir / image_path.name)
            max_label = max(max_label, int(label.max()))
            min_label = min(min_label, int(label.min()))
        if min_label < 0:
            raise ValueError(f"{self.label_dir} 存在负标签值: {min_label}")
        if max_label >= self.num_classes:
            raise ValueError(
                f"标签最大值 {max_label} 超出 class_nums={self.num_classes}，"
                f"请将 config.toml 中 class_nums 设为 {max_label + 1}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        import numpy as np

        image_path = self.samples[index]
        label_path = self.label_dir / image_path.name
        image = np.load(image_path).astype("float32")
        label = np.load(label_path).astype("int64")

        image = torch.from_numpy(image).unsqueeze(0)
        label = torch.from_numpy(label)

        image = F.interpolate(
            image.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        label = F.interpolate(
            label.unsqueeze(0).unsqueeze(0).float(),
            size=(self.image_size, self.image_size),
            mode="nearest",
        ).squeeze(0).squeeze(0).long()
        return image, label


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_background: bool = True, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_background = ignore_background
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        if self.ignore_background:
            dice = dice[1:]
        return 1.0 - dice.mean()


@torch.no_grad()
def compute_mean_dice(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_background: bool = True,
    eps: float = 1e-6,
) -> float:
    preds = torch.argmax(logits, dim=1)
    class_ids = range(1, num_classes) if ignore_background else range(num_classes)
    dice_scores = []
    for class_id in class_ids:
        pred_mask = preds == class_id
        target_mask = targets == class_id
        intersection = (pred_mask & target_mask).sum().item()
        union = pred_mask.sum().item() + target_mask.sum().item()
        if union == 0:
            continue
        dice_scores.append((2.0 * intersection + eps) / (union + eps))
    if not dice_scores:
        return 0.0
    return float(sum(dice_scores) / len(dice_scores))


def load_config(config_path: Path) -> dict:
    with config_path.open("rb") as f:
        return tomllib.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.toml"),
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_dice += compute_mean_dice(logits, labels, criterion.num_classes)
    count = len(loader)
    return total_loss / count, total_dice / count


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        total_dice += compute_mean_dice(logits, labels, criterion.num_classes)
    count = len(loader)
    return total_loss / count, total_dice / count


@torch.no_grad()
def compute_per_class_dice(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_ids: tuple[int, ...] | list[int],
    eps: float = 1e-6,
) -> dict[int, float | None]:
    model.eval()
    intersection = {class_id: 0.0 for class_id in class_ids}
    union = {class_id: 0.0 for class_id in class_ids}
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        preds = torch.argmax(model(images), dim=1)
        for class_id in class_ids:
            pred_mask = preds == class_id
            target_mask = labels == class_id
            intersection[class_id] += (pred_mask & target_mask).sum().item()
            union[class_id] += pred_mask.sum().item() + target_mask.sum().item()
    results: dict[int, float | None] = {}
    for class_id in class_ids:
        if union[class_id] == 0:
            results[class_id] = None
        else:
            results[class_id] = float((2.0 * intersection[class_id] + eps) / (union[class_id] + eps))
    return results


def should_log_class_metrics(epoch: int, num_epochs: int, every: int) -> bool:
    if every <= 0:
        return False
    return epoch == 1 or epoch == num_epochs or epoch % every == 0


def log_class_metrics(
    epoch: int,
    split: str,
    class_dice: dict[int, float | None],
    class_metrics_logger: ClassMetricsLogger,
    logger,
) -> None:
    rows = []
    parts = []
    for class_id in sorted(class_dice):
        dice = class_dice[class_id]
        class_name = LABEL_NAMES.get(class_id, f"class_{class_id}")
        rows.append(
            {
                "epoch": epoch,
                "split": split,
                "class_id": class_id,
                "class_name": class_name,
                "dice": round(dice, 4) if dice is not None else "",
            }
        )
        dice_text = f"{dice:.4f}" if dice is not None else "N/A"
        parts.append(f"{class_name}={dice_text}")
    class_metrics_logger.log_rows(rows)
    logger.info(f"Epoch [{epoch:03d}] class dice [{split}]: " + ", ".join(parts))


def main():
    args = parse_args()
    cfg = load_config(args.config)
    dataset_cfg = cfg["dataset"]
    hyper_cfg = cfg["hyperparameters"]
    model_cfg = cfg["model"]
    train_cfg = cfg.get("train", {})

    exp_cfg = cfg.get("experiments", {})
    experiment_name = exp_cfg.get("name")
    if not experiment_name:
        raise ValueError("config 缺少 [experiments].name，每次训练请指定唯一名称")

    data_dir = ROOT / train_cfg.get("data_dir", "data/synapse_processed")
    run = create_experiment_run(
        resolve_runs_root(EXP_DIR, cfg),
        experiment_name,
        cfg,
        args.config,
    )

    device = torch.device(args.device or train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    image_size = int(dataset_cfg["image_size"])
    batch_size = int(dataset_cfg["batch_size"])
    num_workers = int(dataset_cfg["num_workers"])
    num_classes = int(model_cfg["class_nums"])
    hidden_channels = int(model_cfg["hidden_channels"])
    num_epochs = int(hyper_cfg["num_epochs"])
    class_metrics_every = int(train_cfg.get("class_metrics_every", 0))
    class_metrics_splits = list(train_cfg.get("class_metrics_splits", ["val"]))
    metric_class_ids = tuple(int(v) for v in train_cfg.get("metric_class_ids", EVAL_CLASS_IDS))

    train_set = SynapseSliceDataset(data_dir, "train", image_size, num_classes)
    val_set = SynapseSliceDataset(data_dir, "val", image_size, num_classes)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = AttentionGateUnet(
        in_channels=1,
        out_channels=num_classes,
        hidden_channels=hidden_channels,
    ).to(device)
    criterion = DiceLoss(num_classes=num_classes, ignore_background=True)
    optimizer_name = hyper_cfg.get("optimizer", "sgd").lower()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(hyper_cfg["learning_rate"]),
            momentum=float(hyper_cfg["momentum"]),
            weight_decay=float(hyper_cfg["weight_decay"]),
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(hyper_cfg["learning_rate"]),
            weight_decay=float(hyper_cfg["weight_decay"]),
        )

    logger = setup_file_logger(run.run_dir)
    ckpt_mgr = CheckpointManager(run.checkpoints_dir)

    logger.info(f"Experiment: {run.name}")
    logger.info(f"Run dir: {run.run_dir}")

    split_loaders = {"train": train_loader, "val": val_loader}

    with CsvMetricsLogger(run.history_path) as metrics_logger, ClassMetricsLogger(
        run.run_dir / "class_metrics.csv"
    ) as class_metrics_logger:
        logger.info(f"Device: {device}")
        logger.info("Model: AttentionGateUnet")
        logger.info(f"Train slices: {len(train_set)}, Val slices: {len(val_set)}")
        if class_metrics_every > 0:
            logger.info(
                f"Class metrics every {class_metrics_every} epochs on splits: {class_metrics_splits}"
            )

        for epoch in range(1, num_epochs + 1):
            start = time.time()
            train_loss, train_dice = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_dice = validate(model, val_loader, criterion, device)
            elapsed = time.time() - start
            metrics_logger.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_dice": train_dice,
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "seconds": round(elapsed, 2),
                }
            )
            logger.info(
                f"Epoch [{epoch:03d}/{num_epochs}] "
                f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} "
                f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} "
                f"time={elapsed:.1f}s"
            )

            state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_dice,
                "config": cfg,
            }
            ckpt_mgr.save_last(epoch, state)
            ckpt_mgr.maybe_save_best(val_dice, epoch, state)

            if should_log_class_metrics(epoch, num_epochs, class_metrics_every):
                for split in class_metrics_splits:
                    loader = split_loaders.get(split)
                    if loader is None:
                        continue
                    class_dice = compute_per_class_dice(
                        model, loader, device, metric_class_ids
                    )
                    log_class_metrics(
                        epoch, split, class_dice, class_metrics_logger, logger
                    )

    update_training_summary(
        run.summary_path,
        status="completed",
        best_val_dice=ckpt_mgr.best_metric,
        best_epoch=ckpt_mgr.best_epoch,
        completed_epochs=num_epochs,
    )
    logger.info(f"Best val dice: {ckpt_mgr.best_metric:.4f}")
    logger.info(f"Best checkpoint: {ckpt_mgr.best_path}")
    logger.info(f"Last checkpoint: {ckpt_mgr.last_path}")


if __name__ == "__main__":
    main()
