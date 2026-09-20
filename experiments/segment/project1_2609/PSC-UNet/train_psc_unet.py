"""在 Synapse 2D 切片上训练 PSC-UNet，使用 Dice Loss。

用法（在项目根目录执行）：
    python experiments/segment/project1_2609/PSC-UNet/train_psc_unet.py
    python experiments/segment/project1_2609/PSC-UNet/train_psc_unet.py --config experiments/segment/project1_2609/PSC-UNet/config.toml
    python experiments/segment/project1_2609/PSC-UNet/train_psc_unet.py --epochs 1 --quick
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
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

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore


def load_psc_unet_class():
    module_path = ROOT / "src" / "models" / "segment" / "PSC-UNet" / "PSC-UNet.py"
    spec = importlib.util.spec_from_file_location("psc_unet_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 PSC-UNet 模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PSC_UNet


PSC_UNet = load_psc_unet_class()


class SynapseSliceDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        split: str,
        image_size: int,
        num_classes: int,
        repeat_gray_to_rgb: bool = False,
    ):
        self.image_dir = data_dir / split / "images"
        self.label_dir = data_dir / split / "labels"
        self.image_size = image_size
        self.num_classes = num_classes
        self.repeat_gray_to_rgb = repeat_gray_to_rgb
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

        if self.repeat_gray_to_rgb:
            image = image.repeat(3, 1, 1)
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


def validate_image_size(image_size: int, patch_size: int = 4, window_size: int = 7) -> None:
    if image_size % 32 != 0:
        raise ValueError(
            f"PSC-UNet 要求 image_size 能被 32 整除（patch=4 + 3 次 2x 下采样），当前为 {image_size}"
        )
    if (image_size // patch_size) % window_size != 0:
        raise ValueError(
            f"PSC-UNet 要求 image_size/{patch_size} 能被 window_size={window_size} 整除，"
            f"当前 image_size={image_size}"
        )


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
    parser.add_argument("--epochs", type=int, default=None, help="覆盖 config 中的 num_epochs")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速试跑：base_dim=16, batch_size=1, swin_depths=[1,1,1,1]",
    )
    return parser.parse_args()


def build_optimizer(hyper_cfg: dict, model: nn.Module) -> torch.optim.Optimizer:
    optimizer_name = hyper_cfg.get("optimizer", "adam").lower()
    lr = float(hyper_cfg["learning_rate"])
    weight_decay = float(hyper_cfg["weight_decay"])
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(hyper_cfg["momentum"]),
            weight_decay=weight_decay,
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


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


def main():
    args = parse_args()
    cfg = load_config(args.config)
    dataset_cfg = cfg["dataset"]
    hyper_cfg = cfg["hyperparameters"]
    model_cfg = cfg["model"]
    train_cfg = cfg.get("train", {})

    data_dir = ROOT / train_cfg.get("data_dir", "data/synapse_processed")
    output_dir = ROOT / train_cfg.get("output_dir", str(EXP_DIR.relative_to(ROOT) / "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    image_size = int(dataset_cfg["image_size"])
    batch_size = int(dataset_cfg["batch_size"])
    num_workers = int(dataset_cfg["num_workers"])
    num_classes = int(model_cfg["class_nums"])
    base_dim = int(model_cfg["base_dim"])
    in_channels = int(model_cfg.get("in_channels", 1))
    repeat_gray_to_rgb = bool(model_cfg.get("repeat_gray_to_rgb", False))
    swin_depths = tuple(int(v) for v in model_cfg.get("swin_depths", [2, 2, 2, 2]))
    num_epochs = int(args.epochs or hyper_cfg["num_epochs"])

    if args.quick:
        base_dim = 16
        batch_size = 1
        swin_depths = (1, 1, 1, 1)

    if repeat_gray_to_rgb:
        in_channels = 3

    validate_image_size(image_size)

    train_set = SynapseSliceDataset(
        data_dir, "train", image_size, num_classes, repeat_gray_to_rgb=repeat_gray_to_rgb
    )
    val_set = SynapseSliceDataset(
        data_dir, "val", image_size, num_classes, repeat_gray_to_rgb=repeat_gray_to_rgb
    )
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

    model = PSC_UNet(
        in_channels=in_channels,
        out_channels=num_classes,
        base_dim=base_dim,
        swin_depths=swin_depths,
    ).to(device)
    criterion = DiceLoss(num_classes=num_classes, ignore_background=True)
    optimizer = build_optimizer(hyper_cfg, model)

    log_path = output_dir / "train_log.csv"
    best_dice = -1.0
    best_ckpt = output_dir / "best_psc_unet.pth"
    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["epoch", "train_loss", "train_dice", "val_loss", "val_dice", "seconds"])

        print(f"Device: {device}", flush=True)
        print(f"Train slices: {len(train_set)}, Val slices: {len(val_set)}", flush=True)
        print(f"Output dir: {output_dir}", flush=True)
        print(
            f"Model: image_size={image_size}, in_channels={in_channels}, base_dim={base_dim}, "
            f"swin_depths={swin_depths}, repeat_gray_to_rgb={repeat_gray_to_rgb}",
            flush=True,
        )

        for epoch in range(1, num_epochs + 1):
            start = time.time()
            train_loss, train_dice = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_dice = validate(model, val_loader, criterion, device)
            elapsed = time.time() - start
            writer.writerow([epoch, train_loss, train_dice, val_loss, val_dice, round(elapsed, 2)])
            print(
                f"Epoch [{epoch:03d}/{num_epochs}] "
                f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} "
                f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} "
                f"time={elapsed:.1f}s",
                flush=True,
            )

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_dice": best_dice,
                        "config": cfg,
                    },
                    best_ckpt,
                )

            if epoch % int(train_cfg.get("save_every", 50)) == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_dice": val_dice,
                        "config": cfg,
                    },
                    output_dir / f"psc_unet_epoch_{epoch:03d}.pth",
                )

    print(f"Best val dice: {best_dice:.4f}")
    print(f"Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
