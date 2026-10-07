"""在 Synapse 2D 切片上训练 PSC-UNet，使用 Dice Loss + Cross Entropy。

用法（在项目根目录执行）：
    python experiments/segment/project1_2609/PSC-UNet/train_psc_unet.py
    python experiments/segment/project1_2609/PSC-UNet/train_psc_unet.py --config experiments/segment/project1_2609/PSC-UNet/config.toml
    python experiments/segment/project1_2609/PSC-UNet/train_psc_unet.py --epochs 1 --quick
"""

from __future__ import annotations
 
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

EXP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXP_DIR.parent
ROOT = EXP_DIR.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT_DIR))

from synapse_dataloader import (
    build_train_dataloader,
    build_train_val_datasets,
    get_dataset_format,
    validate_image_size,
)
from src.utils.data.synapse.labels import (
    ALL_METRIC_CLASS_IDS,
    EVAL_CLASS_IDS,
    EVAL_MODEL_CLASS_IDS,
    EVAL_MODEL_CLASS_NUM,
    LABEL_NAMES,
    eval_model_class_name,
    remap_to_eval_model_torch,
    resolve_eval_label_map,
)
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


def load_psc_unet_class():
    module_path = ROOT / "src" / "models" / "segment" / "PSC-UNet" / "PSC-UNet.py"
    spec = importlib.util.spec_from_file_location("psc_unet_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 PSC-UNet 模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PSC_UNet


PSC_UNet = load_psc_unet_class()


class DiceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        ignore_background: bool = True,
        eps: float = 1e-6,
        class_weights: torch.Tensor | None = None,
        skip_absent_classes: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_background = ignore_background
        self.eps = eps
        self.skip_absent_classes = skip_absent_classes
        if class_weights is not None:
            if class_weights.numel() != num_classes - (1 if ignore_background else 0):
                raise ValueError(
                    f"class_weights 长度应为 {num_classes - (1 if ignore_background else 0)}，"
                    f"当前为 {class_weights.numel()}"
                )
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        gt_per_class = targets_one_hot.sum(dim=dims)
        if self.ignore_background:
            dice = dice[1:]
            gt_per_class = gt_per_class[1:]
        loss = 1.0 - dice

        if self.skip_absent_classes:
            present = gt_per_class > 0
            if not present.any():
                return logits.sum() * 0.0
            loss = loss[present]
            if self.class_weights is None:
                return loss.mean()
            weights = self.class_weights.to(loss.device)[present]
            return (loss * weights).sum() / weights.sum()

        if self.class_weights is None:
            return loss.mean()
        weights = self.class_weights.to(loss.device)
        return (loss * weights).sum() / weights.sum()


class DiceCELoss(nn.Module):
    """Dice Loss + Cross Entropy 组合损失。"""

    def __init__(
        self,
        num_classes: int,
        ignore_background: bool = True,
        eps: float = 1e-6,
        class_weights: torch.Tensor | None = None,
        ce_class_weights: torch.Tensor | None = None,
        skip_absent_classes: bool = True,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss(
            num_classes=num_classes,
            ignore_background=ignore_background,
            eps=eps,
            class_weights=class_weights,
            skip_absent_classes=skip_absent_classes,
        )
        self.ce_loss = nn.CrossEntropyLoss(weight=ce_class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(logits, targets)
        ce = self.ce_loss(logits, targets)
        return self.dice_weight * dice + self.ce_weight * ce


def remap_for_eval(
    preds: torch.Tensor,
    targets: torch.Tensor,
    train_label_map: str | None,
    eval_label_map: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if eval_label_map == "eval8" and train_label_map != "eval8":
        return remap_to_eval_model_torch(preds), remap_to_eval_model_torch(targets)
    return preds, targets


@torch.no_grad()
def compute_mean_dice(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    class_ids: tuple[int, ...] | list[int] | None = None,
    ignore_background: bool = True,
    eps: float = 1e-6,
    train_label_map: str | None = None,
    eval_label_map: str | None = None,
) -> float:
    preds = torch.argmax(logits, dim=1)
    preds, targets = remap_for_eval(preds, targets, train_label_map, eval_label_map)
    if eval_label_map == "eval8" and train_label_map != "eval8":
        num_classes = EVAL_MODEL_CLASS_NUM
    if class_ids is None:
        class_ids = tuple(range(1, num_classes) if ignore_background else range(num_classes))
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


def build_dice_class_weights(
    num_classes: int,
    weight_values: list[float] | None,
) -> torch.Tensor | None:
    if not weight_values:
        return None
    expected = num_classes - 1
    if len(weight_values) != expected:
        raise ValueError(
            f"dice_class_weights 需要 {expected} 个值（class 1..{num_classes - 1}），"
            f"当前为 {len(weight_values)}"
        )
    return torch.tensor(weight_values, dtype=torch.float32)


def build_ce_class_weights(
    num_classes: int,
    dice_weights: torch.Tensor | None,
    weight_values: list[float] | None,
) -> torch.Tensor | None:
    if weight_values:
        if len(weight_values) != num_classes:
            raise ValueError(
                f"ce_class_weights 需要 {num_classes} 个值（class 0..{num_classes - 1}），"
                f"当前为 {len(weight_values)}"
            )
        return torch.tensor(weight_values, dtype=torch.float32)
    if dice_weights is not None:
        background = torch.tensor([1.0], dtype=torch.float32)
        return torch.cat([background, dice_weights])
    return None


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


def build_lr_scheduler(
    hyper_cfg: dict,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
):
    scheduler_name = str(hyper_cfg.get("lr_scheduler", "")).lower()
    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(hyper_cfg.get("lr_step_size", 50)),
            gamma=float(hyper_cfg.get("lr_gamma", 0.5)),
        )
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(hyper_cfg.get("lr_cosine_t_max", num_epochs)),
            eta_min=float(hyper_cfg.get("lr_min", 1e-6)),
        )
    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(hyper_cfg.get("lr_plateau_factor", 0.5)),
            patience=int(hyper_cfg.get("lr_plateau_patience", 15)),
            min_lr=float(hyper_cfg.get("lr_plateau_min_lr", 1e-6)),
        )
    return None


def describe_lr_scheduler(hyper_cfg: dict, num_epochs: int) -> str:
    scheduler_name = str(hyper_cfg.get("lr_scheduler", "")).lower()
    if scheduler_name == "step":
        return (
            f"StepLR(step_size={hyper_cfg.get('lr_step_size', 50)}, "
            f"gamma={hyper_cfg.get('lr_gamma', 0.5)})"
        )
    if scheduler_name == "cosine":
        return (
            f"CosineAnnealingLR(T_max={hyper_cfg.get('lr_cosine_t_max', num_epochs)}, "
            f"eta_min={hyper_cfg.get('lr_min', 1e-6)})"
        )
    if scheduler_name == "plateau":
        return (
            f"ReduceLROnPlateau(mode=max, factor={hyper_cfg.get('lr_plateau_factor', 0.5)}, "
            f"patience={hyper_cfg.get('lr_plateau_patience', 15)}, "
            f"min_lr={hyper_cfg.get('lr_plateau_min_lr', 1e-6)})"
        )
    return "none"


def step_lr_scheduler(scheduler, metric: float | None = None) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        if metric is None:
            raise ValueError("ReduceLROnPlateau 需要传入 val 指标")
        scheduler.step(metric)
    else:
        scheduler.step()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: DiceCELoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    val_metric_class_ids: tuple[int, ...],
    train_label_map: str | None = None,
    eval_label_map: str | None = None,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_eval_dice = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_dice += compute_mean_dice(
            logits, labels, criterion.num_classes, train_label_map=train_label_map
        )
        total_eval_dice += compute_mean_dice(
            logits,
            labels,
            criterion.num_classes,
            class_ids=val_metric_class_ids,
            train_label_map=train_label_map,
            eval_label_map=eval_label_map,
        )
    count = len(loader)
    return total_loss / count, total_dice / count, total_eval_dice / count


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: DiceCELoss,
    device: torch.device,
    val_metric_class_ids: tuple[int, ...],
    train_label_map: str | None = None,
    eval_label_map: str | None = None,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_eval_dice = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        total_dice += compute_mean_dice(
            logits, labels, criterion.num_classes, train_label_map=train_label_map
        )
        total_eval_dice += compute_mean_dice(
            logits,
            labels,
            criterion.num_classes,
            class_ids=val_metric_class_ids,
            train_label_map=train_label_map,
            eval_label_map=eval_label_map,
        )
    count = len(loader)
    return total_loss / count, total_dice / count, total_eval_dice / count


@torch.no_grad()
def compute_per_class_dice(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_ids: tuple[int, ...] | list[int],
    eps: float = 1e-6,
    train_label_map: str | None = None,
    eval_label_map: str | None = None,
) -> dict[int, float | None]:
    model.eval()
    intersection = {class_id: 0.0 for class_id in class_ids}
    union = {class_id: 0.0 for class_id in class_ids}
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        preds = torch.argmax(model(images), dim=1)
        preds, labels = remap_for_eval(preds, labels, train_label_map, eval_label_map)
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
    label_map: str | None = None,
) -> None:
    rows = []
    parts = []
    for class_id in sorted(class_dice):
        dice = class_dice[class_id]
        if label_map == "eval8":
            class_name = eval_model_class_name(class_id)
        else:
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

    num_epochs = int(args.epochs or hyper_cfg["num_epochs"])
    hyper_cfg["num_epochs"] = num_epochs
    if args.quick:
        model_cfg["base_dim"] = 16
        dataset_cfg["batch_size"] = 1
        model_cfg["swin_depths"] = [1, 1, 1, 1]

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
    base_dim = int(model_cfg["base_dim"])
    in_channels = int(model_cfg.get("in_channels", 1))
    repeat_gray_to_rgb = bool(model_cfg.get("repeat_gray_to_rgb", False))
    label_map = model_cfg.get("label_map")
    eval_label_map = resolve_eval_label_map(label_map, train_cfg.get("eval_label_map"))
    swin_depths = tuple(int(v) for v in model_cfg.get("swin_depths", [2, 2, 2, 2]))
    class_metrics_every = int(train_cfg.get("class_metrics_every", 0))
    class_metrics_splits = list(train_cfg.get("class_metrics_splits", ["val"]))
    if eval_label_map == "eval8":
        default_metric_class_ids = EVAL_MODEL_CLASS_IDS
        default_val_metric_class_ids = EVAL_MODEL_CLASS_IDS
    elif label_map == "eval8":
        default_metric_class_ids = EVAL_MODEL_CLASS_IDS
        default_val_metric_class_ids = EVAL_MODEL_CLASS_IDS
    else:
        default_metric_class_ids = ALL_METRIC_CLASS_IDS
        default_val_metric_class_ids = EVAL_CLASS_IDS
    metric_class_ids = tuple(
        int(v) for v in train_cfg.get("metric_class_ids", default_metric_class_ids)
    )
    val_metric_class_ids = tuple(
        int(v) for v in train_cfg.get("val_metric_class_ids", default_val_metric_class_ids)
    )
    dice_class_weights = build_dice_class_weights(
        num_classes,
        train_cfg.get("dice_class_weights"),
    )
    ce_class_weights = build_ce_class_weights(
        num_classes,
        dice_class_weights,
        train_cfg.get("ce_class_weights"),
    )
    dice_loss_weight = float(train_cfg.get("dice_loss_weight", 1.0))
    ce_loss_weight = float(train_cfg.get("ce_loss_weight", 1.0))
    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 0))
    early_stopping_min_delta = float(train_cfg.get("early_stopping_min_delta", 0.0))

    if repeat_gray_to_rgb:
        in_channels = 3

    validate_image_size(image_size)
    model_dropout = float(model_cfg.get("dropout", 0.0))

    train_set, val_set = build_train_val_datasets(dataset_cfg, model_cfg, train_cfg)
    augment_config = getattr(train_set, "augment_config", None)
    train_loader, train_sampler = build_train_dataloader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        train_cfg=train_cfg,
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
        dropout=model_dropout,
    ).to(device)
    criterion = DiceCELoss(
        num_classes=num_classes,
        ignore_background=True,
        class_weights=dice_class_weights,
        ce_class_weights=ce_class_weights,
        skip_absent_classes=bool(train_cfg.get("skip_absent_classes", True)),
        dice_weight=dice_loss_weight,
        ce_weight=ce_loss_weight,
    ).to(device)
    optimizer = build_optimizer(hyper_cfg, model)
    scheduler = build_lr_scheduler(hyper_cfg, optimizer, num_epochs)

    logger = setup_file_logger(run.run_dir)
    ckpt_mgr = CheckpointManager(run.checkpoints_dir)
    split_loaders = {"train": train_loader, "val": val_loader}

    logger.info(f"Experiment: {run.name}")
    logger.info(f"Run dir: {run.run_dir}")

    with CsvMetricsLogger(run.history_path) as metrics_logger, ClassMetricsLogger(
        run.run_dir / "class_metrics.csv"
    ) as class_metrics_logger:
        logger.info(f"Device: {device}")
        logger.info(f"Train slices: {len(train_set)}, Val slices: {len(val_set)}")
        logger.info(
            f"Model: image_size={image_size}, in_channels={in_channels}, base_dim={base_dim}, "
            f"swin_depths={swin_depths}, dropout={model_dropout}, "
            f"repeat_gray_to_rgb={repeat_gray_to_rgb}, "
            f"class_nums={num_classes}, label_map={label_map or 'none'}, "
            f"eval_label_map={eval_label_map or 'none'}"
        )
        data_format = get_dataset_format(dataset_cfg, train_cfg)
        logger.info(f"Dataset format: {data_format}")
        if data_format == "transunet" and dataset_cfg.get("augment", True):
            logger.info("Augmentation: TransUNet RandomGenerator (rot/flip + zoom)")
        elif augment_config is not None and augment_config.enabled:
            logger.info(
                "Augmentation: flip_h=%s flip_v=%s rotate=%s intensity_jitter=%s "
                "scale=%s shift=%s noise_std=%s"
                % (
                    augment_config.flip_horizontal,
                    augment_config.flip_vertical,
                    augment_config.rotate,
                    augment_config.intensity_jitter,
                    augment_config.intensity_scale,
                    augment_config.intensity_shift,
                    augment_config.gaussian_noise_std,
                )
            )
        if class_metrics_every > 0:
            logger.info(
                f"Class metrics every {class_metrics_every} epochs on splits: {class_metrics_splits}"
            )
        logger.info(f"Val metric classes (best/early-stop): {list(val_metric_class_ids)}")
        logger.info(
            f"Loss: DiceCE (dice_weight={dice_loss_weight}, ce_weight={ce_loss_weight})"
        )
        if dice_class_weights is not None:
            logger.info(f"Dice class weights (1..{num_classes - 1}): {dice_class_weights.tolist()}")
        if ce_class_weights is not None:
            logger.info(f"CE class weights (0..{num_classes - 1}): {ce_class_weights.tolist()}")
        if early_stopping_patience > 0:
            logger.info(
                f"Early stopping: patience={early_stopping_patience}, "
                f"min_delta={early_stopping_min_delta}"
            )
        logger.info(
            f"Optimizer: {hyper_cfg.get('optimizer', 'adam')} | "
            f"LR scheduler: {describe_lr_scheduler(hyper_cfg, num_epochs)}"
        )
        logger.info(
            f"skip_absent_classes={bool(train_cfg.get('skip_absent_classes', True))}"
        )
        if train_sampler is not None:
            logger.info(
                "Guaranteed sampling enabled: "
                + ", ".join(
                    f"{name}={size}"
                    for name, size in zip(
                        train_sampler.group_names, train_sampler.pool_sizes
                    )
                )
            )
            logger.info(
                f"Min samples per group per epoch: "
                f"{train_sampler.min_samples_per_group}"
            )
            if label_map == "eval8":
                logger.info(
                    "注意: eval8 下肾上腺(12/13)会被映射为背景，"
                    "但仍会保证含肾上腺的切片进入训练。"
                )

        epochs_without_improve = 0
        completed_epochs = 0
        for epoch in range(1, num_epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            start = time.time()
            train_loss, train_dice, train_eval_dice = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                val_metric_class_ids,
                train_label_map=label_map,
                eval_label_map=eval_label_map,
            )
            val_loss, val_dice, val_eval_dice = validate(
                model,
                val_loader,
                criterion,
                device,
                val_metric_class_ids,
                train_label_map=label_map,
                eval_label_map=eval_label_map,
            )
            elapsed = time.time() - start
            completed_epochs = epoch
            metrics_logger.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_dice": train_dice,
                    "train_eval_dice": train_eval_dice,
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "val_eval_dice": val_eval_dice,
                    "seconds": round(elapsed, 2),
                }
            )
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch [{epoch:03d}/{num_epochs}] "
                f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} "
                f"train_eval_dice={train_eval_dice:.4f} "
                f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} "
                f"val_eval_dice={val_eval_dice:.4f} "
                f"lr={current_lr:.6f} time={elapsed:.1f}s"
            )
            step_lr_scheduler(scheduler, val_eval_dice)

            state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_eval_dice,
                "val_dice_all": val_dice,
                "config": cfg,
            }
            ckpt_mgr.save_last(epoch, state)
            if val_eval_dice > ckpt_mgr.best_metric + early_stopping_min_delta:
                ckpt_mgr.maybe_save_best(val_eval_dice, epoch, state)
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            if should_log_class_metrics(epoch, num_epochs, class_metrics_every):
                for split in class_metrics_splits:
                    loader = split_loaders.get(split)
                    if loader is None:
                        continue
                    class_dice = compute_per_class_dice(
                        model,
                        loader,
                        device,
                        metric_class_ids,
                        train_label_map=label_map,
                        eval_label_map=eval_label_map,
                    )
                    log_class_metrics(
                        epoch,
                        split,
                        class_dice,
                        class_metrics_logger,
                        logger,
                        eval_label_map or label_map,
                    )

            if (
                early_stopping_patience > 0
                and epochs_without_improve >= early_stopping_patience
            ):
                logger.info(
                    f"Early stopping at epoch {epoch}: "
                    f"val_eval_dice 连续 {epochs_without_improve} epoch 未提升"
                )
                break

    stop_status = "completed"
    if early_stopping_patience > 0 and completed_epochs < num_epochs:
        stop_status = "early_stopped"

    update_training_summary(
        run.summary_path,
        status=stop_status,
        best_val_dice=ckpt_mgr.best_metric,
        best_epoch=ckpt_mgr.best_epoch,
        completed_epochs=completed_epochs,
    )
    logger.info(f"Best val dice: {ckpt_mgr.best_metric:.4f}")
    logger.info(f"Best checkpoint: {ckpt_mgr.best_path}")
    logger.info(f"Last checkpoint: {ckpt_mgr.last_path}")


if __name__ == "__main__":
    main()
