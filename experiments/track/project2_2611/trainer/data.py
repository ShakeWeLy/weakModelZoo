"""Anti-UAV 孪生对 DataLoader 构建。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from torch.utils.data import DataLoader

from src.utils.data.anti_uav import AntiUAVPairDataset, collate_anti_uav

from .config import ROOT


def resolve_data_dir(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _as_splits(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = tuple(s.strip() for s in value.split(",") if s.strip())
        return parts or None
    if isinstance(value, Sequence):
        parts = tuple(str(s) for s in value)
        return parts or None
    raise TypeError(f"splits 需要 str 或 list，当前 {type(value)}")


def build_pair_dataset(cfg: Mapping[str, Any], *, splits_key: str = "splits") -> AntiUAVPairDataset:
    ds = cfg["dataset"]
    root = resolve_data_dir(str(ds["data_dir"]))
    if not root.is_dir():
        raise FileNotFoundError(f"Anti-UAV 数据目录不存在: {root}")
    return AntiUAVPairDataset(
        root,
        modality=str(ds.get("modality", "ir")),  # type: ignore[arg-type]
        splits=_as_splits(ds.get(splits_key)),
        pairs_per_epoch=int(ds.get("pairs_per_epoch", 60000)),
        max_frame_gap=int(ds.get("max_frame_gap", 100)),
        exemplar_size=int(ds.get("exemplar_size", 127)),
        search_size=int(ds.get("search_size", 255)),
        context_amount=float(ds.get("context_amount", 0.5)),
        search_shift=float(ds.get("search_shift", 0.2)),
        score_size=int(ds.get("score_size", 25)),
        stride=int(ds.get("stride", 8)),
        pos_thr=float(ds.get("pos_thr", 0.6)),
        neg_thr=float(ds.get("neg_thr", 0.3)),
    )


def try_build_pair_dataset(cfg: Mapping[str, Any], *, splits_key: str) -> AntiUAVPairDataset | None:
    """验证集目录不存在或无序列时返回 None。"""
    try:
        return build_pair_dataset(cfg, splits_key=splits_key)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[data] 跳过 {splits_key}: {exc}")
        return None


def build_dataloader(
    dataset: AntiUAVPairDataset,
    cfg: Mapping[str, Any],
    *,
    shuffle: bool,
    device_type: str,
) -> DataLoader:
    ds = cfg["dataset"]
    return DataLoader(
        dataset,
        batch_size=int(ds["batch_size"]),
        shuffle=shuffle,
        num_workers=int(ds.get("num_workers", 0)),
        pin_memory=device_type == "cuda",
        collate_fn=collate_anti_uav,
        drop_last=True,
    )
