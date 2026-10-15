"""数据：读取 npz 或生成 mock 皮尔逊相关矩阵。

npz 格式::

    X: [S, T+1, N, N]  每个受试者 T+1 个皮尔逊相关矩阵（1 个 DTI + T 个 fMRI 时间段）
    y: [S]             标签（0 / 1，二分类）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch

from .config import ROOT


def load_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with np.load(path) as f:
        X = f["X"].astype(np.float32)
        y = f["y"].astype(np.int64)
    if X.ndim != 4 or X.shape[2] != X.shape[3]:
        raise ValueError(f"X 形状应为 [S, T+1, N, N]，实际 {X.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X 与 y 样本数不一致: {X.shape[0]} vs {y.shape[0]}")
    return X, y


def save_npz(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y)


def make_mock_dataset(
    num_subjects: int, num_matrices: int, num_nodes: int, num_timepoints: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """生成带弱类别差异的 mock 皮尔逊矩阵（仅用于跑通流程，结果不代表真实效应）。

    正类（y=1）在前 1/4 的脑区之间注入共享成分，使相关矩阵呈现可学习的差异。
    """
    rng = np.random.default_rng(seed)
    y = np.zeros(num_subjects, dtype=np.int64)
    y[: num_subjects // 2] = 1
    rng.shuffle(y)

    hub = max(2, num_nodes // 4)
    X = np.empty((num_subjects, num_matrices, num_nodes, num_nodes), dtype=np.float32)
    for s in range(num_subjects):
        for m in range(num_matrices):
            signals = rng.standard_normal((num_nodes, num_timepoints))
            if y[s] == 1:
                signals[:hub] += 0.5 * rng.standard_normal(num_timepoints)
            X[s, m] = np.corrcoef(signals).astype(np.float32)
    return X, y


def load_data(dataset_cfg: Mapping[str, Any], seed: int, run_dir: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """返回 (X, y, 来源描述)。data_path 为空时生成 mock 数据并保存到 run_dir/data/。"""
    T = int(dataset_cfg["time_segments"])
    data_path = str(dataset_cfg.get("data_path") or "").strip()
    if data_path:
        X, y = load_npz(data_path)
        source = f"npz: {data_path}"
    else:
        X, y = make_mock_dataset(
            num_subjects=int(dataset_cfg["mock_subjects"]),
            num_matrices=T + 1,
            num_nodes=int(dataset_cfg["mock_nodes"]),
            num_timepoints=int(dataset_cfg["mock_timepoints"]),
            seed=seed,
        )
        saved = run_dir / "data" / "mock_pearson.npz"
        save_npz(saved, X, y)
        source = f"mock (已保存: {saved.name})"
    if X.shape[1] != T + 1:
        raise ValueError(f"数据的矩阵数 {X.shape[1]} 与 T+1={T + 1} 不一致，请检查 dataset.time_segments")
    return X, y, source


def make_loader(X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool, seed: int = 0) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle, generator=generator)
