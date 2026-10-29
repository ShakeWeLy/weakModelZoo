"""SVM 基线：用 SC + 6 段 FC 的上三角展平特征，做与 DFSC 相同的五折分层交叉验证。

- 数据: npz (X: [S, T+1, N, N], y: [S])，y=1 为 SMC，y=0 为 CN/NC
- 特征: 每个矩阵取上三角（不含对角线）展平后拼接
- 标准化: 在每折训练集上拟合，再应用到验证集（避免信息泄漏）
- 指标定义与 DFSC 一致: AUC / ACC / SEN(SMC 召回) / SPE(CN 召回) / BAC

用法（在 weakModelZoo 根目录执行）:
    python src/ml/svm.py --data F:/myproject/weakModelZoo/data/andi41_dfsc_train.npz
    python src/ml/svm.py --data ... --kernels linear rbf --C 1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def upper_tri_features(X: np.ndarray) -> np.ndarray:
    """X: [S, M, N, N] -> [S, M * N(N-1)/2]"""
    S, M, N, _ = X.shape
    iu = np.triu_indices(N, k=1)
    feats = np.stack([X[:, m][:, iu[0], iu[1]] for m in range(M)], axis=1)  # [S, M, E]
    feats = feats.reshape(S, -1)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray) -> dict[str, float]:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    sen = tp / (tp + fn) if (tp + fn) else 0.0
    spe = tn / (tn + fp) if (tn + fp) else 0.0
    acc = (tp + tn) / len(y_true)
    auc = roc_auc_score(y_true, score) if len(np.unique(y_true)) == 2 else float("nan")
    return {"auc": auc, "acc": acc, "sen": sen, "spe": spe, "bac": (sen + spe) / 2}


def run_cv(X: np.ndarray, y: np.ndarray, kernel: str, C: float, folds: int, seed: int) -> list[dict]:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    out = []
    for k, (tr, va) in enumerate(skf.split(X, y), start=1):
        clf = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=C, gamma="scale"))
        clf.fit(X[tr], y[tr])
        score = clf.decision_function(X[va])
        pred = (score > 0).astype(int)
        m = metrics(y[va], pred, score)
        out.append(m)
        print(f"  [{kernel} fold {k}] n_val={len(va)} " +
              " ".join(f"{key.upper()} {val:.4f}" for key, val in m.items()), flush=True)
    return out


def summarize(results: list[dict]) -> str:
    keys = ["auc", "acc", "sen", "spe", "bac"]
    parts = []
    for key in keys:
        vals = np.array([r[key] for r in results], dtype=float)
        parts.append(f"{key.upper()} {np.nanmean(vals):.4f}±{np.nanstd(vals):.4f}")
    return " | ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--kernels", nargs="+", default=["linear", "rbf"])
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X_raw, y = d["X"], d["y"].astype(int)
    X = upper_tri_features(X_raw)
    print(f"数据: X_raw {X_raw.shape} -> 特征 {X.shape}; y 计数 {np.bincount(y).tolist()} "
          f"(0=CN/NC, 1=SMC)", flush=True)

    for kernel in args.kernels:
        print(f"== SVM kernel={kernel} C={args.C} folds={args.folds} seed={args.seed}", flush=True)
        res = run_cv(X, y, kernel, args.C, args.folds, args.seed)
        print(f"五折结果 (mean±std): {summarize(res)}", flush=True)


if __name__ == "__main__":
    main()
