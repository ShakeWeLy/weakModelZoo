"""锚框匹配分析（不训练）：训练集目标尺寸分布、当前 COCO 锚框的覆盖率，以及 k-means 锚框的对比。

目标尺寸按 640 letterbox 画布换算：原图宽高 (W, H)，缩放 s = 640 / max(W, H)，
画布上的宽 = bw·W·s，高 = bh·H·s（与 YOLO 标签归一化方式一致）。
匹配规则与损失一致（anchor_t = 4）：目标与锚框的宽比、高比都在 [1/4, 4] 内即匹配。

用法（仓库根目录）：python experiments/detection/project2_2611/anchor_analysis.py
输出：控制台表格（不写文件）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "Det_Fly"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
ANCHOR_T = 4.0
COCO = np.array([[10, 13], [16, 30], [33, 23],          # P3
                 [30, 61], [62, 45], [59, 119],          # P4
                 [116, 90], [156, 198], [373, 326]],     # P5
                dtype=float)


def load_boxes(split: str = "train") -> np.ndarray:
    lbl_dir, img_dir = DATA / "labels" / split, DATA / "images" / split
    out = []
    for lf in sorted(lbl_dir.glob("*.txt")):
        ip = next((img_dir / (lf.stem + e) for e in IMG_EXTS if (img_dir / (lf.stem + e)).exists()), None)
        if ip is None:
            continue
        W, H = Image.open(ip).size
        s = 640.0 / max(W, H)
        for line in lf.read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                out.append((float(p[3]) * W * s, float(p[4]) * H * s))
    return np.array(out, dtype=float)


def shape_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    inter = np.minimum(a[:, None, 0], b[None, :, 0]) * np.minimum(a[:, None, 1], b[None, :, 1])
    union = a[:, None, 0] * a[:, None, 1] + b[None, :, 0] * b[None, :, 1] - inter
    return inter / union


def matched(wh: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    r = wh[:, None, :] / anchors[None, :, :]
    worst = np.maximum(r, 1.0 / r).max(axis=2)
    return worst < ANCHOR_T


def kmeans(wh: np.ndarray, k: int = 9, iters: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = wh[rng.choice(len(wh), k, replace=False)].copy()
    for _ in range(iters):
        a = shape_iou(wh, c).argmax(axis=1)
        new = np.array([wh[a == j].mean(axis=0) if np.any(a == j) else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    return c[np.argsort(c.prod(axis=1))]


def report(name: str, wh: np.ndarray, anchors: np.ndarray) -> None:
    m = matched(wh, anchors)
    best_iou = shape_iou(wh, anchors).max(axis=1)
    print(f"{name}：BPR（至少一个锚框匹配）= {100 * m.any(axis=1).mean():.2f}%，"
          f"平均最佳 IoU = {best_iou.mean():.3f}，IoU≥0.5 的比例 = {100 * (best_iou >= 0.5).mean():.1f}%")


def main() -> None:
    # 本地 data/Det_Fly 只有 test 标签完整（train 仅 14 张，完整训练集在服务器上），用 test 作为分布代理
    wh = load_boxes("test")
    side = np.sqrt(wh.prod(axis=1))
    print(f"test 目标数 {len(wh)}（分布代理；画布 640 像素坐标）")
    print(f"目标几何均值边长：5% {np.percentile(side, 5):.1f} px，中位 {np.median(side):.1f} px，"
          f"95% {np.percentile(side, 95):.1f} px")
    print(f"<24 px 的比例（几何均值边长）：{100 * np.mean(side < 24):.1f}%；<16 px：{100 * np.mean(side < 16):.1f}%\n")

    report("当前 COCO 锚框（9 个）", wh, COCO)
    m = matched(wh, COCO)
    for name, sl in [("P3", slice(0, 3)), ("P4", slice(3, 6)), ("P5", slice(6, 9))]:
        print(f"  匹配到 {name} 锚框的目标比例：{100 * m[:, sl].any(axis=1).mean():.1f}%")
    miss = wh[~m.any(axis=1)]
    print(f"  无锚框匹配的目标：{len(miss)} 个；宽/高中位数 {np.median(miss[:, 0]):.1f} / {np.median(miss[:, 1]):.1f} px，"
          f"最长边中位数 {np.median(miss.max(axis=1)):.1f} px，最长边 <4 px 的比例 {100 * np.mean(miss.max(axis=1) < 4):.0f}%")
    print(f"  全部目标最长边中位数 {np.median(wh.max(axis=1)):.1f} px，宽/高比中位数 {np.median(np.maximum(wh[:,0]/wh[:,1], wh[:,1]/wh[:,0])):.2f}")

    km = kmeans(wh, k=9)
    report("k-means 锚框（9 个）", wh, km)
    print("  k-means 锚框（宽×高，画布 px）：" + "，".join(f"{w:.0f}×{h:.0f}" for w, h in km))


if __name__ == "__main__":
    main()
