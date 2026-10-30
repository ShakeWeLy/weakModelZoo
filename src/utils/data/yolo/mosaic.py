"""YOLO 格式数据的 Mosaic 数据增强（图像与检测框同步变换）。

标签格式：每行 `class x_center y_center width height`，坐标均已归一化到 [0, 1]。

Mosaic 做法（与 YOLOv5 一致）：
1. 随机选 4 张图，各自把长边缩放到 img_size；
2. 在 2s x 2s 的画布上，以随机中心点 (xc, yc) 把 4 张图拼到四个象限；
3. 检测框跟随图像平移，并裁剪到各自图片的可见区域；
4. 取画布中心 s x s 区域作为输出，框再次裁剪并过滤掉太小或大部分被裁掉的框。

用法：
    # 生成 20 个增强样本（同时保存图片和标签），用于检查效果
    python src/utils/data/yolo/mosaic.py \
        --img-dir data/Det_Fly/images/train \
        --label-dir data/Det_Fly/labels/train \
        --out-dir outputs/mosaic_preview --num 20 --img-size 640
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def read_yolo_labels(label_path: Path) -> np.ndarray:
    """读取 YOLO 标签，返回形状 (N, 5) 的数组，列为 cls, xc, yc, w, h（归一化）。"""
    if not label_path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            rows.append([float(v) for v in parts])
    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _resize_long_side(image: np.ndarray, size: int) -> np.ndarray:
    """把长边缩放到 size，保持宽高比。"""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def _yolo_to_xyxy(labels: np.ndarray, w: int, h: int) -> np.ndarray:
    """归一化 (cls, xc, yc, w, h) -> 像素 (cls, x1, y1, x2, y2)。"""
    out = np.empty((len(labels), 5), dtype=np.float32)
    out[:, 0] = labels[:, 0]
    out[:, 1] = (labels[:, 1] - labels[:, 3] / 2) * w
    out[:, 2] = (labels[:, 2] - labels[:, 4] / 2) * h
    out[:, 3] = (labels[:, 1] + labels[:, 3] / 2) * w
    out[:, 4] = (labels[:, 2] + labels[:, 4] / 2) * h
    return out


def mosaic(
    images: list[np.ndarray],
    labels: list[np.ndarray],
    img_size: int = 640,
    fill: int = 114,
    min_box_px: float = 2.0,
    min_area_ratio: float = 0.2,
    rng: random.Random | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """把 4 张图拼成一张 img_size x img_size 的 mosaic。

    Args:
        images: 4 张 BGR 图像（numpy, uint8）。
        labels: 与 images 一一对应的归一化标签，每个形状 (N, 5)。
        img_size: 输出边长。
        fill: 画布填充值。
        min_box_px: 输出框的最小宽高（像素），小于该值的框被丢弃。
        min_area_ratio: 框被裁剪后保留面积比例不足该值时丢弃。
        rng: 随机数生成器，便于复现。

    Returns:
        image: (img_size, img_size, 3) 的 uint8 图像。
        boxes: (M, 5) 的归一化标签 (cls, xc, yc, w, h)，相对于输出图像。
    """
    assert len(images) == 4 and len(labels) == 4, "mosaic 需要 4 张图像"
    rng = rng or random.Random()
    s = img_size
    canvas = np.full((2 * s, 2 * s, 3), fill, dtype=np.uint8)
    xc = int(rng.uniform(0.5 * s, 1.5 * s))
    yc = int(rng.uniform(0.5 * s, 1.5 * s))

    all_boxes = []
    for i, (img, lab) in enumerate(zip(images, labels)):
        im = _resize_long_side(img, s)
        h, w = im.shape[:2]

        # 四个象限的放置逻辑（画布坐标 a，图像局部坐标 b）
        if i == 0:  # 左上
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b = w - (x2a - x1a), h - (y2a - y1a)
            x2b, y2b = w, h
        elif i == 1:  # 右上
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, 2 * s), yc
            x1b, y1b = 0, h - (y2a - y1a)
            x2b, y2b = min(w, x2a - x1a), h
        elif i == 2:  # 左下
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(2 * s, yc + h)
            x1b, y1b = w - (x2a - x1a), 0
            x2b, y2b = w, min(y2a - y1a, h)
        else:  # 右下
            x1a, y1a, x2a, y2a = xc, yc, min(2 * s, xc + w), min(2 * s, yc + h)
            x1b, y1b = 0, 0
            x2b, y2b = min(w, x2a - x1a), min(y2a - y1a, h)

        canvas[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]

        if len(lab):
            boxes = _yolo_to_xyxy(lab, w, h)
            areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])
            # 裁剪到该图在 mosaic 中可见的局部区域
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(x1b, x2b)
            boxes[:, [2, 4]] = boxes[:, [2, 4]].clip(y1b, y2b)
            new_areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])
            keep = (new_areas > 0) & (new_areas >= min_area_ratio * np.maximum(areas, 1e-6))
            boxes = boxes[keep]
            # 局部坐标 -> 画布坐标
            boxes[:, [1, 3]] += x1a - x1b
            boxes[:, [2, 4]] += y1a - y1b
            all_boxes.append(boxes)

    # 取画布中心 s x s 区域（与 YOLOv5 的 border=-s/2 一致）
    off = s // 2
    image = canvas[off:off + s, off:off + s].copy()
    boxes = np.concatenate(all_boxes, axis=0) if all_boxes else np.zeros((0, 5), np.float32)
    if len(boxes):
        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - off).clip(0, s)
        boxes[:, [2, 4]] = (boxes[:, [2, 4]] - off).clip(0, s)
        new_areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])
        keep = (
            (new_areas > 0)
            & (new_areas >= min_area_ratio * np.maximum(areas, 1e-6))
            & ((boxes[:, 3] - boxes[:, 1]) >= min_box_px)
            & ((boxes[:, 4] - boxes[:, 2]) >= min_box_px)
        )
        boxes = boxes[keep]

    out = np.zeros((len(boxes), 5), dtype=np.float32)
    if len(boxes):
        out[:, 0] = boxes[:, 0]
        out[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2 / s
        out[:, 2] = (boxes[:, 2] + boxes[:, 4]) / 2 / s
        out[:, 3] = (boxes[:, 3] - boxes[:, 1]) / s
        out[:, 4] = (boxes[:, 4] - boxes[:, 2]) / s
    return image, out


class YoloMosaicDataset:
    """每次取样时随机挑 4 张图进行 mosaic，返回 (image, boxes)。"""

    def __init__(
        self,
        img_dir: Path,
        label_dir: Path,
        img_size: int = 640,
        seed: int | None = None,
    ) -> None:
        self.img_paths = sorted(
            p for p in Path(img_dir).iterdir() if p.suffix.lower() in IMG_EXTS
        )
        if not self.img_paths:
            raise FileNotFoundError(f"{img_dir} 中没有图片")
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.img_paths)

    def _load(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        img = cv2.imread(str(path))
        if img is None:
            raise IOError(f"无法读取图片: {path}")
        lab = read_yolo_labels(self.label_dir / f"{path.stem}.txt")
        return img, lab

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        # 当前图作为第一张，其余 3 张随机抽取（数据量不足 4 张时允许重复）
        others = [p for j, p in enumerate(self.img_paths) if j != idx]
        picks = [self.img_paths[idx]] + self.rng.sample(
            others, min(3, len(others))
        )
        while len(picks) < 4:
            picks.append(self.rng.choice(self.img_paths))
        self.rng.shuffle(picks)
        loaded = [self._load(p) for p in picks]
        return mosaic(
            [x[0] for x in loaded],
            [x[1] for x in loaded],
            img_size=self.img_size,
            rng=self.rng,
        )


def _save_preview(out_dir: Path, name: str, image: np.ndarray, boxes: np.ndarray) -> None:
    h, w = image.shape[:2]
    vis = image.copy()
    for cls, xc, yc, bw, bh in boxes:
        x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
        x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(str(out_dir / f"{name}.jpg"), image)
    cv2.imwrite(str(out_dir / f"{name}_vis.jpg"), vis)
    lines = [f"{int(c)} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}" for c, x, y, bw, bh in boxes]
    (out_dir / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = PROJECT_ROOT
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--img-dir", type=Path,
                        default=root / "data" / "Det_Fly" / "images" / "train")
    parser.add_argument("--label-dir", type=Path,
                        default=root / "data" / "Det_Fly" / "labels" / "train")
    parser.add_argument("--out-dir", type=Path,
                        default=root / "outputs" / "mosaic_preview")
    parser.add_argument("--num", type=int, default=20, help="生成的样本数")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = YoloMosaicDataset(args.img_dir, args.label_dir,
                                img_size=args.img_size, seed=args.seed)
    for i in range(args.num):
        image, boxes = dataset[i % len(dataset)]
        _save_preview(args.out_dir, f"mosaic_{i:04d}", image, boxes)
    print(f"已生成 {args.num} 个 mosaic 样本 -> {args.out_dir}")


if __name__ == "__main__":
    main()
