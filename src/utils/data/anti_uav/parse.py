"""解析 Anti-UAV 序列：视频/帧目录 + JSON → 帧级框列表。

支持的常见布局::

    root/
      train|val|test/                 # 可选
        seq_name/
          IR/ | ir/ | RGB/ | visible/ # 帧目录（jpg/png）
          IR.json | IR_label.json | visible.json | ...
          # 或 IR.mp4 / visible.mp4 + 同名 json

JSON 字段：``gt_rect`` 为 ``[x,y,w,h]`` 列表，``exist`` 为 0/1 列表。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal, Sequence

import cv2
import numpy as np

Modality = Literal["ir", "rgb", "auto"]

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

# (模态偏好关键字, json 名候选)
_MODALITY_HINTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "ir": (("ir", "thermal"), ("IR.json", "IR_label.json", "ir.json", "label.json")),
    "rgb": (("rgb", "visible", "vis"), ("visible.json", "RGB.json", "rgb.json", "visible_label.json")),
}


@dataclass
class FrameAnno:
    """单帧标注。"""

    index: int
    bbox_xywh: np.ndarray  # (4,) float32；不存在时为 NaN
    exist: bool


@dataclass
class SequenceMeta:
    """一条 Anti-UAV 序列的元信息。"""

    name: str
    root: Path
    modality: str
    frames: list[Path] | None = None  # 帧文件列表；None 表示从 video 读
    video: Path | None = None
    annos: list[FrameAnno] = field(default_factory=list)

    @property
    def valid_indices(self) -> list[int]:
        return [a.index for a in self.annos if a.exist]


def _is_valid_box(box: Sequence[float]) -> bool:
    if len(box) < 4:
        return False
    x, y, w, h = map(float, box[:4])
    if w <= 1 or h <= 1:
        return False
    if x < -1 and y < -1 and w < -1 and h < -1:
        return False
    return True


def load_json_annos(json_path: Path) -> list[FrameAnno]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rects = data.get("gt_rect") or data.get("gt_rects") or []
    exists = data.get("exist")
    if exists is None:
        exists = [1 if _is_valid_box(r) else 0 for r in rects]
    if len(exists) != len(rects):
        raise ValueError(f"{json_path}: exist 与 gt_rect 长度不一致 ({len(exists)} vs {len(rects)})")

    annos: list[FrameAnno] = []
    for i, (box, ex) in enumerate(zip(rects, exists)):
        exist = bool(int(ex)) and _is_valid_box(box)
        bbox = np.array(box[:4], dtype=np.float32) if exist else np.full(4, np.nan, dtype=np.float32)
        annos.append(FrameAnno(index=i, bbox_xywh=bbox, exist=exist))
    return annos


def _list_images(folder: Path) -> list[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]
    return sorted(files, key=lambda p: p.name)


def _pick_json(seq_dir: Path, modality: Modality) -> Path | None:
    jsons = list(seq_dir.glob("*.json"))
    if not jsons:
        return None
    order = []
    if modality in ("ir", "auto"):
        order += list(_MODALITY_HINTS["ir"][1])
    if modality in ("rgb", "auto"):
        order += list(_MODALITY_HINTS["rgb"][1])
    by_name = {p.name: p for p in jsons}
    for name in order:
        if name in by_name:
            return by_name[name]
    # 回退：名字里含 ir / visible 的
    for p in jsons:
        low = p.stem.lower()
        if modality == "ir" and "ir" in low:
            return p
        if modality == "rgb" and ("vis" in low or "rgb" in low):
            return p
    return jsons[0] if modality == "auto" else None


def _pick_media(seq_dir: Path, modality: Modality, json_path: Path) -> tuple[list[Path] | None, Path | None, str]:
    """返回 (frames, video, resolved_modality)。"""
    # 1) 与 json stem 同名的目录 / 视频
    stem = json_path.stem.replace("_label", "")
    cand_dirs = [seq_dir / stem, seq_dir / stem.upper(), seq_dir / stem.lower()]
    for d in cand_dirs:
        if d.is_dir():
            imgs = _list_images(d)
            if imgs:
                mod = "ir" if "ir" in d.name.lower() else "rgb"
                return imgs, None, mod

    # 2) 常见目录名
    names = []
    if modality in ("ir", "auto"):
        names += ["IR", "ir", "thermal"]
    if modality in ("rgb", "auto"):
        names += ["RGB", "rgb", "visible", "Visible", "vis"]
    for name in names:
        d = seq_dir / name
        if d.is_dir():
            imgs = _list_images(d)
            if imgs:
                mod = "ir" if name.lower() in {"ir", "thermal"} else "rgb"
                return imgs, None, mod

    # 3) 视频文件
    for name in [stem, stem.upper(), stem.lower(), "IR", "visible", "RGB"]:
        for ext in _VIDEO_EXTS:
            vp = seq_dir / f"{name}{ext}"
            if vp.is_file():
                mod = "ir" if "ir" in name.lower() else "rgb"
                return None, vp, mod

    # 4) 序列目录下直接放图
    imgs = _list_images(seq_dir)
    if imgs:
        return imgs, None, "ir" if modality == "ir" else "rgb"

    raise FileNotFoundError(f"序列 {seq_dir} 找不到帧目录或视频（json={json_path.name}）")


def discover_sequences(
    root: str | Path,
    modality: Modality = "ir",
    splits: Sequence[str] | None = None,
) -> list[SequenceMeta]:
    """扫描 ``root`` 下所有可用序列。

    ``splits`` 若给定（如 ``("train",)``），只扫描这些子目录；否则扫描 root 本身及
    常见 ``train/val/test`` 子目录中存在的部分。
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Anti-UAV 根目录不存在: {root}")

    if splits:
        search_roots = [root / s for s in splits if (root / s).is_dir()]
    else:
        search_roots = []
        for s in ("train", "val", "test", "Train", "Val", "Test"):
            if (root / s).is_dir():
                search_roots.append(root / s)
        if not search_roots:
            search_roots = [root]

    sequences: list[SequenceMeta] = []
    for base in search_roots:
        for seq_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            json_path = _pick_json(seq_dir, modality)
            if json_path is None:
                continue
            try:
                frames, video, mod = _pick_media(seq_dir, modality, json_path)
            except FileNotFoundError:
                continue
            if modality != "auto" and mod != modality:
                # 严格模态时跳过不匹配
                if modality == "ir" and mod != "ir":
                    continue
                if modality == "rgb" and mod != "rgb":
                    continue
            annos = load_json_annos(json_path)
            if frames is not None and len(frames) != len(annos):
                n = min(len(frames), len(annos))
                frames = frames[:n]
                annos = annos[:n]
            sequences.append(
                SequenceMeta(
                    name=seq_dir.name,
                    root=seq_dir,
                    modality=mod,
                    frames=frames,
                    video=video,
                    annos=annos,
                )
            )
    return sequences


def read_frame(seq: SequenceMeta, index: int) -> np.ndarray:
    """读取一帧，返回 RGB ``uint8``（单通道 IR 会复制为 3 通道）。"""
    if seq.frames is not None:
        raw = cv2.imread(str(seq.frames[index]), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(seq.frames[index])
    elif seq.video is not None:
        cap = cv2.VideoCapture(str(seq.video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, raw = cap.read()
        cap.release()
        if not ok or raw is None:
            raise RuntimeError(f"无法从 {seq.video} 读取第 {index} 帧")
    else:
        raise RuntimeError(f"序列 {seq.name} 无帧源")

    if raw.ndim == 2:
        return np.stack([raw, raw, raw], axis=-1)
    if raw.shape[2] == 4:
        raw = raw[:, :, :3]
    return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)


def iter_valid_pairs(
    seq: SequenceMeta,
    max_frame_gap: int = 100,
) -> Iterator[tuple[int, int]]:
    """同序列内有效帧对 ``(template_idx, search_idx)``。"""
    valid = seq.valid_indices
    for i, ti in enumerate(valid):
        for sj in valid[i:]:
            if sj - ti > max_frame_gap:
                break
            yield ti, sj
