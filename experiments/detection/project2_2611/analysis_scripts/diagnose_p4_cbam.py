"""诊断两个现象（best 权重，test 集上的小目标 / 中等目标单目标图）：

(1) P4 小目标响应被 SDGhost 压制：
    - 逐图统计 P4 下采样输出、块输入、恒等支路、各瓶颈输出，以及 Ghost 瓶颈内部（残差 DW 支路、门控 Ghost 的输入 / 输出）
      中 GT 框的响应分位（框内均值在全图格子中的分位，越高越突出）；
    - 检查压制是否与“局部对比度”（框内均值 − 外圈均值）相关，并与 YOLOv5s / c3topo_1 的瓶颈对照；
    - 统计 3×3 卷积核的权重和、中心-外圈结构，判断响应是否依赖局部对比度。
(2) CBAM（P3）“目标外圈亮、本体暗”：
    - 比较 CBAM 前、CAM 之后、SAM 注意力、CBAM 之后：框内 / 外圈 在全图中的分位（不用易受零点和符号影响的比值）；
    - SAM 7×7 核的整体增益与中心 / 外圈结构；SAM 注意力与颈部特征的秩相关（负相关 = 反向注意力）。

用法（仓库根目录）：
  CUDA_VISIBLE_DEVICES="" python experiments/detection/project2_2611/diagnose_p4_cbam.py
输出：experiments/detection/project2_2611/diagnose_p4_cbam/
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import redundancy_analysis as ra  # noqa: E402
from trainer.config import ROOT  # noqa: E402
from trainer.data import YoloDetDataset  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent / "diagnose_p4_cbam"
MODELS = {
    "ours": {"label": "本模型 (SGhost↓ + Ghost瓶颈dw5)", "model": "sgc_yolov5s_c3_sgdown_gbot_dw5",
             "run": "2026-11-13_029_SGC_c3_sgdown_gbot_dw5_s_V1", "kind": "sgc_ghost"},
    "c3topo": {"label": "c3topo_1 (Conv↓ + 稠密瓶颈)", "model": "sgc_yolov5s_c3topo_1",
               "run": "2026-11-12_021_SGC_c3topo_1_s_V1", "kind": "sgc_dense"},
    "yolo": {"label": "YOLOv5s (C3, 无CBAM)", "model": "yolov5s",
             "run": "2026-10-30_002_YOLOv5s_V3", "kind": "c3"},
}
S4, S3 = 16, 8  # P4、P3 的步长（640 输入）


# ---------------------------------------------------------------- 样本选择
def pick_samples(data_dir: Path, n_small: int = 60, n_med: int = 30, seed: int = 0):
    lbl, imd = data_dir / "labels" / "test", data_dir / "images" / "test"
    small, med = [], []
    for lf in sorted(lbl.glob("*.txt")):
        lines = [l.split() for l in lf.read_text().splitlines() if l.strip()]
        if len(lines) != 1:
            continue
        ip = next((imd / (lf.stem + e) for e in (".jpg", ".jpeg", ".png", ".bmp")
                   if (imd / (lf.stem + e)).exists()), None)
        if ip is None:
            continue
        W, H = Image.open(ip).size
        s = 640 / max(W, H)
        d = max(float(lines[0][3]) * W * s, float(lines[0][4]) * H * s)
        if 14 <= d <= 24:
            small.append(lf.stem)
        elif 40 <= d <= 70:
            med.append(lf.stem)
    rng = np.random.default_rng(seed)
    small = [str(x) for x in rng.choice(small, size=min(n_small, len(small)), replace=False)]
    med = [str(x) for x in rng.choice(med, size=min(n_med, len(med)), replace=False)]
    return small, med


# ---------------------------------------------------------------- 区域统计
def box_cells(box, s, H, W):
    x1, y1, x2, y2 = box
    r1, r2 = int(np.floor(y1 / s)), int(np.ceil(y2 / s))
    c1, c2 = int(np.floor(x1 / s)), int(np.ceil(x2 / s))
    r1, c1 = max(r1, 0), max(c1, 0)
    r2, c2 = min(max(r2, r1 + 1), H), min(max(c2, c1 + 1), W)
    return r1, r2, c1, c2


def region_stats(M: np.ndarray, box, s: int) -> dict:
    """M: (H, W) 通道均值图。
    pct：框内均值在全图格子中的分位；pct_ring：外圈均值的分位；pct_center：中心格的分位。
    box / ring / center / bg / std 为原始数值，用于对比度与背景比。
    """
    H, W = M.shape
    r1, r2, c1, c2 = box_cells(box, s, H, W)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    rc, cc = min(int(cy // s), H - 1), min(int(cx // s), W - 1)
    bmask = np.zeros((H, W), bool)
    bmask[r1:r2, c1:c2] = True
    dmask = np.zeros((H, W), bool)
    dmask[max(r1 - 1, 0):min(r2 + 1, H), max(c1 - 1, 0):min(c2 + 1, W)] = True
    rmask = dmask & ~bmask
    emask = bmask.copy()
    emask[rc, cc] = False
    box_mean = float(M[bmask].mean())
    ring_mean = float(M[rmask].mean()) if rmask.any() else float("nan")
    center = float(M[rc, cc])
    return {
        "pct": float((M <= box_mean).mean()),
        "box": box_mean,
        "center": center,
        "edge": float(M[emask].mean()) if emask.any() else float("nan"),
        "ring": ring_mean,
        "bg": float(M[~dmask].mean()) if (~dmask).any() else float("nan"),
        "std": float(M.std()),
        "pct_ring": float((M <= ring_mean).mean()) if rmask.any() else float("nan"),
        "pct_center": float((M <= center).mean()),
    }


def stats_mean_rms(A: np.ndarray, box, s: int) -> dict:
    """A: (C, H, W)。pct：通道均值图的分位；pct_rms：通道 RMS 图的分位（不受正负抵消影响）。"""
    st = region_stats(A.mean(0), box, s)
    st["pct_rms"] = region_stats(np.sqrt((A ** 2).mean(0)), box, s)["pct"]
    return st


def masks_for(box, s: int, H: int, W: int):
    r1, r2, c1, c2 = box_cells(box, s, H, W)
    bmask = np.zeros((H, W), bool)
    bmask[r1:r2, c1:c2] = True
    dmask = np.zeros((H, W), bool)
    dmask[max(r1 - 1, 0):min(r2 + 1, H), max(c1 - 1, 0):min(c2 + 1, W)] = True
    return bmask, dmask & ~bmask, dmask


def ring_dominant_frac(A: np.ndarray, box, s: int) -> float:
    """逐通道比较：外圈均值 > 框内均值 的通道比例（A: (C, H, W)）。"""
    H, W = A.shape[1:]
    bm, rm, _ = masks_for(box, s, H, W)
    body = A[:, bm].mean(axis=1)
    ring = A[:, rm].mean(axis=1)
    return float(np.mean(ring > body))


def spearman(a, b) -> float:
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    ra_ = np.argsort(np.argsort(a))
    rb_ = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra_, rb_)[0, 1])


def contrast(st: dict) -> float:
    """输入的局部对比度：（框内均值 − 外圈均值）/ 全图标准差。"""
    return (st["box"] - st["ring"]) / max(st["std"], 1e-9)


def bg_ratio(st: dict) -> float:
    return st["bg"] / max(st["box"], 1e-9)


# ---------------------------------------------------------------- 钩子名称
def bottleneck_name(kind: str, k: int) -> str:
    seq = "backbone.stage4.1.m" if kind == "c3" else "backbone.stage4.1.bottleneck"
    return f"{seq}.{k}"


def need_names(kind: str) -> list[str]:
    names = ["backbone.stage4.0", "backbone.stage4.1", "backbone.stage4.1.cv1", "backbone.stage4.1.cv2",
             "neck.c3_p3", "neck.cbam_p3", "neck.cbam_p3.channel_attention"]
    names += [bottleneck_name(kind, k) for k in range(3)]
    if kind == "sgc_ghost":
        for k in range(3):
            b = bottleneck_name(kind, k)
            names += [f"{b}.DWConv2", f"{b}.SGhostConv2", f"{b}.SGhostConv1", f"{b}.DFCAttention"]
    return names


def pre_names_for(kind: str) -> list[str]:
    """需要记录“输入”的模块：瓶颈输入（= cv1 输出）；Ghost 瓶颈还记录门控 Ghost 路径的输入 x1·x2。"""
    names = [bottleneck_name(kind, k) for k in range(3)]
    if kind == "sgc_ghost":
        names += [f"{bottleneck_name(kind, k)}.SGhostConv2" for k in range(3)]
    return names


def make_pre_hook(cap: dict, key: str):
    def _h(_m, inp):
        cap[key] = inp[0].detach().float().cpu().numpy()

    return _h


# ---------------------------------------------------------------- 卷积核统计（只看权重）
def conv_stats(conv: nn.Conv2d):
    """返回每个输出通道的：空间权重和（对输入通道取平均）、中心权重 − 外圈平均（对输入通道取平均）。
    权重和 < 0 或 中心 < 外圈：对“均匀输入”不响应或反向响应，即依赖局部对比度。"""
    w = conv.weight.detach().float().cpu().numpy()
    kh, kw = w.shape[-2:]
    center = w[:, :, kh // 2, kw // 2]
    total = w.sum(axis=(2, 3))
    dc = total.mean(axis=1)
    cs = (center - (total - center) / (kh * kw - 1)).mean(axis=1)
    return dc, cs


def sam_stats(w: np.ndarray) -> dict:  # w: (2, 7, 7)，平均分支 / 最大分支
    out = {}
    for name, b in [("avg", 0), ("max", 1)]:
        wb = w[b]
        c3 = wb[2:5, 2:5]
        out[f"dc_{name}"] = float(wb.sum())
        out[f"c3_{name}"] = float(c3.mean())
        out[f"out_{name}"] = float((wb.sum() - c3.sum()) / (wb.size - c3.size))
    return out


def kernel_stats(model: nn.Module, kind: str) -> dict:
    mods = dict(model.named_modules())
    out: dict = {}
    if "neck.cbam_p3.spatial_attention" in mods:
        conv = next(m for m in mods["neck.cbam_p3.spatial_attention"].conv.modules() if isinstance(m, nn.Conv2d))
        out["sam"] = sam_stats(conv.weight.detach().float().cpu().numpy()[0])
    out["bn"] = {}
    for k in range(3):
        b = bottleneck_name(kind, k)
        d3 = next(m for m in mods[b].modules()
                  if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3) and m.groups == 1)
        dc, cs = conv_stats(d3)
        e = {"dense3_dc_neg": float(np.mean(dc < 0)), "dense3_cs_neg": float(np.mean(cs < 0))}
        if kind == "sgc_ghost":
            dw = next(m for m in mods[f"{b}.DWConv2"].modules() if isinstance(m, nn.Conv2d) and m.groups > 1)
            dcw, csw = conv_stats(dw)
            e.update({"dw_dc_neg": float(np.mean(dcw < 0)), "dw_cs_neg": float(np.mean(csw < 0))})
        out["bn"][k] = e
    return out


# ---------------------------------------------------------------- 逐模型分析
def run_model(key: str, ds: YoloDetDataset, stems_all: list[str], small: list[str], med: list[str]) -> dict:
    spec = MODELS[key]
    model = ra.load_model(spec)
    mods = dict(model.named_modules())
    names = [n for n in need_names(spec["kind"]) if n in mods]
    pre_names = [n for n in pre_names_for(spec["kind"]) if n in mods]
    records, maps = [], {}
    for group, stems in [("small", small), ("medium", med)]:
        for stem in stems:
            img_t, lab = ds[stems_all.index(stem)]
            _, cx, cy, bw, bh = lab[0].tolist()
            box = ((cx - bw / 2) * 640, (cy - bh / 2) * 640, (cx + bw / 2) * 640, (cy + bh / 2) * 640)
            cap: dict = {}
            handles = [mods[n].register_forward_hook(ra.make_hook(cap, n)) for n in names]
            handles += [mods[n].register_forward_pre_hook(make_pre_hook(cap, n + ".in")) for n in pre_names]
            with torch.no_grad():
                model(img_t[None].float() / 255.0)
            for h in handles:
                h.remove()
            rec = {"group": group, "stem": stem, "box": box, "p4": {}, "p3": {}}
            for n in names:
                if not n.startswith("neck."):
                    rec["p4"][n] = stats_mean_rms(cap[n][0], box, S4)
            for n in pre_names:
                rec["p4"][n + ".in"] = stats_mean_rms(cap[n + ".in"][0], box, S4)
            if "neck.c3_p3" in cap:
                rec["p3"]["pre"] = stats_mean_rms(cap["neck.c3_p3"][0], box, S3)
                rec["p3"]["pre"]["ring_dom_ch"] = ring_dominant_frac(cap["neck.c3_p3"][0], box, S3)
            mask = None
            if "neck.cbam_p3.channel_attention" in cap:
                cam = cap["neck.cbam_p3.channel_attention"]
                cam_t = torch.from_numpy(cam)
                avg = cam_t.mean(1, keepdim=True)
                mx = cam_t.max(1, keepdim=True).values
                with torch.no_grad():
                    sam = mods["neck.cbam_p3.spatial_attention"]
                    mask = torch.sigmoid(sam.conv(torch.cat([avg, mx], 1)))[0, 0].numpy()
                rec["p3"]["cam"] = region_stats(cam[0].mean(0), box, S3)
                rec["p3"]["mask"] = region_stats(mask, box, S3)
                if "neck.c3_p3" in cap:
                    rec["p3"]["rho"] = spearman(mask, cap["neck.c3_p3"][0].mean(0))
            if "neck.cbam_p3" in cap:
                rec["p3"]["final"] = stats_mean_rms(cap["neck.cbam_p3"][0], box, S3)
                rec["p3"]["final"]["ring_dom_ch"] = ring_dominant_frac(cap["neck.cbam_p3"][0], box, S3)
            records.append(rec)
            if key == "ours" and group == "small":
                maps[stem] = {
                    "p4": {n: np.sqrt((cap[n][0] ** 2).mean(0)) for n in names if n.startswith("backbone.")},
                    "p3": {"pre": cap["neck.c3_p3"][0].mean(0),
                           "cam": cap["neck.cbam_p3.channel_attention"][0].mean(0),
                           "mask": mask, "final": cap["neck.cbam_p3"][0].mean(0)},
                }
                for n in pre_names:
                    maps[stem]["p4"][n + ".in"] = np.sqrt((cap[n + ".in"][0] ** 2).mean(0))
    sam_w = None
    if "neck.cbam_p3.spatial_attention" in mods:
        conv2d = next(m for m in mods["neck.cbam_p3.spatial_attention"].conv.modules() if isinstance(m, nn.Conv2d))
        sam_w = conv2d.weight[0].detach().numpy().copy()  # (2, 7, 7)
    return {"records": records, "maps": maps, "sam_w": sam_w, "kind": spec["kind"],
            "kern": kernel_stats(model, spec["kind"])}


# ---------------------------------------------------------------- 汇总
def summarize_p4(res: dict, key: str) -> dict:
    kind = res["kind"]
    cats = [("backbone.stage4.0", "P4下采样"), ("backbone.stage4.1.cv1", "块输入cv1"),
            ("backbone.stage4.1.cv2", "恒等支路cv2"), (bottleneck_name(kind, 0), "瓶颈#0"),
            (bottleneck_name(kind, 1), "瓶颈#1"), (bottleneck_name(kind, 2), "瓶颈#2"),
            ("backbone.stage4.1", "块输出")]
    out = {}
    for group in ("small", "medium"):
        recs = [r for r in res["records"] if r["group"] == group]
        out[group] = {}
        for name, label in cats:
            out[group][label] = [r["p4"][name]["pct"] for r in recs if name in r["p4"]]
        down = np.array([r["p4"]["backbone.stage4.0"]["pct"] for r in recs])
        block = np.array([r["p4"]["backbone.stage4.1"]["pct"] for r in recs])
        bgr = np.array([bg_ratio(r["p4"]["backbone.stage4.0"]) for r in recs])
        out[group]["_drop"] = (block - down).tolist()
        out[group]["_bgratio"] = bgr.tolist()
    return {"cats": [c[1] for c in cats], "vals": out}


# ---------------------------------------------------------------- 绘图
def _stretch(M: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(M, 1), np.percentile(M, 99.5)
    return np.clip((M - lo) / max(hi - lo, 1e-9), 0, 1)


def plot_p4(results: dict, out: Path) -> None:
    labels = ["P4下采样", "块输入cv1", "恒等支路cv2", "瓶颈#0", "瓶颈#1", "瓶颈#2", "块输出"]
    colors = {"ours": "#d62728", "c3topo": "#1f77b4", "yolo": "#2ca02c"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, group, title in [(axes[0], "small", "小目标（~14–24 px，60 张）"),
                             (axes[1], "medium", "中等目标（~40–70 px，30 张）")]:
        width = 0.25
        for j, key in enumerate(MODELS):
            vals = results[key]["vals"][group]
            data = [vals[l] for l in labels]
            pos = np.arange(len(labels)) + (j - 1) * width
            bp = ax.boxplot(data, positions=pos, widths=width * 0.9, patch_artist=True, showfliers=False)
            for patch in bp["boxes"]:
                patch.set_facecolor(colors[key])
                patch.set_alpha(0.6)
            ax.plot([], [], color=colors[key], label=MODELS[key]["label"], lw=6, alpha=0.6)
        ax.axhline(0.9, color="gray", ls="--", lw=1)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=20, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("GT 框内响应的分位数（越高越突出）")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    ra.save_fig(fig, out, 110)
    plt.close(fig)


def plot_ghost_decomp(recs: list, out: Path) -> None:
    items = []
    for k in range(3):
        b = f"backbone.stage4.1.bottleneck.{k}"
        items += [(f"{b}.DWConv2", f"残差DW#{k}", "#1f77b4"),
                  (f"{b}.SGhostConv2.in", f"SG2输入#{k}", "#ff7f0e"),
                  (f"{b}.SGhostConv2", f"SG2输出#{k}", "#d62728"),
                  (b, f"瓶颈输出#{k}", "#7f7f7f")]
    data = [[r["p4"][n]["pct"] for r in recs] for n, _, _ in items]
    fig, ax = plt.subplots(figsize=(13, 4.8))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    for patch, (_, _, c) in zip(bp["boxes"], items):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.axhline(0.9, color="gray", ls="--", lw=1)
    ax.set_xticks(np.arange(1, len(items) + 1))
    ax.set_xticklabels([lab for _, lab, _ in items], rotation=35, fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("GT 框内响应的分位数")
    ax.set_title("本模型 P4 SDGhost 内部（小目标，60 张）：残差 DW 支路 vs 门控 Ghost 路径", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    ra.save_fig(fig, out, 110)
    plt.close(fig)


def plot_p4_case(p4m: dict, rec: dict, out: Path) -> None:
    panels = [("backbone.stage4.0", "P4 下采样"),
              ("backbone.stage4.1.bottleneck.0.DWConv2", "残差DW（#0）"),
              ("backbone.stage4.1.bottleneck.0.SGhostConv2.in", "SG2 输入 x1·x2（#0）"),
              ("backbone.stage4.1.bottleneck.0.SGhostConv2", "SG2 输出（#0）"),
              ("backbone.stage4.1.bottleneck.0", "瓶颈输出（#0）"),
              ("backbone.stage4.1", "块输出")]
    box = np.array(rec["box"]) / S4
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.4))
    for ax, (name, title) in zip(axes, panels):
        ax.imshow(_stretch(p4m[name]), cmap="jet", interpolation="nearest")
        ax.add_patch(Rectangle((box[0] - 0.5, box[1] - 0.5), box[2] - box[0], box[3] - box[1],
                               fill=False, edgecolor="white", lw=1.5))
        st = rec["p4"][name]
        ax.set_title(f"{title}\nGT框分位：均值 {st['pct']:.2f} / RMS {st['pct_rms']:.2f}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"P4 块输出分位下降最大的图 {rec['stem']}（图为通道 RMS 能量，白框为 GT）", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    ra.save_fig(fig, out, 110)
    plt.close(fig)


def plot_cbam_full(p3: dict, rec: dict, out: Path) -> None:
    box = np.array(rec["box"]) / S3
    panels = [("CBAM 前（颈部输出）", "pre"), ("CAM 之后", "cam"), ("SAM 注意力", "mask"), ("CBAM 之后（N3）", "final")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, (title, key) in zip(axes, panels):
        ax.imshow(_stretch(p3[key]), cmap="jet", interpolation="nearest")
        ax.add_patch(Rectangle((box[0] - 0.5, box[1] - 0.5), box[2] - box[0], box[3] - box[1],
                               fill=False, edgecolor="white", lw=1.5))
        st = rec["p3"][key]
        ax.set_title(f"{title}\n框内 {st['pct']:.2f} / 外圈 {st['pct_ring']:.2f}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"CBAM 图 {rec['stem']}（白框为 GT；标题为 框内 / 外圈 均值在全图中的分位）", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    ra.save_fig(fig, out, 110)
    plt.close(fig)


def plot_cbam_zoom(p3: dict, rec: dict, out: Path, radius: int = 5) -> None:
    H, W = p3["pre"].shape
    box = rec["box"]
    br1, br2, bc1, bc2 = box_cells(box, S3, H, W)
    rc = min(int(((box[1] + box[3]) / 2) // S3), H - 1)
    cc = min(int(((box[0] + box[2]) / 2) // S3), W - 1)
    r0, r1 = max(rc - radius, 0), min(rc + radius + 1, H)
    c0, c1 = max(cc - radius, 0), min(cc + radius + 1, W)
    panels = [("CBAM 前", "pre"), ("CAM 之后", "cam"), ("SAM 注意力", "mask"), ("CBAM 之后（N3）", "final")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, (title, key) in zip(axes, panels):
        M = p3[key]
        lo, hi = np.percentile(M, 1), np.percentile(M, 99.5)
        crop = M[r0:r1, c0:c1]
        vn = np.clip((crop - lo) / max(hi - lo, 1e-9), 0, 1)
        ax.imshow(vn, cmap="jet", vmin=0, vmax=1, interpolation="nearest")
        for i in range(crop.shape[0]):
            for j in range(crop.shape[1]):
                ax.text(j, i, f"{crop[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="black" if vn[i, j] > 0.55 else "white")
        ax.add_patch(Rectangle((bc1 - c0 - 0.5, br1 - r0 - 0.5), bc2 - bc1, br2 - br1,
                               fill=False, edgecolor="black", lw=2))
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"CBAM 图 {rec['stem']}：P3 中心附近 {r1 - r0}×{c1 - c0} 格的数值（黑框为 GT 所占格）", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    ra.save_fig(fig, out, 110)
    plt.close(fig)


def plot_sam_kernels(res_by_key: dict, out: Path) -> None:
    keys = [k for k in ("ours", "c3topo") if res_by_key[k]["sam_w"] is not None]
    fig, axes = plt.subplots(len(keys), 2, figsize=(7, 3.4 * len(keys)))
    axes = np.atleast_2d(axes)
    for r, key in enumerate(keys):
        w = res_by_key[key]["sam_w"]
        for c, (branch, name) in enumerate([(0, "平均通道分支"), (1, "最大通道分支")]):
            v = np.abs(w[branch]).max()
            im = axes[r, c].imshow(w[branch], cmap="RdBu_r", vmin=-v, vmax=v)
            axes[r, c].set_title(f"{MODELS[key]['label']}\n{name}（权重和 {w[branch].sum():+.2f}）", fontsize=8)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            fig.colorbar(im, ax=axes[r, c], fraction=0.046)
    fig.suptitle("CBAM 的 SAM 7×7 卷积核（P3）", fontsize=10)
    fig.tight_layout()
    ra.save_fig(fig, out, 110)
    plt.close(fig)


# ---------------------------------------------------------------- 主流程
def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    data_dir = ROOT / "data" / "Det_Fly"
    ds = YoloDetDataset(data_dir / "images" / "test", data_dir / "labels" / "test", 640, train=False)
    stems_all = [p.stem for p in ds.paths]
    small, med = pick_samples(data_dir)
    print(f"样本：小目标 {len(small)} 张，中等目标 {len(med)} 张", flush=True)

    results = {}
    for key in MODELS:
        results[key] = run_model(key, ds, stems_all, small, med)
        print(f"完成：{MODELS[key]['label']}（{time.time() - t0:.0f}s）", flush=True)

    def recs_of(key: str, group: str = "small") -> list:
        return [r for r in results[key]["records"] if r["group"] == group]

    # (1) P4 总表
    p4 = {key: summarize_p4(results[key], key) for key in MODELS}
    print("\n=== (1) P4：GT 框内响应分位（中位数 / ≥0.9 比例） ===")
    for group in ("small", "medium"):
        print(f"[{group}]")
        for label in p4["ours"]["cats"]:
            row = f"  {label:<10}"
            for key in MODELS:
                v = np.array(p4[key]["vals"][group][label])
                row += f" | {key:>7}: {np.median(v):.2f} / {100 * np.mean(v >= 0.9):3.0f}%"
            print(row)
    drop = np.array(p4["ours"]["vals"]["small"]["_drop"])
    bgr = np.array(p4["ours"]["vals"]["small"]["_bgratio"])
    print("\n块输出 − 下采样（小目标，分位差）：")
    for key in MODELS:
        d2 = np.array(p4[key]["vals"]["small"]["_drop"])
        print(f"  {MODELS[key]['label']:<28} 中位数 {np.median(d2):+.2f}，IQR [{np.percentile(d2, 25):+.2f}, "
              f"{np.percentile(d2, 75):+.2f}]，下降 >0.3 的图 {100 * np.mean(d2 < -0.3):3.0f}%，"
              f">0.5 的图 {100 * np.mean(d2 < -0.5):3.0f}%")
    print(f"  本模型：下采样背景/目标比 与 下降量 的秩相关 {spearman(bgr, drop):+.2f}")
    print("\n=== (1r) 同上，改用通道 RMS 分位（小目标）：中位数 / ≥0.9 比例 ===")
    for key in MODELS:
        rs = recs_of(key)
        kind = MODELS[key]["kind"]
        cats_r = [("backbone.stage4.0", "P4下采样"), ("backbone.stage4.1.cv2", "恒等cv2")]
        cats_r += [(bottleneck_name(kind, k), f"瓶颈#{k}") for k in range(3)]
        cats_r += [("backbone.stage4.1", "块输出")]
        row = f"  {MODELS[key]['label']:<28}"
        for name, lab in cats_r:
            v = np.array([r["p4"][name]["pct_rms"] for r in rs])
            row += f" | {lab} {np.median(v):.2f}/{100 * np.mean(v >= 0.9):.0f}%"
        print(row)
        d = np.array([r["p4"]["backbone.stage4.1"]["pct_rms"] - r["p4"]["backbone.stage4.0"]["pct_rms"] for r in rs])
        print(f"  {'':<28} 块输出−下采样（RMS）：中位 {np.median(d):+.2f}，IQR [{np.percentile(d, 25):+.2f}, "
              f"{np.percentile(d, 75):+.2f}]，下降 >0.3 的图 {100 * np.mean(d < -0.3):.0f}%")

    # (1b) Ghost 瓶颈内部
    rs_ours = recs_of("ours")
    print("\n=== (1b) 本模型 P4 瓶颈内部（小目标）：分位 中位数 [IQR] / ≥0.9 比例 ===")
    for k in range(3):
        b = f"backbone.stage4.1.bottleneck.{k}"
        for name, lab in [(f"{b}.DWConv2", "残差DW"), (f"{b}.SGhostConv1", "SG1输出 x1"),
                          (f"{b}.DFCAttention", "门控 x2"), (f"{b}.SGhostConv2.in", "SG2输入 x1·x2"),
                          (f"{b}.SGhostConv2", "SG2输出"), (b, "瓶颈输出")]:
            v = np.array([r["p4"][name]["pct"] for r in rs_ours])
            vr = np.array([r["p4"][name]["pct_rms"] for r in rs_ours])
            print(f"  #{k} {lab:<14} 通道均值 {np.median(v):.2f} [{np.percentile(v, 25):.2f}, {np.percentile(v, 75):.2f}]"
                  f" ≥0.9 {100 * np.mean(v >= 0.9):3.0f}% | 通道RMS 中位 {np.median(vr):.2f}"
                  f" ≥0.9 {100 * np.mean(vr >= 0.9):3.0f}%")

    print("\n=== (1c) 瓶颈输出分位 与 其输入的局部对比度 / 背景目标比 的秩相关（小目标） ===")
    for key in MODELS:
        rs = recs_of(key)
        for k in range(3):
            b = bottleneck_name(MODELS[key]["kind"], k)
            v = np.array([r["p4"][b]["pct"] for r in rs])
            c = np.array([contrast(r["p4"][b + ".in"]) for r in rs])
            g = np.array([bg_ratio(r["p4"][b + ".in"]) for r in rs])
            line = f"  {MODELS[key]['label']} 瓶颈#{k}: ρ(对比度)={spearman(v, c):+.2f}  ρ(背景/目标)={spearman(v, g):+.2f}"
            if MODELS[key]["kind"] == "sgc_ghost":
                s = np.array([r["p4"][b + ".SGhostConv2"]["pct"] for r in rs])
                cs = np.array([contrast(r["p4"][b + ".SGhostConv2.in"]) for r in rs])
                line += f" | SG2输出 ρ(SG2输入对比度)={spearman(s, cs):+.2f}"
            print(line)

    # (2) CBAM
    print("\n=== (2) CBAM（P3，小目标）：框内 / 外圈 在全图中的分位（中位数） ===")
    for key in MODELS:
        rs = recs_of(key)
        for stage, label in [("pre", "CBAM前"), ("cam", "CAM后"), ("mask", "SAM注意力"), ("final", "CBAM后")]:
            st = [r["p3"][stage] for r in rs if stage in r["p3"]]
            if not st:
                continue
            pb = np.array([s["pct"] for s in st])
            pr = np.array([s["pct_ring"] for s in st])
            ring_gt = np.array([s["ring"] > s["box"] for s in st])
            print(f"  {MODELS[key]['label']:<28} {label:<9} 框内 {np.median(pb):.2f}  外圈 {np.median(pr):.2f}"
                  f"  外圈均值 > 框内均值 的图 {100 * ring_gt.mean():3.0f}%")
        rho = np.array([r["p3"]["rho"] for r in rs if "rho" in r["p3"]])
        if rho.size:
            print(f"  {MODELS[key]['label']:<28} SAM 注意力 与 颈部特征 的秩相关（中位数）：{np.median(rho):+.2f}")
        for stage, label in [("pre", "CBAM前"), ("final", "CBAM后")]:
            fr = np.array([r["p3"][stage]["ring_dom_ch"] for r in rs if stage in r["p3"]])
            if fr.size:
                print(f"  {MODELS[key]['label']:<28} 逐通道 外圈均值>框内均值 的通道比例 {label}：中位 {100 * np.median(fr):.0f}%，"
                      f"平均 {100 * np.mean(fr):.0f}%")
        if "mask" in rs[0]["p3"]:
            mr = np.array([r["p3"]["mask"]["ring"] / r["p3"]["mask"]["box"] for r in rs])
            print(f"  {MODELS[key]['label']:<28} SAM 注意力 外圈均值 / 框内均值（中位）：{np.median(mr):.2f}")
        if "final" in rs[0]["p3"]:
            c_pre = np.array([contrast(r["p3"]["pre"]) for r in rs])
            c_fin = np.array([contrast(r["p3"]["final"]) for r in rs])
            print(f"  {MODELS[key]['label']:<28} 框内−外圈 对比度（÷全图std，中位）：CBAM前 {np.median(c_pre):+.2f} → "
                  f"CBAM后 {np.median(c_fin):+.2f}，CBAM后 <0 的图 {100 * np.mean(c_fin < 0):.0f}%")
        else:
            c_pre = np.array([contrast(r["p3"]["pre"]) for r in rs])
            print(f"  {MODELS[key]['label']:<28} 框内−外圈 对比度（÷全图std，中位）：CBAM前 {np.median(c_pre):+.2f}，"
                  f"<0 的图 {100 * np.mean(c_pre < 0):.0f}%")

    # (2b) 卷积核权重
    print("\n=== (2b) 卷积核（只看权重）===")
    for key in ("ours", "c3topo"):
        s = results[key]["kern"]["sam"]
        print(f"  {MODELS[key]['label']} SAM 7×7：平均分支 权重和 {s['dc_avg']:+.2f}（中心3×3 {s['c3_avg']:+.3f}，"
              f"外圈 {s['out_avg']:+.3f}）；最大分支 权重和 {s['dc_max']:+.2f}（中心3×3 {s['c3_max']:+.3f}，"
              f"外圈 {s['out_max']:+.3f}）")
    for key in MODELS:
        for k in range(3):
            e = results[key]["kern"]["bn"][k]
            line = (f"  {MODELS[key]['label']} 瓶颈#{k} 3×3 稠密卷积：权重和<0 的输出通道 {100 * e['dense3_dc_neg']:.0f}%，"
                    f"中心<外圈 的输出通道 {100 * e['dense3_cs_neg']:.0f}%")
            if "dw_dc_neg" in e:
                line += (f" | 残差DW 3×3：权重和<0 {100 * e['dw_dc_neg']:.0f}%，"
                         f"中心<外圈 {100 * e['dw_cs_neg']:.0f}%")
            print(line)

    # 最差样例
    gap4 = np.array([r["p4"]["backbone.stage4.0"]["pct"] - r["p4"]["backbone.stage4.1"]["pct"] for r in rs_ours])
    worst4 = rs_ours[int(np.argmax(gap4))]
    gap3 = np.array([r["p3"]["final"]["pct_ring"] - r["p3"]["final"]["pct"] for r in rs_ours])
    worst3 = rs_ours[int(np.argmax(gap3))]
    print(f"\n最差 P4 图：{worst4['stem']}（下采样 {worst4['p4']['backbone.stage4.0']['pct']:.2f} → "
          f"块输出 {worst4['p4']['backbone.stage4.1']['pct']:.2f}）")
    print(f"最差 CBAM 图：{worst3['stem']}（CBAM 后 框内 {worst3['p3']['final']['pct']:.2f}，"
          f"外圈 {worst3['p3']['final']['pct_ring']:.2f}）")
    print(f"CBAM 后 外圈分位 > 框内分位 的图：{100 * np.mean(gap3 > 0):.0f}%")
    print(f"\n最差 P4 图 {worst4['stem']} 逐节点（分位：通道均值 / 通道RMS）：")
    for name, lab in [("backbone.stage4.0", "P4下采样"), ("backbone.stage4.1.cv1", "块输入cv1"),
                      ("backbone.stage4.1.cv2", "恒等支路cv2"), (bottleneck_name("sgc_ghost", 0), "瓶颈#0输出"),
                      (bottleneck_name("sgc_ghost", 1), "瓶颈#1输出"), (bottleneck_name("sgc_ghost", 2), "瓶颈#2输出"),
                      ("backbone.stage4.1", "块输出")]:
        st = worst4["p4"][name]
        print(f"  {lab:<10} {st['pct']:.2f} / {st['pct_rms']:.2f}")

    # 输出
    plot_p4(p4, OUT / "p4_by_model.png")
    plot_ghost_decomp(rs_ours, OUT / "p4_ghost_decomp.png")
    plot_p4_case(results["ours"]["maps"][worst4["stem"]]["p4"], worst4, OUT / "p4_case_worst.png")
    plot_cbam_full(results["ours"]["maps"][worst3["stem"]]["p3"], worst3, OUT / "cbam_example.png")
    plot_cbam_zoom(results["ours"]["maps"][worst3["stem"]]["p3"], worst3, OUT / "cbam_zoom_worst.png")
    plot_sam_kernels(results, OUT / "sam_kernels.png")
    (OUT / "summary.json").write_text(json.dumps({
        "p4": {k: {"cats": v["cats"], "vals": v["vals"]} for k, v in p4.items()},
        "kernels": {k: results[k]["kern"] for k in MODELS},
        "worst_p4": worst4["stem"], "worst_cbam": worst3["stem"],
        "n_small": len(small), "n_medium": len(med)}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n输出目录：{OUT}（总耗时 {time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
