"""Ghost 层特征图可视化，以及 Ghost 本征/幽灵通道相关性统计（使用 run 的权重）。

输出目录：<run>/feature_maps/
  {图名}_01_overview.png     骨干各下采样 / 阶段输出 + 颈部 N3~N5 的通道均值叠加在原图上
  {图名}_02_ghost_block.png  Stage3 第 1 个 Ghost 瓶颈（P3 尺度 80×80）的内部中间结果
  {图名}_03_ghost_pairs.png  本征通道与其对应幽灵通道的逐对对比（标注 Pearson r）
  ghost_corr_stats.png/json  全部 Ghost 卷积的本征-幽灵相关性统计（--stats-num 张测试图平均）

本脚本的块级图针对 sgc_yolov5s_c3_sgdown_gbot_dw5 的结构编写（Stage3 第 1 个瓶颈）。

用法（在仓库根目录执行）：
  python experiments/detection/project2_2611/visualize_features.py \
      -m sgc_yolov5s_c3_sgdown_gbot_dw5 -n 2026-11-13_029_SGC_c3_sgdown_gbot_dw5_s_V1 \
      --images 0100061 0100015 --stats-num 50
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trainer import RunRecorder, build_config, build_model, model_output_dir, resolve_device  # noqa: E402
from trainer.config import ROOT  # noqa: E402
from trainer.data import YoloDetDataset  # noqa: E402

from src.module.conv import GhostConv  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

def save_fig(fig, out: Path, dpi: int) -> None:
    """先写临时文件再原子替换，并在 Windows 文件被短暂占用时重试。"""
    tmp = out.with_name(out.stem + ".tmp.png")
    for attempt in range(6):
        try:
            fig.savefig(tmp, dpi=dpi)
            os.replace(tmp, out)
            return
        except OSError:
            time.sleep(0.5 * (attempt + 1))
    raise OSError(f"无法写入 {out}")


OVERVIEW = [
    ("backbone.stage2.0", "P2 下采样 SGhost(3×3,s2)"),
    ("backbone.stage2.1", "Stage2 SDGhost 输出"),
    ("backbone.stage3.0", "P3 下采样 SGhost(3×3,s2)"),
    ("backbone.stage3.1", "Stage3 SDGhost 输出 (P3)"),
    ("backbone.stage4.0", "P4 下采样 SGhost(3×3,s2)"),
    ("backbone.stage4.1", "Stage4 SDGhost 输出 (P4)"),
    ("backbone.stage5.0", "P5 下采样 SGhost(3×3,s2)"),
    ("backbone.stage5.1", "Stage5 SDGhost 输出"),
    ("neck.cbam_p3", "Neck N3 (CBAM 后)"),
    ("neck.cbam_p4", "Neck N4 (CBAM 后)"),
    ("neck.cbam_p5", "Neck N5 (CBAM 后)"),
]
BLOCK = "backbone.stage3.1"  # P3 尺度 SDGhost 块（80×80）
BOTTLE = f"{BLOCK}.bottleneck.0"  # 块内第 1 个 Ghost 瓶颈
SG1 = f"{BOTTLE}.SGhostConv1"  # 1×1 本征 + 廉价 dw
SG2 = f"{BOTTLE}.SGhostConv2"  # 3×3 本征 + 廉价 dw（mid_dw）
DFC = f"{BOTTLE}.DFCAttention"
RES = f"{BOTTLE}.DWConv2"


def _norm(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    lo, hi = float(np.percentile(a, 1.0)), float(np.percentile(a, 99.5))
    if hi - lo < 1e-8:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def energy(a: np.ndarray) -> np.ndarray:
    """通道 RMS 能量图 sqrt(mean_c x²)：正负通道不会相互抵消（通道均值会）。a: (C, H, W)。"""
    a = np.asarray(a, dtype=np.float64)
    return np.sqrt((a * a).mean(axis=0))


def overlay(img_u8: np.ndarray, heat: np.ndarray) -> np.ndarray:
    h, w = img_u8.shape[:2]
    heat_up = cv2.resize(_norm(heat), (w, h), interpolation=cv2.INTER_LINEAR)
    col = plt.get_cmap("jet")(heat_up)[..., :3]
    return 0.5 * img_u8.astype(np.float32) / 255.0 + 0.5 * col


def draw_boxes(img_u8: np.ndarray, lab: torch.Tensor) -> np.ndarray:
    out = img_u8.copy()
    S = img_u8.shape[0]
    for row in lab.numpy():
        _, cx, cy, bw, bh = row
        x1, y1 = int((cx - bw / 2) * S), int((cy - bh / 2) * S)
        x2, y2 = int((cx + bw / 2) * S), int((cy + bh / 2) * S)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return out.astype(np.float32) / 255.0


def dw_kernel(mod: torch.nn.Module) -> int:
    for m in mod.modules():
        if isinstance(m, torch.nn.Conv2d):
            return int(m.kernel_size[0])
    return -1


def fmt(arr: np.ndarray) -> str:
    return f"{arr.shape[0]}×{arr.shape[1]}×{arr.shape[2]}"


def pick_top_var(arr: np.ndarray, k: int) -> list[int]:
    var = arr.reshape(arr.shape[0], -1).var(axis=1)
    return [int(i) for i in np.argsort(-var)[:k]]


def pair_corr(intr: np.ndarray, ghost: np.ndarray) -> tuple[np.ndarray, float]:
    """逐通道 Pearson r（本征通道 i 与其幽灵通道 i），以及幽灵/本征的平均绝对激活比。"""
    C = min(intr.shape[0], ghost.shape[0])
    a = intr[:C].reshape(C, -1).astype(np.float64)
    b = ghost[:C].reshape(C, -1).astype(np.float64)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    r = np.full(C, np.nan)
    ok = den > 1e-10
    r[ok] = (a * b).sum(axis=1)[ok] / den[ok]
    ratio = float(np.abs(ghost).mean() / (np.abs(intr).mean() + 1e-12))
    return r, ratio


def short_label(name: str) -> str:
    p = name.split(".")
    stage = p[1].replace("stage", "")
    if "bottleneck" in p:
        b = p[p.index("bottleneck") + 1]
        return f"S{stage}-B{b}-{p[-1].replace('SGhostConv', 'SG')}"
    return f"S{stage}-down"


def draw_grid(panels: list, ncols: int, out: Path, suptitle: str) -> None:
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.9 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)
    for k, ax in enumerate(axes):
        if k >= len(panels):
            ax.axis("off")
            continue
        title, data, kind = panels[k]
        if kind == "rgb":
            ax.imshow(data)
        else:
            ax.imshow(data, cmap=kind, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    save_fig(fig, out, 110)
    plt.close(fig)


def plot_overview(img: np.ndarray, lab: torch.Tensor, cap: dict, out: Path, stem: str) -> None:
    panels = [("输入图像 + GT 框", draw_boxes(img, lab), "rgb")]
    for name, label in OVERVIEW:
        arr = cap[name][0]
        panels.append((f"{label}\n{fmt(arr)}", overlay(img, energy(arr)), "rgb"))
    draw_grid(panels, 4, out, f"{stem}：骨干与颈部特征图（通道 RMS 能量叠加在原图上）")


def plot_block(cap: dict, dw_sg2: int, out: Path, stem: str) -> None:
    def A(name: str) -> np.ndarray:
        return cap[name][0]

    items = [
        ("Bottleneck 输入 (cv1)", A(f"{BLOCK}.cv1")),
        ("SG1 本征 (1×1)", A(f"{SG1}.conv")),
        ("SG1 幽灵 (dw=1)", A(f"{SG1}.cheap_conv")),
        ("DFC 注意力门 (sigmoid)", A(DFC)),
        ("门控输出 x1·x2", A(SG1) * A(DFC)),
        ("SG2 本征 (3×3)", A(f"{SG2}.conv")),
        (f"SG2 幽灵 (dw={dw_sg2})", A(f"{SG2}.cheap_conv")),
        ("残差 DW3×3 支路", A(RES)),
        ("Bottleneck#1 输出", A(BOTTLE)),
        ("Bottleneck#2 输出", A(f"{BLOCK}.bottleneck.1")),
        ("cv2 恒等支路", A(f"{BLOCK}.cv2")),
        ("SDGhost 输出 (P3)", A(BLOCK)),
    ]
    panels = [(f"{t}\n{fmt(a)}", _norm(energy(a)), "jet") for t, a in items]
    draw_grid(panels, 4, out, f"{stem}：Stage3 第 1 个 Ghost 瓶颈内部（P3 尺度，通道 RMS 能量）")


def plot_pairs(cap: dict, out: Path, stem: str) -> None:
    panels = []
    for tag, mod in [("SG1", SG1), ("SG2", SG2)]:
        intr, ghost = cap[f"{mod}.conv"][0], cap[f"{mod}.cheap_conv"][0]
        r, _ = pair_corr(intr, ghost)
        for i in pick_top_var(intr, 4):
            panels.append((f"{tag} 本征 ch{i}", _norm(intr[i]), "viridis"))
            panels.append((f"{tag} 幽灵 ch{i}  r={r[i]:.2f}", _norm(ghost[i]), "viridis"))
    draw_grid(
        panels, 4, out,
        f"{stem}：Ghost 本征通道与对应幽灵通道逐对对比（r 为两者的 Pearson 相关系数）",
    )


def plot_stats(rows: list[dict], out_png: Path, out_json: Path) -> None:
    labels = [r["label"] for r in rows]
    colors = [{1: "#1f77b4", 3: "#ff7f0e", 5: "#d62728"}.get(r["dw"], "#7f7f7f") for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].bar(x, [r["mean_abs_r"] for r in rows], color=colors)
    ax[0].set_ylabel("mean |r|（本征 vs 幽灵，通道级）")
    ax[0].set_title("各 Ghost 卷积：本征与幽灵通道的相关性")
    ax[1].bar(x, [r["magnitude_ratio"] for r in rows], color=colors)
    ax[1].set_ylabel("平均绝对激活幅值比 (幽灵 / 本征)")
    ax[1].set_title("各 Ghost 卷积：幽灵与本征的激活幅值比")
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(labels, rotation=75, fontsize=8)
        a.grid(axis="y", alpha=0.3)
    from matplotlib.patches import Patch

    handles = [Patch(color="#1f77b4", label="dw=1"), Patch(color="#ff7f0e", label="dw=3"),
               Patch(color="#d62728", label="dw=5")]
    ax[0].legend(handles=handles, fontsize=8, loc="upper right")
    fig.tight_layout()
    save_fig(fig, out_png, 120)
    plt.close(fig)
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--model", required=True, help="configs/models/<model>.toml")
    p.add_argument("-n", "--name", required=True, help="run 名称（runs 目录下的文件夹名）")
    p.add_argument("--split", default="test")
    p.add_argument("--checkpoint", choices=["best", "last"], default="best")
    p.add_argument("--images", nargs="+", required=True, help="图像文件名（不含扩展名）")
    p.add_argument("--stats-num", type=int, default=50, help="用于相关性统计的随机测试图数量")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = model_output_dir(build_config(args.model)) / args.name
    rec = RunRecorder.open(run_dir)
    cfg = rec.load_config()
    device = resolve_device(args.device)

    ckpt = torch.load(rec.checkpoints_dir / f"{args.checkpoint}.pth", map_location=device, weights_only=False)
    nc, img_size = int(ckpt["nc"]), int(ckpt["img_size"])
    model = build_model(cfg, nc).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    mods = dict(model.named_modules())
    ghost_names = [n for n, m in model.named_modules() if isinstance(m, GhostConv)]
    need = {n for n, _ in OVERVIEW} | {BLOCK, f"{BLOCK}.cv1", f"{BLOCK}.cv2", f"{BLOCK}.bottleneck.1",
                                        BOTTLE, SG1, SG2, DFC, RES}
    for n in ghost_names:
        need |= {n, f"{n}.conv", f"{n}.cheap_conv"}

    captured: dict[str, np.ndarray] = {}

    def make_hook(key: str):
        def _hook(_m, _inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            captured[key] = o.detach().float().cpu().numpy()

        return _hook

    for key in sorted(need):
        mods[key].register_forward_hook(make_hook(key))

    data_dir = Path(cfg["dataset"]["data_dir"])
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()
    ds = YoloDetDataset(data_dir / "images" / args.split, data_dir / "labels" / args.split, img_size, train=False)
    stems = [p.stem for p in ds.paths]

    out_dir = run_dir / "feature_maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    dw_sg2 = dw_kernel(mods[f"{SG2}.cheap_conv"])

    for stem in args.images:
        if stem not in stems:
            raise SystemExit(f"{stem} 不在 {args.split} 集中")
        img_t, lab = ds[stems.index(stem)]
        img = img_t.permute(1, 2, 0).contiguous().numpy()
        captured.clear()
        with torch.no_grad():
            model(img_t[None].float().to(device) / 255.0)
        plot_overview(img, lab, captured, out_dir / f"{stem}_01_overview.png", stem)
        plot_block(captured, dw_sg2, out_dir / f"{stem}_02_ghost_block.png", stem)
        plot_pairs(captured, out_dir / f"{stem}_03_ghost_pairs.png", stem)
        print(f"已输出 {stem} 的 3 张图", flush=True)

    rng = np.random.default_rng(0)
    pick = rng.choice(len(ds), size=min(args.stats_num, len(ds)), replace=False)
    acc = {n: {"r": [], "ratio": []} for n in ghost_names}
    for j, idx in enumerate(pick, 1):
        img_t, _ = ds[int(idx)]
        captured.clear()
        with torch.no_grad():
            model(img_t[None].float().to(device) / 255.0)
        for n in ghost_names:
            r, ratio = pair_corr(captured[f"{n}.conv"][0], captured[f"{n}.cheap_conv"][0])
            fin = r[np.isfinite(r)]
            acc[n]["r"].append(float(np.abs(fin).mean()) if fin.size else float("nan"))
            acc[n]["ratio"].append(ratio)
        if j % 10 == 0:
            print(f"  统计进度 {j}/{len(pick)}", flush=True)

    rows = []
    for n in ghost_names:
        rows.append({
            "name": n,
            "label": short_label(n),
            "dw": dw_kernel(mods[f"{n}.cheap_conv"]),
            "mean_abs_r": float(np.nanmean(acc[n]["r"])),
            "std_abs_r": float(np.nanstd(acc[n]["r"])),
            "magnitude_ratio": float(np.mean(acc[n]["ratio"])),
            "n_images": len(pick),
        })
    plot_stats(rows, out_dir / "ghost_corr_stats.png", out_dir / "ghost_corr_stats.json")

    print(f"{'层':<14}{'dw':>4}{'mean|r|':>10}{'std':>8}{'幅值比':>9}")
    for r in rows:
        print(f"{r['label']:<14}{r['dw']:>4}{r['mean_abs_r']:>10.3f}{r['std_abs_r']:>8.3f}"
              f"{r['magnitude_ratio']:>9.3f}")
    print("输出目录:", out_dir)


if __name__ == "__main__":
    main()
