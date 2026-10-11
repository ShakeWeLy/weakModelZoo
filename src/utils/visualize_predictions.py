"""UNet / TransUNet / Swin-UNet 在 weakUnet_Synapse 测试集上的预测诊断与可视化。

流程：
1. TransUNet、Swin-UNet 各自在子进程中加载 checkpoint，对 12 个测试体积逐切片推理（224x224）；
   UNet 直接读取 analyze.py 已保存的切片预测（同为 224x224）。
2. 诊断：逐病例、逐器官判断真值/预测是否存在，统计漏检、只有假阳性、两者都无的情况，
   并计算 224 分辨率下的 3D HD95 与孤立误检连通域数量。
3. 为指定病例绘制对比图（原图 / 真值 / 三个模型）。

用法（项目根目录）：
    python src/utils/visualize_predictions.py
    python src/utils/visualize_predictions.py --vis_cases case0038 case0008
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from medpy import metric  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.ndimage import zoom  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/weakUnet_Synapse"
TEST_LIST = ROOT / "data/weakUnet_Synapse_lists/test_vol.txt"
OUT_DIR = ROOT / "outputs/paper_report/vis"
UNET_PRED_DIR = ROOT / "experiments/segment/project1_2610/UNet/runs/2026-10-08_001_UNet_V1/predictions/test/predictions"
TRANSUNET_DIR = ROOT / "TransUNet"
SWIN_DIR = ROOT / "Swin-Unet"
CKPT = {
    "TransUNet": TRANSUNET_DIR / "model_out/TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs16_224/epoch_149.pth",
    "Swin-UNet": SWIN_DIR / "model_out/best_model.pth",
}
IMG = 224
N_CLASSES = 9
ORGANS = ["spleen", "right kidney", "left kidney", "gallbladder", "liver", "stomach", "aorta", "pancreas"]
MODELS = ["UNet", "TransUNet", "Swin-UNet"]
DEFAULT_VIS_CASES = ["case0038", "case0008"]
# 与 trainer/labels.py 中的标签顺序一致：1..8
LABEL_COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6"]


def load_cases() -> list[str]:
    return [line.strip() for line in TEST_LIST.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_case(case: str) -> tuple[np.ndarray, np.ndarray]:
    """读取 3D 体积并逐切片缩放到 224x224（图像 三次插值，标签 最近邻）。"""
    with h5py.File(DATA_DIR / "test_vol_h5" / f"{case}.npy.h5", "r") as h:
        img = h["image"][:]
        lab = h["label"][:]
    factor = IMG / img.shape[1]
    img224 = np.stack([zoom(s, (factor, factor), order=3) for s in img]).astype(np.float32)
    lab224 = np.stack([zoom(s, (factor, factor), order=0) for s in lab]).astype(np.uint8)
    return img224, lab224


# ---------------- 推理（子进程内执行，避免 networks/utils 包名冲突） ----------------

def _predict_volume(net, img224: np.ndarray) -> np.ndarray:
    import torch

    preds = []
    with torch.no_grad():
        for s in img224:
            x = torch.from_numpy(s).float()[None, None].cuda()
            out = net(x)
            preds.append(torch.argmax(torch.softmax(out, dim=1), dim=1)[0].cpu().numpy())
    return np.stack(preds).astype(np.uint8)


def _build_transunet():
    sys.path.insert(0, str(TRANSUNET_DIR))
    os.chdir(TRANSUNET_DIR)
    import torch
    from networks.vit_seg_modeling import CONFIGS, VisionTransformer

    cfg = CONFIGS["R50-ViT-B_16"]
    cfg.n_classes = N_CLASSES
    cfg.n_skip = 3
    cfg.patches.size = (16, 16)
    cfg.patches.grid = (IMG // 16, IMG // 16)
    net = VisionTransformer(cfg, img_size=IMG, num_classes=N_CLASSES).cuda()
    net.load_state_dict(torch.load(CKPT["TransUNet"], map_location="cuda"))
    net.eval()
    return net


def _build_swin():
    sys.path.insert(0, str(SWIN_DIR))
    os.chdir(SWIN_DIR)
    from argparse import Namespace

    import torch
    from config import get_config
    from networks.vision_transformer import SwinUnet

    # 与 Swin-Unet/test.py 训练时的参数一致（见 log.txt 中的 Namespace）
    args = Namespace(cfg="configs/swin_tiny_patch4_window7_224_lite.yaml", opts=None, batch_size=24, zip=False,
                     cache_mode="part", resume=None, accumulation_steps=None, use_checkpoint=False,
                     amp_opt_level="O1", tag=None, eval=False, throughput=False)
    config = get_config(args)
    net = SwinUnet(config, img_size=IMG, num_classes=N_CLASSES).cuda()
    net.load_state_dict(torch.load(CKPT["Swin-UNet"], map_location="cuda"))
    net.eval()
    return net


def run_inference_subprocess(model: str, cases: list[str]) -> None:
    """在子进程中推理，结果保存为 OUT_DIR/preds/<model>/<case>.npy。"""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--_infer", model, "--cases", *cases]
    subprocess.run(cmd, cwd=ROOT, check=True)


def _infer_main(model: str, cases: list[str]) -> None:
    import torch  # noqa: F401  (确保 CUDA 可用时才继续)

    net = _build_transunet() if model == "TransUNet" else _build_swin()
    save_dir = OUT_DIR / "preds" / model
    save_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        img224, _ = load_case(case)
        pred = _predict_volume(net, img224)
        np.save(save_dir / f"{case}.npy", pred)
        print(f"[{model}] {case}: {pred.shape}", flush=True)


def load_unet_prediction(case: str) -> np.ndarray:
    files = sorted(UNET_PRED_DIR.glob(f"{case}_slice*.npy"))
    if not files:
        raise FileNotFoundError(f"未找到 UNet 预测: {case}")
    return np.stack([np.load(f) for f in files]).astype(np.uint8)


def load_prediction(model: str, case: str) -> np.ndarray:
    if model == "UNet":
        return load_unet_prediction(case)
    return np.load(OUT_DIR / "preds" / model / f"{case}.npy")


# ---------------- 诊断 ----------------

def diagnose_organ(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    p, g = pred_mask.astype(bool), gt_mask.astype(bool)
    p_any, g_any = bool(p.any()), bool(g.any())
    if g_any and p_any:
        category = "both"
        dice = 2.0 * np.logical_and(p, g).sum() / (p.sum() + g.sum())
        hd95 = float(metric.binary.hd95(p, g))
    elif g_any:
        category, dice, hd95 = "miss", 0.0, float("nan")
    elif p_any:
        category, dice, hd95 = "fp_only", float("nan"), float("nan")
    else:
        category, dice, hd95 = "absent", float("nan"), float("nan")
    stray = 0
    if p_any:
        comp, n = ndimage.label(p)
        for k in range(1, n + 1):
            if not np.logical_and(comp == k, g).any():
                stray += 1
    return {"category": category, "dice": dice, "hd95_224": hd95, "stray_components": stray}


def run_diagnostics(cases: list[str], gts: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for case in cases:
        for model in MODELS:
            pred = load_prediction(model, case)
            if pred.shape != gts[case].shape:
                raise ValueError(f"{model} {case} 形状不一致: {pred.shape} vs {gts[case].shape}")
            for i, organ in enumerate(ORGANS, start=1):
                info = diagnose_organ(pred == i, gts[case] == i)
                rows.append({"case": case, "model": model, "organ": organ, **info})
    return rows


def summarize(rows: list[dict]) -> None:
    print("\n各模型的器官-病例对统计（共 %d 对/模型）：" % (len(rows) // len(MODELS)))
    print(f"{'model':<10}{'both':>6}{'miss':>6}{'fp_only':>9}{'absent':>8}{'mean_stray':>12}{'HD95_224(both)':>17}")
    for model in MODELS:
        sub = [r for r in rows if r["model"] == model]
        cats = {c: sum(1 for r in sub if r["category"] == c) for c in ["both", "miss", "fp_only", "absent"]}
        stray = np.mean([r["stray_components"] for r in sub])
        hd = [r["hd95_224"] for r in sub if r["category"] == "both"]
        print(f"{model:<10}{cats['both']:>6}{cats['miss']:>6}{cats['fp_only']:>9}{cats['absent']:>8}"
              f"{stray:>12.2f}{np.mean(hd):>17.2f}")


def write_diagnostics(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------- 可视化 ----------------

def pick_slices(gt: np.ndarray, k: int = 3) -> list[int]:
    n_cls = np.array([len(np.unique(s[s > 0])) for s in gt])
    cand = np.where(n_cls >= max(3, n_cls.max() - 1))[0]
    if len(cand) == 0:
        cand = np.arange(gt.shape[0])
    idx = np.linspace(0, len(cand) - 1, k).round().astype(int)
    return [int(cand[i]) for i in idx]


def plot_case(case: str, img: np.ndarray, gt: np.ndarray, preds: dict[str, np.ndarray], out_path: Path) -> None:
    slices = pick_slices(gt)
    cols = ["Image", "Ground truth"] + MODELS
    cmap = ListedColormap(LABEL_COLORS)
    fig, axes = plt.subplots(len(slices), len(cols), figsize=(2.0 * len(cols), 2.1 * len(slices)))
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    for r, z in enumerate(slices):
        panels = [img[z], gt[z]] + [preds[m][z] for m in MODELS]
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img[z], cmap="gray", vmin=0, vmax=1)
            if c > 0:
                mask = np.ma.masked_where(panel == 0, panel)
                ax.imshow(mask, cmap=cmap, vmin=0.5, vmax=N_CLASSES - 0.5, alpha=0.5, interpolation="nearest")
            if r == 0:
                ax.set_title(cols[c])
            if c == 0:
                ax.set_ylabel(f"slice {z}")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"{case}: ground truth vs. predictions (224x224)", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------- 主流程 ----------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vis_cases", nargs="+", default=DEFAULT_VIS_CASES)
    parser.add_argument("--skip_infer", action="store_true", help="跳过推理，直接使用已保存的预测")
    parser.add_argument("--_infer", dest="infer_model", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cases", nargs="+", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.infer_model:
        _infer_main(args.infer_model, args.cases or load_cases())
        return

    cases = load_cases()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.skip_infer:
        for model in ["TransUNet", "Swin-UNet"]:
            run_inference_subprocess(model, cases)

    gts = {}
    imgs = {}
    for case in cases:
        imgs[case], gts[case] = load_case(case)

    rows = run_diagnostics(cases, gts)
    write_diagnostics(rows, OUT_DIR / "diagnostics.csv")
    summarize(rows)

    for case in args.vis_cases:
        preds = {m: load_prediction(m, case) for m in MODELS}
        plot_case(case, imgs[case], gts[case], preds, OUT_DIR / f"vis_{case}.png")
        print(f"已保存可视化: {OUT_DIR / f'vis_{case}.png'}")


if __name__ == "__main__":
    main()
