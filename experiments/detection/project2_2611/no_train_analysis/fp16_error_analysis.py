"""FP16 输出误差的分通道分析（用于解释 mAP50-95 的差异来源）。

对测试集前 N 张图（与 trt_map_eval 相同的预处理），比较 BN 折叠 PyTorch（参考，TF32 关闭）与
TRT FP32 / FP16 的原始检测输出 out0（形状 1×25200×6：x,y,w,h,obj,cls，坐标单位为像素）：
1. 分通道的最大 / 平均绝对误差；
2. 只看高置信锚点（参考 obj > 0.25）的坐标误差；
3. 坐标量化检查：FP16 输出中 >=256 的坐标是否落在 0.25 的网格上（FP16 在 [256,512) 的步长为 0.25）。

用法（仓库根目录）：python experiments/detection/project2_2611/no_train_analysis/fp16_error_analysis.py
输出：outputs/map_eval/fp16_error_analysis.json
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_PROJ = _HERE.parent
_ROOT = _PROJ.parents[2]
for p in (_ROOT, _PROJ, _PROJ / "analysis_scripts", _HERE):
    sys.path.insert(0, str(p))

from trainer import build_config, model_output_dir  # noqa: E402
from trainer.data import YoloDetDataset  # noqa: E402
import redundancy_analysis as ra  # noqa: E402
from bn_fold import fold_bn_inplace  # noqa: E402
from trt_deploy_bench import TRTRunner  # noqa: E402
from trt_map_eval import DATA_DIR, MODELS, TRT_DIR  # noqa: E402

OUT = _HERE / "outputs" / "map_eval"
N_IMG = 40
CH = ["x", "y", "w", "h", "obj", "cls"]


def stats(ref: np.ndarray, got: np.ndarray, mask: np.ndarray | None = None) -> dict:
    d = np.abs(ref - got)
    if mask is not None:
        d = d[mask]
    return {"max": float(d.max()), "mean": float(d.mean())}


def main() -> None:
    t0 = time.time()
    dev = torch.device("cuda")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    res: dict[str, dict] = {}
    for label, m in MODELS.items():
        cfg = build_config(m["key"])
        run_dir = model_output_dir(cfg) / m["spec"]["run"]
        ckpt = torch.load(run_dir / "checkpoints" / "best.pth", map_location="cpu", weights_only=False)
        img_size = int(ckpt["img_size"])
        orig = ra.load_model(m["spec"]).eval().to(dev)
        folded = copy.deepcopy(orig).eval()
        fold_bn_inplace(folded)
        r16 = TRTRunner((TRT_DIR / f"{m['tag']}_fp16.engine").read_bytes())
        r32 = TRTRunner((TRT_DIR / f"{m['tag']}_fp32.engine").read_bytes())
        ds = YoloDetDataset(DATA_DIR / "images" / "test", DATA_DIR / "labels" / "test", img_size, train=False)

        acc = {p: {"all": {c: [] for c in CH}, "hi": {c: [] for c in CH}} for p in ("fp32", "fp16")}
        q_total = {"ge256": 0, "on025": 0, "ge512": 0, "on05": 0}
        q_fp16 = dict(q_total)
        for i in range(N_IMG):
            img, _ = ds[i]
            x = img[None].to(dev).float() / 255.0
            with torch.no_grad():
                ref = folded(x)[0][0].float().cpu().numpy()
            outs = {}
            for p, r in (("fp32", r32), ("fp16", r16)):
                r.set_input(x)
                r.run()
                torch.cuda.synchronize()
                outs[p] = dict(r.outputs)["out0"][0].float().cpu().numpy()
            hi = ref[:, 4] > 0.25
            for p in ("fp32", "fp16"):
                got = outs[p]
                for ci, c in enumerate(CH):
                    acc[p]["all"][c].append(stats(ref[:, ci], got[:, ci])["max"])
                    if hi.any():
                        acc[p]["hi"][c].append(stats(ref[:, ci], got[:, ci], hi)["max"])
            # 量化检查：coords 绝对值 >= 256 时是否为 0.25 / 0.5 的整数倍
            coords = outs["fp16"][:, :4].reshape(-1)
            big = coords[np.abs(coords) >= 256]
            q_fp16["ge256"] += big.size
            q_fp16["on025"] += int(np.sum(np.isclose((big * 4) % 1, 0) | np.isclose((big * 4) % 1, 1)))
            big512 = coords[np.abs(coords) >= 512]
            q_fp16["ge512"] += big512.size
            q_fp16["on05"] += int(np.sum(np.isclose((big512 * 2) % 1, 0) | np.isclose((big512 * 2) % 1, 1)))

        summary = {}
        for p in ("fp32", "fp16"):
            summary[p] = {
                "max_abs_err_all_anchors": {c: float(np.max(acc[p]["all"][c])) for c in CH},
                "mean_of_per_image_max_all": {c: float(np.mean(acc[p]["all"][c])) for c in CH},
                "max_abs_err_obj_gt_0.25": {c: float(np.max(acc[p]["hi"][c])) for c in CH},
            }
        summary["fp16_coord_quantization"] = {
            "coords_abs_ge_256": q_fp16["ge256"],
            "fraction_on_0.25_grid_ge256": q_fp16["on025"] / max(q_fp16["ge256"], 1),
            "coords_abs_ge_512": q_fp16["ge512"],
            "fraction_on_0.5_grid_ge512": q_fp16["on05"] / max(q_fp16["ge512"], 1),
        }
        res[label] = summary
        print(f"\n== {label}（{N_IMG} 张测试图，参考 = BN 折叠 PyTorch，TF32 关闭）", flush=True)
        for p in ("fp32", "fp16"):
            print(f"  {p}：全部锚点最大误差 " + " ".join(
                f"{c}={summary[p]['max_abs_err_all_anchors'][c]:.4f}" for c in CH), flush=True)
            print(f"       obj>0.25 锚点最大误差 " + " ".join(
                f"{c}={summary[p]['max_abs_err_obj_gt_0.25'][c]:.4f}" for c in CH), flush=True)
        q = summary["fp16_coord_quantization"]
        print(f"  FP16 坐标量化：|v|>=256 的 {q['coords_abs_ge_256']} 个中，"
              f"{q['fraction_on_0.25_grid_ge256'] * 100:.1f}% 落在 0.25 网格；"
              f"|v|>=512 的 {q['coords_abs_ge_512']} 个中，{q['fraction_on_0.5_grid_ge512'] * 100:.1f}% 落在 0.5 网格",
              flush=True)
        del orig, folded, r16, r32
        torch.cuda.empty_cache()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fp16_error_analysis.json").write_text(
        json.dumps({"n_images": N_IMG, "reference": "BN 折叠 PyTorch（TF32 关闭）", "results": res,
                    "seconds": round(time.time() - t0)}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已保存 {OUT / 'fp16_error_analysis.json'}（总耗时 {time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
