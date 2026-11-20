"""无训练：五个模型（每组 3 个 seed）的目标对比度分析。

样本：test 集小目标 60 张（pick_samples，seed 0，与 diagnose_p4_cbam 相同）。
对每个 checkpoint，在三个位置计算 GT 框的对比度 = (框内均值 − 外圈均值) / 全图标准差（通道均值图）：
- P4 下采样输出：backbone.stage4.0（步长 16）
- P4 块输出：backbone.stage4.1（步长 16）
- P3 最终输出：有 CBAM 时取 neck.cbam_p3，否则取 neck.c3_p3（步长 8）；cbam_mode=none 时 cbam_p3 为恒等，因此两者一致
另记录：框内分位 pct、RMS 分位 pct_rms、P3 外圈均值 > 框内均值的通道比例 ring_dom_ch。

用法（仓库根目录）：python experiments/detection/project2_2611/no_train_analysis/object_contrast_five.py
输出：outputs/object_contrast/object_contrast_five_2026-11-20.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_PROJ = _HERE.parent  # project2_2611
sys.path.insert(0, str(_PROJ))
sys.path.insert(0, str(_PROJ / "analysis_scripts"))
sys.path.insert(0, str(_HERE))
import redundancy_analysis as ra  # noqa: E402
from diagnose_p4_cbam import S3, S4, pick_samples, ring_dominant_frac, stats_mean_rms  # noqa: E402
from size_ap_five import ALL_RUNS  # noqa: E402
from trainer.config import ROOT  # noqa: E402
from trainer.data import YoloDetDataset  # noqa: E402

OUT = _HERE / "outputs" / "object_contrast"
LAYERS = ["backbone.stage4.0", "backbone.stage4.1", "neck.c3_p3"]


def contrast(st: dict) -> float:
    return (st["box"] - st["ring"]) / max(st["std"], 1e-9)


def group_stats(vals: list[float]) -> tuple[float, float]:
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = ROOT / "data" / "Det_Fly"
    ds = YoloDetDataset(data_dir / "images" / "test", data_dir / "labels" / "test", 640, train=False)
    stems_all = [p.stem for p in ds.paths]
    small, _ = pick_samples(data_dir)
    print(f"小目标样本 {len(small)} 张，设备 {device.type}", flush=True)

    keys = [(g, s, spec) for g, items in ALL_RUNS.items() for s, spec in items]
    models = {(g, s): ra.load_model(spec).eval().to(device) for g, s, spec in keys}
    rec = {k: {"P4down": [], "P4block": [], "P3final": [], "P3pct": [], "P3rms": [], "P3ring": [], "final_layer": None}
           for k in models}

    for stem in small:
        img_t, lab = ds[stems_all.index(stem)]
        _, cx, cy, bw, bh = lab[0].tolist()
        box = ((cx - bw / 2) * 640, (cy - bh / 2) * 640, (cx + bw / 2) * 640, (cy + bh / 2) * 640)
        for k, m in models.items():
            mods = dict(m.named_modules())
            names = LAYERS + (["neck.cbam_p3"] if "neck.cbam_p3" in mods else [])
            cap: dict = {}
            hs = [mods[n].register_forward_hook(ra.make_hook(cap, n)) for n in names]
            with torch.no_grad():
                m(img_t[None].float().to(device) / 255.0)
            for h in hs:
                h.remove()
            final = "neck.cbam_p3" if "neck.cbam_p3" in cap else "neck.c3_p3"
            r = rec[k]
            r["P4down"].append(contrast(stats_mean_rms(cap["backbone.stage4.0"][0], box, S4)))
            r["P4block"].append(contrast(stats_mean_rms(cap["backbone.stage4.1"][0], box, S4)))
            sf = stats_mean_rms(cap[final][0], box, S3)
            r["P3final"].append(contrast(sf))
            r["P3pct"].append(sf["pct"])
            r["P3rms"].append(sf["pct_rms"])
            r["P3ring"].append(ring_dominant_frac(cap[final][0], box, S3))
            r["final_layer"] = final

    per_run = {}
    for (g, s), r in rec.items():
        per_run[f"{g} {s}"] = {
            "group": g, "seed": s, "final_layer": r["final_layer"],
            "P4down_contrast_median": float(np.median(r["P4down"])),
            "P4block_contrast_median": float(np.median(r["P4block"])),
            "P3final_contrast_median": float(np.median(r["P3final"])),
            "P3final_contrast_iqr": [float(np.percentile(r["P3final"], 25)), float(np.percentile(r["P3final"], 75))],
            "P3final_frac_contrast_le0": float(np.mean(np.array(r["P3final"]) <= 0)),
            "P3final_pct_median": float(np.median(r["P3pct"])),
            "P3final_rms_pct_median": float(np.median(r["P3rms"])),
            "P3_ring_dominant_channel_frac_median": float(np.median(r["P3ring"])),
        }

    groups = {}
    for g in ALL_RUNS:
        rs = [v for v in per_run.values() if v["group"] == g]
        groups[g] = {key: group_stats([v[key] for v in rs]) for key in [
            "P4down_contrast_median", "P4block_contrast_median", "P3final_contrast_median",
            "P3final_pct_median", "P3final_rms_pct_median", "P3_ring_dominant_channel_frac_median",
            "P3final_frac_contrast_le0"]}

    print(f"\n{'组':<22}{'P4下采样对比度':>16}{'P4块对比度':>14}{'P3最终对比度':>16}{'P3框内分位':>12}"
          f"{'P3 RMS分位':>12}{'外圈>框内通道比':>16}")
    for g, v in groups.items():
        f = lambda key: f"{v[key][0]:+.2f}±{v[key][1]:.2f}" if "contrast" in key else f"{v[key][0]:.3f}±{v[key][1]:.3f}"  # noqa: E731
        print(f"{g:<22}{f('P4down_contrast_median'):>16}{f('P4block_contrast_median'):>14}"
              f"{f('P3final_contrast_median'):>16}{f('P3final_pct_median'):>12}{f('P3final_rms_pct_median'):>12}"
              f"{f('P3_ring_dominant_channel_frac_median'):>16}")

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "object_contrast_five_2026-11-20.json"
    out.write_text(json.dumps({"n_samples": len(small), "runs": per_run, "groups": groups,
                               "seconds": round(time.time() - t0)}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已保存 {out}（总耗时 {time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
