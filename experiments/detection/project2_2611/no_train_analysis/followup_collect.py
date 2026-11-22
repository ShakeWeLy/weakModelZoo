"""后续工作 3、5：收集并存档可复现的 ONNX、引擎、TRT Engine Inspector 与 Nsight Systems 结果。

- ONNX：折叠前、置换折叠后、无 Shuffle 上限（非等价）、精简导出（只输出 out0 + float 网格）
- 引擎：原始 FP16、置换折叠 FP16、无 Shuffle 上限 FP16、置换折叠 DETAILED、精简 DETAILED
- Inspector：DETAILED 引擎的逐层 JSON（含 ONNX 来源）；其它引擎的逐层名称列表
- nsys：各引擎的 kernel 汇总 CSV，并汇总为每次推理 kernel 数与耗时
- manifest.json：所有文件的 sha256、大小，以及 SGC/YOLOv5s 检查点的 sha256

用法（仓库根目录）：python experiments/detection/project2_2611/no_train_analysis/followup_collect.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJ = _HERE.parent
_ROOT = _PROJ.parents[2]
for p in (_ROOT, _PROJ, _HERE):
    sys.path.insert(0, str(p))

OUT = _HERE / "outputs" / "followup"
ART = OUT / "artifacts"
N = 100  # nsys 采集的推理次数

SOURCES = {
    "onnx/sgc_original_folded.onnx": _HERE / "outputs" / "trt" / "sgc_folded.onnx",
    "onnx/sgc_shuffle_folded_equivalent.onnx": _HERE / "outputs" / "shuffle_fold" / "sgc_fold.onnx",
    "onnx/sgc_noshuffle_upper_bound_NOT_EQUIVALENT.onnx": _HERE / "outputs" / "shuffle_ablation" / "sgc_noshuffle_folded.onnx",
    "onnx/sgc_lean_out0_floatgrid.onnx": _HERE / "outputs" / "reformat" / "poc_lean.onnx",
    "onnx/yolov5s_original_folded.onnx": _HERE / "outputs" / "trt" / "yolov5s_folded.onnx",
    "engines/sgc_original_fp16.engine": _HERE / "outputs" / "trt" / "sgc_fp16.engine",
    "engines/sgc_shuffle_folded_fp16.engine": _HERE / "outputs" / "shuffle_fold" / "sgc_fold_fp16.engine",
    "engines/sgc_noshuffle_upper_bound_fp16.engine": _HERE / "outputs" / "shuffle_ablation" / "sgc_noshuffle_fp16.engine",
    "engines/sgc_shuffle_folded_fp16_detailed.engine": _HERE / "outputs" / "reformat" / "sgc_fold_fp16_detailed.engine",
    "engines/sgc_lean_fp16_detailed.engine": _HERE / "outputs" / "reformat" / "poc_lean_fp16_detailed.engine",
    "inspector/sgc_shuffle_folded_detailed_layers.json": _HERE / "outputs" / "reformat" / "layers_verbose.json",
    "inspector/sgc_lean_detailed_layers.json": _HERE / "outputs" / "reformat" / "poc_lean_layers.json",
    "fx/original_graph.txt": OUT / "fx" / "original_graph.txt",
    "fx/rewritten_graph.txt": OUT / "fx" / "rewritten_graph.txt",
    "fx/shuffle_nodes.json": OUT / "shuffle_nodes.json",
    "fx/conv_perm_sites.json": OUT / "conv_perm_sites.json",
}

NSYS_DIRS = {
    "original": _HERE / "outputs" / "shuffle_ablation" / "nsys" / "orig_kern_cuda_gpu_kern_sum.csv",
    "noshuffle_upper_bound": _HERE / "outputs" / "shuffle_ablation" / "nsys" / "noshuffle_kern_cuda_gpu_kern_sum.csv",
    "shuffle_folded": _HERE / "outputs" / "shuffle_fold" / "nsys" / "fold_kern_cuda_gpu_kern_sum.csv",
    "shuffle_folded_detailed": _HERE / "outputs" / "reformat" / "nsys" / "detailed_kern_cuda_gpu_kern_sum.csv",
    "lean_detailed": _HERE / "outputs" / "reformat" / "nsys" / "lean_kern_cuda_gpu_kern_sum.csv",
}
NSYS_REPS = {
    "nsys/original.nsys-rep": _HERE / "outputs" / "shuffle_fold" / "nsys" / "orig.nsys-rep",
    "nsys/shuffle_folded.nsys-rep": _HERE / "outputs" / "shuffle_fold" / "nsys" / "fold.nsys-rep",
    "nsys/noshuffle_upper_bound.nsys-rep": _HERE / "outputs" / "shuffle_fold" / "nsys" / "noshuffle.nsys-rep",
    "nsys/shuffle_folded_detailed.nsys-rep": _HERE / "outputs" / "reformat" / "nsys" / "detailed.nsys-rep",
    "nsys/lean_detailed.nsys-rep": _HERE / "outputs" / "reformat" / "nsys" / "lean.nsys-rep",
}

def ckpt_paths() -> dict[str, Path]:
    from trainer import build_config, model_output_dir
    specs = {
        "SGC 无注意力 best.pth": ("sgc_yolov5s_c3_sgdown_gbot_dw5",
                                 "2026-11-17_015_SGC_c3_sgdown_gbot_dw5_s_noDFC_cbamNone_seed42_V1"),
        "YOLOv5s best.pth": ("yolov5s", "2026-10-30_002_YOLOv5s_V3"),
    }
    return {k: model_output_dir(build_config(m)) / run / "checkpoints" / "best.pth" for k, (m, run) in specs.items()}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def kern_summary(csv_path: Path) -> dict:
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    inst = sum(int(r["Instances"]) for r in rows)
    tot = sum(float(r["Total Time (ns)"]) for r in rows)
    reformat_n = reformat_ms = layout_n = layout_ms = 0.0
    for r in rows:
        name = r["Name"].lower()
        t = float(r["Total Time (ns)"])
        if "reformat" in name or "copypacked" in name or "copyvectorized" in name:
            reformat_n += int(r["Instances"]); reformat_ms += t
        elif "nchwto" in name or "nhwcto" in name:
            layout_n += int(r["Instances"]); layout_ms += t
    return {"kernels_per_inference": inst / N, "kernel_time_ms_per_inference": tot / N / 1e6,
            "distinct_kernel_names": len(rows),
            "reformat_kernels_per_inference": reformat_n / N, "reformat_ms_per_inference": reformat_ms / N / 1e6,
            "layout_kernels_per_inference": layout_n / N, "layout_ms_per_inference": layout_ms / N / 1e6}


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"files": {}, "checkpoints": {}, "nsys_summary": {}}
    for rel, src in {**SOURCES, **NSYS_REPS}.items():
        dst = ART / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            print(f"缺失：{src}", flush=True)
            manifest["files"][rel] = {"missing_source": str(src)}
            continue
        shutil.copy2(src, dst)
        manifest["files"][rel] = {"sha256": sha256(dst), "bytes": dst.stat().st_size,
                                  "source": str(src.relative_to(_ROOT)) if _ROOT in src.parents else str(src)}
        print(f"已存档 {rel}", flush=True)

    # 非 DETAILED 引擎的逐层名称（Inspector 默认输出）
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.ERROR)
    for rel in ["engines/sgc_original_fp16.engine", "engines/sgc_shuffle_folded_fp16.engine",
                "engines/sgc_noshuffle_upper_bound_fp16.engine"]:
        eng = trt.Runtime(logger).deserialize_cuda_engine((ART / rel).read_bytes())
        info = json.loads(eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
        out = ART / "inspector" / (Path(rel).stem + "_layer_names.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"n_layers": len(info["Layers"]), "layers": info["Layers"],
                                   "bindings": info.get("Bindings", [])}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        manifest["files"][f"inspector/{out.name}"] = {"sha256": sha256(out), "bytes": out.stat().st_size,
                                                      "n_layers": len(info["Layers"])}
        print(f"Inspector {out.name}：{len(info['Layers'])} 层", flush=True)

    for name, csv_path in NSYS_DIRS.items():
        if csv_path.exists():
            dst = ART / "nsys" / csv_path.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(csv_path, dst)
            manifest["nsys_summary"][name] = kern_summary(csv_path)
    (ART / "nsys" / "kernel_summary.json").write_text(json.dumps(manifest["nsys_summary"], ensure_ascii=False,
                                                                indent=1), encoding="utf-8")

    for label, p in ckpt_paths().items():
        if p.exists():
            manifest["checkpoints"][label] = {"path": str(p.relative_to(_ROOT)), "sha256": sha256(p)}
        else:
            manifest["checkpoints"][label] = {"path": str(p), "missing": True}
    (ART / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已写入 {ART / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
