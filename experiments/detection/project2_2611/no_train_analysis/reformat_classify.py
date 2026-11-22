"""把 DETAILED 引擎中的 Reformat / copy 层按来源归类，并汇总逐层耗时（IProfiler，偏高，仅作相对参考）。

输入：outputs/reformat/layers_verbose.json（inspector 逐层信息）、layer_times.json（逐层耗时）
输出：outputs/reformat/classified.json，并打印分类汇总

分类规则（按层名与张量名）：
- 网络输入转换：层名含 Input Tensor 且输入为 images；
- 网络输出转换：输出为 out0/out1/out2/out3 或 onnx::Sigmoid_*（最终输出绑定）；
- Concat 输入拷贝：名称以 copy 结尾，且 ONNX 来源包含 Concat；
- Detect 头：张量或层名涉及 /detect/；
- 内部布局转换：卷积（NHWC8）或逐元素融合所需的格式转换；
- 其它。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs" / "reformat"


def classify(layer: dict) -> str:
    name = layer["Name"]
    ins = [x["Name"] for x in layer.get("Inputs", [])]
    outs = [x["Name"] for x in layer.get("Outputs", [])]
    meta = layer.get("Metadata", "")
    if "Input Tensor" in name and any(i == "images" for i in ins):
        return "网络输入转换（FP32 NCHW → FP16 NHWC）"
    if any(o.startswith("out") or o.startswith("onnx::Sigmoid") for o in outs) and ("Output Tensor" in name or name.endswith("copy_output") or " copy" in name or "copy_output" in name):
        return "网络输出转换（out0 / 原始特征图）"
    if (name.endswith(" copy") or "_copy_output" in name) and "Concat" in meta:
        return "Concat 输入拷贝"
    if "/detect/" in name or any("/detect/" in s for s in ins + outs):
        return "Detect 头格式/精度转换"
    if "Reformatting CopyNode" in name:
        return "内部布局转换（卷积 / 逐元素所需）"
    if name.endswith(" copy") or "_copy_output" in name:
        return "其它拷贝"
    return "其它"


def main() -> None:
    info = json.loads((OUT / "layers_verbose.json").read_text(encoding="utf-8"))
    times = json.loads((OUT / "layer_times.json").read_text(encoding="utf-8"))
    rows = []
    for l in info["Layers"]:
        n = l["Name"]
        if not (("reformat" in n.lower()) or ("copynode" in n.lower()) or n.endswith(" copy") or "_copy_output" in n):
            continue
        cat = classify(l)
        ms = times.get(n, 0.0)
        rows.append({"name": n, "category": cat, "ms": ms,
                     "inputs": [f"{x['Name']} [{x.get('Format/Datatype', '')}]" for x in l.get("Inputs", [])],
                     "outputs": [f"{x['Name']} [{x.get('Format/Datatype', '')}]" for x in l.get("Outputs", [])],
                     "onnx": [s.replace("[ONNX Layer: ", "").rstrip("]") for s in l.get("Metadata", "").split("\u001f")
                              if "ONNX Layer" in s]})
    cats: dict[str, dict] = defaultdict(lambda: {"n": 0, "n_timed": 0, "ms": 0.0, "items": []})
    for r in rows:
        c = cats[r["category"]]
        c["n"] += 1
        if r["ms"] > 0.001:
            c["n_timed"] += 1
        c["ms"] += r["ms"]
        c["items"].append(r)
    total_ms = sum(c["ms"] for c in cats.values())
    print(f"copy/reformat 层共 {len(rows)} 个；逐层耗时合计 {total_ms:.3f} ms（IProfiler）")
    print(f"{'类别':<34}{'层数':>6}{'有耗时层':>10}{'耗时ms':>10}{'占比':>8}")
    summary = {}
    for k, c in sorted(cats.items(), key=lambda kv: -kv[1]["ms"]):
        share = 100 * c["ms"] / total_ms if total_ms else 0
        print(f"{k:<34}{c['n']:>6}{c['n_timed']:>10}{c['ms']:>10.3f}{share:>7.1f}%")
        summary[k] = {"layers": c["n"], "timed_layers": c["n_timed"], "ms": c["ms"], "share_pct": share}
        for it in sorted(c["items"], key=lambda r: -r["ms"])[:6]:
            if it["ms"] > 0.001:
                print(f"      {it['ms']:.4f} ms  {it['name'][:70]}  {it['inputs'][:1]} -> {it['outputs'][:1]}")
    (OUT / "classified.json").write_text(json.dumps({"summary": summary, "layers": rows},
                                                    ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已保存 {OUT / 'classified.json'}")


if __name__ == "__main__":
    main()
