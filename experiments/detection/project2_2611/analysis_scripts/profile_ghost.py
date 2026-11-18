"""按模块类别统计 MACs 与参数量（640 输入，batch=1），用于决定轻量化的优化方向。

MACs 口径与报告一致：每个 Conv2d 的 输出元素数 × (Cin/groups) × kh × kw（不含 BN / SiLU / 上采样 / 通道洗牌）。

用法（仓库根目录）：
    CUDA_VISIBLE_DEVICES="" python experiments/detection/project2_2611/profile_ghost.py --model ours
输出：控制台表格 + profile_ghost/<model>.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import redundancy_analysis as ra  # noqa: E402
from diagnose_p4_cbam import MODELS  # noqa: E402

OUT = Path(__file__).resolve().parent / "profile_ghost"


def category(name: str) -> str:
    if re.match(r"^backbone\.stage\d\.0\.", name):
        return "下采样（SGhost 3×3 s2）" if ("cheap" in name or "conv" in name) else "下采样 其它"
    if "SGhostConv1" in name:
        return "SG1：1×1 主卷积 + dw 幽灵" if "cheap" not in name else "SG1：1×1 主卷积 + dw 幽灵"
    if "SGhostConv2" in name:
        return "SG2：3×3 主卷积 + dw5 幽灵"
    if "DFCAttention" in name:
        return "DFC 门控（1×5 / 5×1 DW）"
    if "DWConv2" in name:
        return "残差 3×3 DW"
    if name.startswith("backbone"):
        return "骨干 其它（cv1/cv2/cv3 投影等）"
    if name.startswith("neck"):
        return "颈部（PAN / CBAM / 下采样）"
    if name.startswith("detect"):
        return "检测头"
    return "其它"


def sub_kind(name: str) -> str:
    if "cheap_conv" in name:
        return "幽灵（cheap）"
    if name.endswith(".conv") or "SGhost" in name or "stage" in name:
        return "本征（primary）"
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="ours")
    args = ap.parse_args()
    spec = MODELS[args.model]
    model = ra.load_model(spec)
    rows = []
    hooks = []

    def make_hook(name: str):
        def _h(m: nn.Module, _i, out):
            kh, kw = m.kernel_size
            macs = out.numel() * (m.in_channels // m.groups) * kh * kw
            params = sum(p.numel() for p in m.parameters())
            rows.append({"name": name, "cat": category(name), "macs": int(macs), "params": int(params),
                         "k": f"{kh}x{kw}", "groups": int(m.groups), "cin": int(m.in_channels),
                         "cout": int(m.out_channels), "out_hw": list(out.shape[-2:])})
        return _h

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(make_hook(name)))
    with torch.no_grad():
        model(torch.zeros(1, 3, 640, 640))
    for h in hooks:
        h.remove()

    total_macs = sum(r["macs"] for r in rows)
    total_params = sum(p.numel() for p in model.parameters())
    cats: dict[str, dict] = {}
    for r in rows:
        c = cats.setdefault(r["cat"], {"macs": 0, "params": 0, "n": 0})
        c["macs"] += r["macs"]
        c["params"] += r["params"]
        c["n"] += 1
    print(f"{spec['label']}：卷积 MACs 合计 {total_macs / 1e9:.2f} G，卷积参数 {sum(r['params'] for r in rows) / 1e6:.2f} M，"
          f"模型总参数 {total_params / 1e6:.2f} M")
    print(f"{'类别':<30}{'层数':>5}{'MACs(G)':>10}{'占比':>8}{'参数(M)':>10}{'占比':>8}")
    for c, v in sorted(cats.items(), key=lambda kv: -kv[1]["macs"]):
        print(f"{c:<30}{v['n']:>5}{v['macs'] / 1e9:>10.3f}{100 * v['macs'] / total_macs:>7.1f}%"
              f"{v['params'] / 1e6:>10.3f}{100 * v['params'] / max(sum(r['params'] for r in rows), 1):>7.1f}%")

    print("\n各类别内 MACs 最大的 5 层：")
    for c in sorted(cats, key=lambda c: -cats[c]["macs"]):
        print(f"  [{c}]")
        for r in sorted([r for r in rows if r["cat"] == c], key=lambda r: -r["macs"])[:5]:
            print(f"    {r['macs'] / 1e9:7.3f} G  {r['k']:<5} g={r['groups']:<4} {r['cin']:>4}→{r['cout']:<4} "
                  f"输出 {r['out_hw'][0]}×{r['out_hw'][1]}  {r['name']}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.model}.json").write_text(json.dumps({
        "model": args.model, "total_conv_gmacs": total_macs / 1e9,
        "total_params_m": total_params / 1e6,
        "categories": cats, "layers": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已保存 {OUT / (args.model + '.json')}")


if __name__ == "__main__":
    main()
