"""第一步：定位 SGC 中所有 ChannelShuffle，并分析每个 Shuffle 输出的消费者（基于 torch.fx 计算图）。

- 对 BN 折叠后的 SGC（eval）做 fx 符号追踪；Detect 作为叶子模块（不展开其内部的 shape 逻辑）。
- 对每个 ChannelShuffle 节点，列出其所有直接消费者（节点类型、目标模块/函数、Conv 的 groups 与输入通道）。
- 按消费者类型给出初步可折叠性：单一普通 Conv 消费者 → 可吸收；Add/Mul/Concat/输出 → 需要传播或保留。

用法（仓库根目录）：python experiments/detection/project2_2611/no_train_analysis/shuffle_graph_analysis.py
输出：outputs/shuffle_fold/graph_analysis.json
"""

from __future__ import annotations

import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.fx as fx
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_PROJ = _HERE.parent
_ROOT = _PROJ.parents[2]
for p in (_ROOT, _PROJ, _PROJ / "analysis_scripts", _HERE):
    sys.path.insert(0, str(p))

import redundancy_analysis as ra  # noqa: E402
from bn_fold import fold_bn_inplace  # noqa: E402

OUT = _HERE / "outputs" / "shuffle_fold"
SPEC = {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5",
        "run": "2026-11-17_015_SGC_c3_sgdown_gbot_dw5_s_noDFC_cbamNone_seed42_V1"}


class LeafTracer(fx.Tracer):
    """把 Detect 与 ChannelShuffle 作为叶子，避免追踪其内部逻辑。"""

    def is_leaf_module(self, m: nn.Module, qualname: str) -> bool:
        if type(m).__name__ in ("Detect", "ChannelShuffle"):
            return True
        return super().is_leaf_module(m, qualname)


def describe(gm: fx.GraphModule, node: fx.Node) -> str:
    if node.op == "call_module":
        m = gm.get_submodule(node.target)
        s = type(m).__name__
        if isinstance(m, nn.Conv2d):
            s += f"(in={m.in_channels},out={m.out_channels},k={m.kernel_size[0]},g={m.groups})"
        return f"module:{node.target}:{s}"
    if node.op == "call_function":
        return f"fn:{getattr(node.target, '__name__', str(node.target))}"
    if node.op == "call_method":
        return f"method:{node.target}"
    return f"{node.op}:{node.target}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = ra.load_model(SPEC).eval()
    fold_bn_inplace(model)
    gm = fx.GraphModule(model, LeafTracer().trace(model))
    nodes = list(gm.graph.nodes)
    shuffles = [n for n in nodes if n.op == "call_module" and type(gm.get_submodule(n.target)).__name__ == "ChannelShuffle"]
    print(f"ChannelShuffle 节点数：{len(shuffles)}")
    rows = []
    kind_count = Counter()
    for s in shuffles:
        users = list(s.users)
        ulist = []
        for u in users:
            d = describe(gm, u) if u.op != "output" else "graph_output"
            ulist.append({"node": u.name, "desc": d})
        kinds = tuple(sorted({("conv" if "Conv2d" in x["desc"] else
                               "add" if ("add" in x["desc"].lower()) else
                               "mul" if ("mul" in x["desc"].lower()) else
                               "cat" if ("cat" in x["desc"].lower()) else
                               "output" if x["desc"] == "graph_output" else
                               "other") for x in ulist}))
        kind_count[kinds] += 1
        rows.append({"shuffle_node": s.name, "input_from": str(s.args[0]), "users": ulist, "kinds": list(kinds)})
        print(f"\n{s.name}  <-  {s.args[0]}")
        for x in ulist:
            print(f"    -> {x['node']:<40} {x['desc']}")
    print("\n消费者类型组合统计：")
    for k, v in kind_count.most_common():
        print(f"  {v:3d} 个 Shuffle  消费者类型 {list(k)}")
    (OUT / "graph_analysis.json").write_text(
        json.dumps({"n_shuffle": len(shuffles), "shuffles": rows,
                    "kind_counts": {str(list(k)): v for k, v in kind_count.items()}},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已保存 {OUT / 'graph_analysis.json'}")


if __name__ == "__main__":
    main()
