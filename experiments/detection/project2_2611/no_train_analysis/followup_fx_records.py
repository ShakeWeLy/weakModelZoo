"""后续工作 1–3：FX 节点明细、8 个 Conv 输出置换的模块位置、FX 原始图与改写图的存档。

产出（outputs/followup/）：
- shuffle_nodes.json / .md：30 个 ChannelShuffle 的 FX 节点、输入、全部消费者（节点名、FX target、类型、参数）、折叠状态
- conv_perm_sites.json / .md：8 个输出置换被调整的 Conv 的模块路径、所在父模块、服务的残差 Add 及置换一致性检查
- fx/original_graph.txt、fx/rewritten_graph.txt：折叠前后的 FX 图（graph 文本与 generated code）
- fx/*.json：节点列表（名称、op、target、args）

用法（仓库根目录）：python experiments/detection/project2_2611/no_train_analysis/followup_fx_records.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_PROJ = _HERE.parent
_ROOT = _PROJ.parents[2]
for p in (_ROOT, _PROJ, _PROJ / "analysis_scripts", _HERE):
    sys.path.insert(0, str(p))

from shuffle_fold import FoldEngine, build, fold_graph  # noqa: E402

OUT = _HERE / "outputs" / "followup"
FX = OUT / "fx"


def node_desc(gm, n) -> dict:
    if n.op == "call_module":
        m = gm.get_submodule(n.target)
        d = {"target": n.target, "type": type(m).__name__}
        if type(m).__name__ == "Conv2d":
            d.update(in_ch=m.in_channels, out_ch=m.out_channels, k=m.kernel_size[0], groups=m.groups)
        if type(m).__name__ == "ChannelShuffle":
            d.update(groups=m.groups)
        return d
    if n.op == "call_function":
        return {"target": getattr(n.target, "__name__", str(n.target)), "type": "function"}
    return {"target": str(n.target), "type": n.op}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FX.mkdir(parents=True, exist_ok=True)

    gm, C, _ = build()
    # 1) 折叠前的 FX 原始图
    (FX / "original_graph.txt").write_text(str(gm.graph) + "\n\n# ---- generated code ----\n" + gm.code,
                                           encoding="utf-8")
    orig_nodes = [{"name": n.name, "op": n.op, "target": str(n.target),
                   "args": [str(a) for a in n.args], "users": [u.name for u in n.users]} for n in gm.graph.nodes]
    (FX / "original_nodes.json").write_text(json.dumps(orig_nodes, ensure_ascii=False, indent=1), encoding="utf-8")

    eng = FoldEngine(gm, C)
    eng.resolve()
    pi = {k: v.copy() for k, v in eng.pi.items()}
    kind = eng.kind

    # 2) 30 个 ChannelShuffle 的节点与消费者明细
    shuffles = []
    for n in gm.graph.nodes:
        if kind(n) != "shuffle":
            continue
        src = n.args[0]
        users = []
        for u in n.users:
            d = node_desc(gm, u) if u.op in ("call_module", "call_function") else {"target": u.op, "type": "output"}
            users.append({"node": u.name, "op": u.op, "kind": kind(u), **d})
        kinds = sorted({u["kind"] for u in users})
        if kinds == ["conv"]:
            status = "单一/多 Conv 消费者：吸收进消费者卷积输入权重"
        elif kinds == ["add"]:
            status = "残差 Add：置换由 C3 cv1 的输出置换与残差链（DW/Bottleneck）提供"
        elif "cat" in kinds:
            status = "Concat（可能经 Upsample）：置换并入 Concat 通道块，由下游 Conv 吸收"
        else:
            status = "/".join(kinds)
        shuffles.append({
            "shuffle_node": n.name,
            "fx_target": n.target,
            "input_node": src.name,
            "input_target": node_desc(gm, src).get("target", src.op),
            "input_kind": kind(src),
            "C": C[n.args[0].name],
            "groups": gm.get_submodule(n.target).groups,
            "output_perm_identity": bool(np.array_equal(pi[n.name], np.arange(C[n.name]))),
            "users": users,
            "status": status,
        })
    print(f"ChannelShuffle 节点：{len(shuffles)}", flush=True)

    # 3) 8 个 Conv 输出置换调整：模块位置、服务的 Add、置换一致性检查
    def find_source(name: str) -> str | None:
        """沿残差链向上游回溯，直到找到输出置换可自由选择的普通 Conv。"""
        seen = 0
        while seen < 500:
            seen += 1
            node = eng.node[name]
            k = kind(node)
            if k == "conv":
                return name
            if k in ("dw", "pass", "getitem"):
                name = node.args[0].name
            elif k == "add":
                cands = [a.name for a in node.args if kind(a) != "shuffle"]
                if not cands:
                    return None
                name = cands[0]
            else:
                return None
        return None

    adds = [n for n in gm.graph.nodes if kind(n) == "add"]
    add_rows = []
    for a in adds:
        sh = [x for x in a.args if kind(x) == "shuffle"]
        other = [x for x in a.args if kind(x) != "shuffle"]
        if not sh or not other:
            continue
        s = sh[0]
        src = find_source(other[0].name)
        match = bool(src is not None and np.array_equal(eng.conv_out.get(src, np.arange(C[src])), pi[s.name]))
        add_rows.append({"add": a.name, "shuffle_operand": s.name, "residual_source_conv": src,
                         "perm_equal_to_conv_output_perm": match})

    chosen = [k for k, v in eng.conv_out.items() if not np.array_equal(v, np.arange(v.size))]
    sites = []
    for cname in chosen:
        n = eng.node[cname]
        m = gm.get_submodule(n.target)
        parent_path = n.target.rsplit(".", 1)[0]          # BaseConv 包装层（如 backbone.stage2.1.cv1）
        grand_path = parent_path.rsplit(".", 1)[0]         # 父模块（如 backbone.stage2.1，即 SDGhost / VOVGCSPYolo）
        served = [r for r in add_rows if r["residual_source_conv"] == cname]
        consumers = [{"node": u.name, "kind": kind(u)} for u in n.users]
        sites.append({
            "conv_node": cname,
            "conv_target": n.target,
            "conv_shape": {"in": m.in_channels, "out": m.out_channels, "k": m.kernel_size[0], "groups": m.groups},
            "wrapper_target": parent_path,
            "parent_module_target": grand_path,
            "output_perm_applied": eng.conv_out[cname].tolist(),
            "consumers": consumers,
            "served_adds": [r["add"] for r in served],
            "perm_consistency_checks": [r["perm_equal_to_conv_output_perm"] for r in served],
        })
    print(f"输出置换调整的 Conv：{len(sites)}；残差 Add 映射 {len(add_rows)} 条", flush=True)

    (OUT / "shuffle_nodes.json").write_text(json.dumps(shuffles, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "conv_perm_sites.json").write_text(json.dumps({"sites": sites, "add_mapping": add_rows},
                                                         ensure_ascii=False, indent=1), encoding="utf-8")

    # 4) 改写后的 FX 图（同一次 fold_graph 的结果）
    ref, new, report = fold_graph()
    (FX / "rewritten_graph.txt").write_text(str(new.graph) + "\n\n# ---- generated code ----\n" + new.code,
                                            encoding="utf-8")
    rw_nodes = [{"name": n.name, "op": n.op, "target": str(n.target),
                 "args": [str(a) for a in n.args], "users": [u.name for u in n.users]} for n in new.graph.nodes]
    (FX / "rewritten_nodes.json").write_text(json.dumps(rw_nodes, ensure_ascii=False, indent=1), encoding="utf-8")
    n_shuffle_left = sum(1 for n in new.graph.nodes if n.op == "call_module" and
                         type(new.get_submodule(n.target)).__name__ == "ChannelShuffle")
    summary = {"original_nodes": len(orig_nodes), "rewritten_nodes": len(rw_nodes),
               "shuffle_removed": report["n_shuffle_removed"], "shuffle_left_in_rewritten_graph": n_shuffle_left,
               "conv_weights_rewritten": report["n_conv_weights_rewritten"],
               "conv_output_perm_sites": len(sites)}
    (OUT / "fx_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
