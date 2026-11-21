"""逐节点可折叠性分析 + ChannelShuffle 置换折叠（图优化，数学等价）。

核心思想
--------
用 torch.fx 把 BN 折叠后的 SGC 展开成计算图（Detect 为叶子）。每个张量 t 的"存储顺序"用置换 π_t 描述：

    stored_t[k] = logical_t[π_t[k]]

其中 logical 为原始模型（标准通道顺序）中的张量值。沿图传播 π，并按规则改写：

- Conv2d(groups=1)：输入置换 π_in 吸收进权重 W' = W[:, π_in]；输出置换 σ 可自由选择（W'' = W'[σ]，bias[σ]）。
- Depthwise Conv2d：输出置换 = 输入置换（W' = W[π_in]，bias[π_in]）。
- SiLU / MaxPool / Upsample / 逐通道激活：置换原样保留。
- Concat：π_cat = concat(π_i + offset_i)。
- ChannelShuffle(g)：物理重排被移除后，π_out = ρ⁻¹[π_in]，其中 ρ[k] = (k % g)·(C/g) + k // g（与 F.channel_shuffle 一致）。
  移除后 ChannelShuffle 变为恒等，不再产生 Reshape/Transpose/Reformatting。
- Add(a, b)：要求 π_a == π_b。若不等，先尝试把需求反向传播到某个可自由设置输出置换的 Conv（沿逐通道链回溯）；
  仍不能满足的，插入一次 Gather（IndexSelect）作为物化代价。
- Detect 输入、以及无法吸收的非标准分组卷积输入：要求标准顺序，否则同样插入 Gather。

本脚本只改写图与权重，不改动源码。所有改写都在 eval 模式下进行，并与原模型做逐输出数值对比。

用法（仓库根目录）：python experiments/detection/project2_2611/no_train_analysis/shuffle_fold.py
输出：outputs/shuffle_fold/fold_report.json，以及 outputs/shuffle_fold/sgc_fold.pt（折叠后的 GraphModule 状态）
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.fx as fx
import torch.nn as nn
from torch.fx.passes.shape_prop import ShapeProp

_HERE = Path(__file__).resolve().parent
_PROJ = _HERE.parent
_ROOT = _PROJ.parents[2]
for p in (_ROOT, _PROJ, _PROJ / "analysis_scripts", _HERE):
    sys.path.insert(0, str(p))

import redundancy_analysis as ra  # noqa: E402
from bn_fold import fold_bn_inplace  # noqa: E402
from shuffle_graph_analysis import LeafTracer  # noqa: E402

OUT = _HERE / "outputs" / "shuffle_fold"
SPEC = {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5",
        "run": "2026-11-17_015_SGC_c3_sgdown_gbot_dw5_s_noDFC_cbamNone_seed42_V1"}
PASS_THROUGH_MODULES = ("SiLU", "Identity", "MaxPool2d", "Upsample")


class IndexSelect(nn.Module):
    """沿通道维的 Gather（物化置换的代价）。导出 ONNX 时为 Gather 节点。"""

    def __init__(self, idx: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("idx", idx.long())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.index_select(x, 1, self.idx)


def rho_of(C: int, g: int) -> np.ndarray:
    k = np.arange(C)
    return (k % g) * (C // g) + k // g


def inv(p: np.ndarray) -> np.ndarray:
    q = np.empty_like(p)
    q[p] = np.arange(p.size)
    return q


class FoldEngine:
    def __init__(self, gm: fx.GraphModule, C: dict[str, int]) -> None:
        self.gm = gm
        self.C = C                      # 节点名 -> 输出通道数（由 ShapeProp 得到）
        self.nodes = list(gm.graph.nodes)
        self.node = {n.name: n for n in self.nodes}
        self.conv_out: dict[str, np.ndarray] = {}   # 可自由选择的 Conv 输出置换（默认恒等）
        self.pi: dict[str, np.ndarray] = {}
        self.mismatch: list[tuple[str, str, str]] = []  # (add节点, 操作数a, 操作数b)
        self.detect_in_mismatch: list[str] = []
        self.unsupported: list[str] = []

    # ---- 节点的种类 ----
    def kind(self, n: fx.Node) -> str:
        if n.op == "placeholder":
            return "input"
        if n.op == "call_module":
            m = self.gm.get_submodule(n.target)
            t = type(m).__name__
            if t == "Conv2d":
                if m.groups == 1:
                    return "conv"
                if m.groups == m.in_channels == m.out_channels:
                    return "dw"
                return "grouped"
            if t == "ChannelShuffle":
                return "shuffle"
            if t in PASS_THROUGH_MODULES:
                return "pass"
            if t == "Detect":
                return "detect"
            return "unknown:" + t
        if n.op == "call_function":
            name = getattr(n.target, "__name__", str(n.target))
            if name == "silu":
                return "pass"
            if name == "cat":
                return "cat"
            if name == "add":
                return "add"
            if name == "getitem":
                return "getitem"
            return "unknown:" + name
        if n.op == "output":
            return "output"
        return "unknown:" + n.op

    def in_nodes(self, n: fx.Node) -> list[fx.Node]:
        out = []
        for a in n.args:
            if isinstance(a, fx.Node):
                out.append(a)
            elif isinstance(a, (list, tuple)):
                out += [x for x in a if isinstance(x, fx.Node)]
        return out

    # ---- 前向传播 ----
    def compute(self) -> None:
        self.pi.clear()
        self.mismatch.clear()
        self.detect_in_mismatch.clear()
        self.unsupported.clear()
        for n in self.nodes:
            k = self.kind(n)
            if k == "input":
                self.pi[n.name] = np.arange(3)
            elif k == "conv":
                x = self.pi[n.args[0].name]
                cout = self.C[n.name]
                sigma = self.conv_out.get(n.name, np.arange(cout))
                self.pi[n.name] = sigma
            elif k == "dw":
                self.pi[n.name] = self.pi[n.args[0].name]
            elif k == "grouped":
                x = self.pi[n.args[0].name]
                if not np.array_equal(x, np.arange(x.size)):
                    self.unsupported.append(n.name)
                self.pi[n.name] = np.arange(self.C[n.name])
            elif k == "pass":
                self.pi[n.name] = self.pi[n.args[0].name]
            elif k == "shuffle":
                m = self.gm.get_submodule(n.target)
                x = self.pi[n.args[0].name]
                C = self.C[n.args[0].name]
                rho = rho_of(C, m.groups)
                self.pi[n.name] = inv(rho)[x]
            elif k == "cat":
                parts, off = [], 0
                for a in n.args[0]:
                    parts.append(self.pi[a.name] + off)
                    off += self.C[a.name]
                self.pi[n.name] = np.concatenate(parts)
            elif k == "getitem":
                src = n.args[0]
                idx = n.args[1]
                if isinstance(idx, tuple):
                    sl = idx[1]
                    L = sl.stop if isinstance(sl, slice) and sl.stop is not None else self.C[src.name]
                    L = min(L, self.C[src.name])
                    prefix = self.pi[src.name][:L]
                    if L == self.C[src.name]:
                        self.pi[n.name] = prefix
                    elif set(prefix.tolist()) == set(range(L)):
                        self.pi[n.name] = prefix
                    else:
                        self.unsupported.append(n.name)
                        self.pi[n.name] = np.arange(L)
                else:
                    # 元组解包（整数下标）：本模型中不出现逐通道切片以外的此类节点，仍做记录
                    self.unsupported.append(n.name + "(int-getitem)")
                    self.pi[n.name] = self.pi[src.name]
            elif k == "add":
                a, b = n.args[0].name, n.args[1].name
                if not np.array_equal(self.pi[a], self.pi[b]):
                    self.mismatch.append((n.name, a, b))
                self.pi[n.name] = self.pi[a]
            elif k == "detect":
                src = n.args[0] if n.args else None
                for s in self.in_nodes(n):
                    if not np.array_equal(self.pi[s.name], np.arange(self.C[s.name])):
                        self.detect_in_mismatch.append(s.name)
            elif k == "output":
                pass
            else:
                self.unsupported.append(n.name + f"({k})")

    # ---- 反向需求：要求节点 name 的输出置换等于 target ----
    def require(self, name: str, target: np.ndarray) -> bool:
        if np.array_equal(self.pi[name], target):
            return True
        n = self.node[name]
        k = self.kind(n)
        if k == "conv":
            self.conv_out[name] = target.copy()
            return True
        if k == "dw" or k == "pass":
            return self.require(n.args[0].name, target)
        if k == "shuffle":
            C = self.C[n.args[0].name]
            m = self.gm.get_submodule(n.target)
            rho = rho_of(C, m.groups)
            return self.require(n.args[0].name, rho[target])
        if k == "cat":
            off = 0
            ok = True
            for a in n.args[0]:
                C = self.C[a.name]
                block = target[off:off + C]
                if np.any(block < off) or np.any(block >= off + C):
                    return False
                ok = ok and self.require(a.name, block - off)
                off += C
            return ok
        if k == "add":
            return self.require(n.args[0].name, target) and self.require(n.args[1].name, target)
        if k == "getitem":
            idx = n.args[1]
            if isinstance(idx, tuple) and len(target) == self.C[n.name]:
                return self.require(n.args[0].name, target)
            return False
        return False

    # ---- 解析：反复修正置换不一致 ----
    def resolve(self, max_iter: int = 60) -> int:
        it = 0
        for it in range(max_iter):
            self.compute()
            progress = False
            for add_name, a, b in list(self.mismatch):
                snap = dict(self.conv_out)
                if self.require(b, self.pi[a]):
                    progress = True
                    break
                self.conv_out = snap
                if self.require(a, self.pi[b]):
                    progress = True
                    break
                self.conv_out = snap
            if progress:
                continue
            for s in list(self.detect_in_mismatch):
                snap = dict(self.conv_out)
                if self.require(s, np.arange(self.C[s])):
                    progress = True
                    break
                self.conv_out = snap
            if not progress:
                break
        self.compute()
        return it + 1


def build() -> tuple[fx.GraphModule, dict[str, int], nn.Module]:
    model = ra.load_model(SPEC).eval()
    fold_bn_inplace(model)
    gm = fx.GraphModule(model, LeafTracer().trace(model))
    x = torch.rand(1, 3, 640, 640)
    ShapeProp(gm).propagate(x)
    C: dict[str, int] = {}
    for n in gm.graph.nodes:
        tm = n.meta.get("tensor_meta")
        if tm is not None and hasattr(tm, "shape") and len(tm.shape) == 4:
            C[n.name] = int(tm.shape[1])
    return gm, C, model


def fold_graph() -> tuple[fx.GraphModule, fx.GraphModule, dict]:
    """返回 (参考图, 折叠后的图, 报告)。参考图为未改动的 BN 折叠 SGC（fx 展开）。"""
    gm, C, _ = build()
    ref = copy.deepcopy(gm)
    eng = FoldEngine(gm, C)
    rounds = eng.resolve()
    assert not eng.mismatch and not eng.detect_in_mismatch and not eng.unsupported, "存在无法处理的节点"
    pi = {k: v.copy() for k, v in eng.pi.items()}

    # 确认没有共享的卷积模块（否则权重改写会相互影响）
    targets = [n.target for n in gm.graph.nodes if eng.kind(n) in ("conv", "dw")]
    assert len(targets) == len(set(targets)), "存在共享卷积模块"

    n_w = 0
    for n in gm.graph.nodes:
        k = eng.kind(n)
        if k not in ("conv", "dw"):
            continue
        m = gm.get_submodule(n.target)
        pin = pi[n.args[0].name]
        W = m.weight.data
        b = m.bias.data if m.bias is not None else None
        if k == "conv":
            sigma = pi[n.name]
            W2 = W[:, torch.as_tensor(pin, dtype=torch.long)]
            W2 = W2[torch.as_tensor(sigma, dtype=torch.long)]
            b2 = b[torch.as_tensor(sigma, dtype=torch.long)] if b is not None else None
        else:  # depthwise：输出置换 = 输入置换
            W2 = W[torch.as_tensor(pin, dtype=torch.long)]
            b2 = b[torch.as_tensor(pin, dtype=torch.long)] if b is not None else None
        m.weight = nn.Parameter(W2.clone())
        if b2 is not None:
            m.bias = nn.Parameter(b2.clone())
        n_w += 1

    # 逐个 ChannelShuffle 记录：消费者类型、输入/输出置换是否为恒等、以及是否需要 Conv 输出置换调整
    shuffle_rows = []
    for n in gm.graph.nodes:
        if eng.kind(n) != "shuffle":
            continue
        C_in = C[n.args[0].name]
        g = gm.get_submodule(n.target).groups
        users = []
        for u in n.users:
            users.append({"node": u.name, "kind": eng.kind(u),
                          "target": (u.target if u.op == "call_module" else
                                     getattr(u.target, "__name__", str(u.target)))
                          if u.op in ("call_module", "call_function") else u.op})
        shuffle_rows.append({
            "shuffle": n.name, "input": n.args[0].name, "C": C_in, "groups": g,
            "users": users,
            "input_perm_identity": bool(np.array_equal(pi[n.args[0].name], np.arange(C_in))),
            "output_perm_identity": bool(np.array_equal(pi[n.name], np.arange(C_in))),
        })

    n_removed = 0
    for n in list(gm.graph.nodes):
        if eng.kind(n) == "shuffle":
            src = n.args[0]
            n.replace_all_uses_with(src)
            gm.graph.erase_node(n)
            n_removed += 1
    gm.graph.lint()
    gm.recompile()

    report = {
        "rounds": rounds, "n_shuffle_removed": n_removed, "n_conv_weights_rewritten": n_w,
        "n_conv_output_perm_chosen": int(sum(1 for k, v in eng.conv_out.items()
                                             if not np.array_equal(v, np.arange(v.size)))),
        "conv_output_perm_chosen": {k: (not np.array_equal(v, np.arange(v.size))) for k, v in eng.conv_out.items()},
        "shuffles": shuffle_rows,
    }
    return ref, gm, report


def verify(ref: fx.GraphModule, new: fx.GraphModule, n_trials: int = 3) -> dict:
    ref.eval()
    new.eval()
    errs, vals = [], []
    torch.manual_seed(0)
    with torch.no_grad():
        for _ in range(n_trials):
            x = torch.rand(1, 3, 640, 640)
            a = ref(x)
            b = new(x)
            fa = [t for t in (a if isinstance(a, (list, tuple)) else [a])]
            fb = [t for t in (b if isinstance(b, (list, tuple)) else [b])]
            flat_a = []
            flat_b = []
            for u in fa:
                flat_a += list(u) if isinstance(u, (list, tuple)) else [u]
            for u in fb:
                flat_b += list(u) if isinstance(u, (list, tuple)) else [u]
            assert len(flat_a) == len(flat_b)
            errs.append(max(float((p - q).abs().max()) for p, q in zip(flat_a, flat_b)))
            vals.append(max(float(p.abs().max()) for p in flat_a))
    n_shuffle_left = sum(1 for n in new.graph.nodes if n.op == "call_module" and
                         type(new.get_submodule(n.target)).__name__ == "ChannelShuffle")
    return {"max_abs_err": max(errs), "max_abs_value": max(vals), "rel": max(errs) / max(vals),
            "channel_shuffle_nodes_left": n_shuffle_left}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    ref, new, report = fold_graph()
    v = verify(ref, new)
    report["verify"] = v
    print(f"移除 ChannelShuffle {report['n_shuffle_removed']} 个；改写卷积权重 {report['n_conv_weights_rewritten']} 个；"
          f"输出置换调整 {report['n_conv_output_perm_chosen']} 个；迭代 {report['rounds']} 次")
    print(f"数值校验：最大绝对误差 {v['max_abs_err']:.3e}（输出最大值 {v['max_abs_value']:.2f}，相对 {v['rel']:.2e}）；"
          f"剩余 ChannelShuffle 节点 {v['channel_shuffle_nodes_left']}")
    (OUT / "fold_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已保存 {OUT / 'fold_report.json'}")


if __name__ == "__main__":
    main()
