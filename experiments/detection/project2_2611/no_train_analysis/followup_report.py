"""生成《后续工作_2026-11-21.md》：表格由 JSON 结果自动生成，避免手工抄录错误。

输入：outputs/followup/{shuffle_nodes,conv_perm_sites,fx_summary}.json、outputs/followup/artifacts/manifest.json、
      outputs/sweep/*_trt.json、outputs/followup/artifacts/nsys/kernel_summary.json、inspector 层名 JSON
输出：no_train_analysis/后续工作_2026-11-21.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJ = _HERE.parent
_ROOT = _PROJ.parents[2]
for p in (_ROOT, _PROJ, _PROJ / "analysis_scripts", _HERE):
    sys.path.insert(0, str(p))

F = _HERE / "outputs" / "followup"
SW = _HERE / "outputs" / "sweep"
ART = F / "artifacts"
MD = _HERE / "后续工作_2026-11-21.md"


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def owner_types() -> dict[str, str]:
    """Conv 所在的 BaseConv 包装层的上级模块类型（SDGhost / VOVGCSPYolo 等）。"""
    from shuffle_fold import build
    _, _, model = build()
    out = {}
    for path in ["backbone.stage2.1", "backbone.stage3.1", "backbone.stage4.1", "backbone.stage5.1",
                 "neck.c3_p4", "neck.c3_p3", "neck.c3_n4", "neck.c3_n5"]:
        out[path] = type(model.get_submodule(path)).__name__
    return out


def consumer_text(users: list[dict]) -> str:
    parts = []
    for u in users:
        if u["kind"] == "conv":
            parts.append(f"`{u['node']}`（Conv2d {u['in_ch']}→{u['out_ch']}，k{u['k']}，g{u['groups']}）")
        elif u["kind"] == "add":
            parts.append(f"`{u['node']}`（残差 add）")
        elif u["kind"] == "cat":
            parts.append(f"`{u['node']}`（cat）")
        elif u["kind"] == "pass":
            parts.append(f"`{u['node']}`（{u.get('type', '逐通道算子')}）")
        else:
            parts.append(f"`{u['node']}`（{u['kind']}）")
    return "；".join(parts) if parts else "（无）"


def shuffle_table(rows: list[dict]) -> str:
    lines = ["| # | Shuffle FX 节点 | FX target | C / g | 输入节点（类型） | 消费者 | 折叠方式 |",
             "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        lines.append(f"| {i} | `{r['shuffle_node']}` | `{r['fx_target']}` | {r['C']} / {r['groups']} | "
                     f"`{r['input_node']}`（{r['input_kind']}） | {consumer_text(r['users'])} | {r['status']} |")
    return "\n".join(lines)


def conv_table(sites: list[dict], owners: dict[str, str]) -> str:
    lines = ["| # | Conv FX 节点 | Conv2d target | 形状 (in→out, k, g) | 所在 BaseConv（cv1） | 所属模块（类型） | 服务的残差 Add | 置换一致性 |",
             "|---|---|---|---|---|---|---|---|"]
    for i, s in enumerate(sites, 1):
        c = s["conv_shape"]
        owner = s["parent_module_target"].rsplit(".", 1)[0]
        lines.append(f"| {i} | `{s['conv_node']}` | `{s['conv_target']}` | {c['in']}→{c['out']}, k{c['k']}, g{c['groups']} | "
                     f"`{s['parent_module_target']}` | `{owner}`（{owners.get(owner, '?')}） | "
                     f"{', '.join('`' + a + '`' for a in s['served_adds'])} | "
                     f"{'全部一致' if all(s['perm_consistency_checks']) else '不一致'} |")
    return "\n".join(lines)


def add_table(adds: list[dict]) -> str:
    lines = ["| 残差 Add | 置换来源（Shuffle 操作数） | 残差链回溯到的 Conv | 置换是否一致 |",
             "|---|---|---|---|"]
    for a in adds:
        lines.append(f"| `{a['add']}` | `{a['shuffle_operand']}` | `{a['residual_source_conv']}` | "
                     f"{'是' if a['perm_equal_to_conv_output_perm'] else '否'} |")
    return "\n".join(lines)


def sweep_grid(rows: list[dict], key: str, fmt: str) -> str:
    sizes = sorted({r["size"] for r in rows})
    batches = sorted({r["batch"] for r in rows})
    lines = ["| 输入尺寸 \\ Batch | " + " | ".join(str(b) for b in batches) + " |",
             "|---|" + "---|" * len(batches)]
    for s in sizes:
        vals = []
        for b in batches:
            r = next(x for x in rows if x["size"] == s and x["batch"] == b)
            vals.append(fmt.format(r[key]))
        lines.append(f"| {s}×{s} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def speedup_grid(a: list[dict], b: list[dict]) -> str:
    sizes = sorted({r["size"] for r in a})
    batches = sorted({r["batch"] for r in a})
    lines = ["| 输入尺寸 \\ Batch | " + " | ".join(str(x) for x in batches) + " |", "|---|" + "---|" * len(batches)]
    for s in sizes:
        vals = []
        for bt in batches:
            ra_ = next(x for x in a if x["size"] == s and x["batch"] == bt)["sync_ms"]
            rb_ = next(x for x in b if x["size"] == s and x["batch"] == bt)["sync_ms"]
            vals.append(f"{(rb_ / ra_ - 1) * 100:+.0f}%")   # 相对变化：负值 = b 比 a 快
        lines.append(f"| {s}×{s} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def kernel_table(ks: dict) -> str:
    names = {"original": "原始 SGC FP16（非 DETAILED）", "noshuffle_upper_bound": "无 Shuffle 上限（非等价）",
             "shuffle_folded": "置换折叠 FP16（非 DETAILED）", "shuffle_folded_detailed": "置换折叠 FP16（DETAILED）",
             "lean_detailed": "精简 FP16（out0 + float 网格，DETAILED）"}
    lines = ["| 引擎 | kernel / 次推理 | kernel 耗时 / 次推理 | Reformat kernel / 次 | Reformat 耗时 | 布局 kernel / 次 |",
             "|---|---|---|---|---|---|"]
    for k in ["original", "noshuffle_upper_bound", "shuffle_folded", "shuffle_folded_detailed", "lean_detailed"]:
        v = ks[k]
        lines.append(f"| {names[k]} | {v['kernels_per_inference']:.0f} | {v['kernel_time_ms_per_inference']:.3f} ms | "
                     f"{v['reformat_kernels_per_inference']:.0f} | {v['reformat_ms_per_inference']:.3f} ms | "
                     f"{v['layout_kernels_per_inference']:.0f} |")
    return "\n".join(lines)


def artifact_table(files: dict) -> str:
    lines = ["| 存档文件（outputs/followup/artifacts/） | sha256（前 12 位） | 大小 |", "|---|---|---|"]
    for k, v in files.items():
        if "sha256" in v:
            lines.append(f"| `{k}` | `{v['sha256'][:12]}` | {v['bytes'] / 1e6:.2f} MB |")
    return "\n".join(lines)


def main() -> None:
    shuffles = load(F / "shuffle_nodes.json")
    cp = load(F / "conv_perm_sites.json")
    fxs = load(F / "fx_summary.json")
    man = load(ART / "manifest.json")
    ks = load(ART / "nsys" / "kernel_summary.json")
    owners = owner_types()
    sw = {n: load(SW / f"{n}_trt.json")["rows"] for n in ["sgc_original", "sgc_perm", "yolov5s"]}

    # 延迟扫描：置换折叠 vs 原始（同为 BN 折叠的 TRT 动态引擎）
    speed_perm = speedup_grid(sw["sgc_original"], sw["sgc_perm"])
    speed_yolo = speedup_grid(sw["sgc_perm"], sw["yolov5s"])

    def row640(name, rows):
        r = next(x for x in rows if x["size"] == 640 and x["batch"] == 1)
        return f"| {name} | {r['sync_ms']:.2f} | {r['pipe_ms']:.2f} |"

    table640 = "\n".join([
        "| 变体（640×640，batch 1） | 逐次同步 (ms) | 流水线 (ms) |", "|---|---|---|",
        row640("SGC 原始（BN 折叠，未置换）", sw["sgc_original"]),
        row640("SGC 置换折叠（等价）", sw["sgc_perm"]),
        row640("YOLOv5s（对照）", sw["yolov5s"]),
    ])

    md = f"""# 后续工作（2026-11-21）

接续 `shuffle_fold_report_2026-11-21.md`（ChannelShuffle 置换折叠）与 `reformat_remaining_report_2026-11-21.md`（剩余 Reformat 分析）。
本文件的表格由 `followup_report.py` 从 JSON 结果自动生成，原始数据均在 `outputs/followup/` 与 `outputs/sweep/` 中。

## 0. 完成情况

| 任务 | 状态 | 结果位置 |
|---|---|---|
| 1. 补全 30 个 ChannelShuffle 的 FX 节点及消费者明细 | 完成 | §1，`outputs/followup/shuffle_nodes.json` |
| 2. 补全 8 个 Conv 输出置换调整的模块位置 | 完成（并核对 11 个残差 Add 的置换一致性） | §2，`outputs/followup/conv_perm_sites.json` |
| 3. 保存 FX 原始图、改写图和 ONNX 图 | 完成（附 sha256 清单） | §3，`outputs/followup/artifacts/` |
| 4. 不同输入尺寸与 Batch Size 下的延迟测试 | 完成（TRT FP16 动态引擎，3 个变体） | §4，`outputs/sweep/` |
| 5. 保存 TensorRT Engine Inspector 与 Nsight Systems 结果 | 完成 | §5，`outputs/followup/artifacts/inspector/`、`nsys/` |

## 1. 30 个 ChannelShuffle 的 FX 节点与消费者明细

图由 `torch.fx` 追踪 BN 折叠后的 SGC 得到（Detect 为叶子节点）。`FX target` 为模块路径；`C / g` 为通道数与 `ChannelShuffle.groups`。
完整的节点列表（名称、op、target、args、users）见 `outputs/followup/fx/original_nodes.json`。

按消费者类型统计：仅 Conv 消费者 15 个，残差 Add 11 个，Concat 4 个（其中 2 个经 Upsample）。

{shuffle_table(shuffles)}

## 2. 8 个 Conv 输出置换调整的模块位置

这 8 个 Conv 是 C3 类模块的 `cv1` 卷积。它们的输出顺序是自由选择的（Conv 可以任意重排输出通道），因此被设为对应残差链所需的置换。
它们分别位于以下模块中：

{conv_table(cp['sites'], owners)}

注：`所属模块` 中的类型来自折叠前模型的实际实例：`backbone.stage*.1` 为 SDGhost，`neck.c3_*` 为 VOVGCSPYolo。
`所在 BaseConv（cv1）` 是模块内的 `cv1` 包装层（BaseConv），其内部 `block.0` 即上表中的 Conv2d。

### 2.1 残差 Add 与置换来源的对应关系

每个残差 Add 的一个操作数来自 Shuffle（置换来源），另一个操作数沿残差链回溯到上表中的某个 Conv。
对全部 11 个 Add，回溯得到的 Conv 输出置换与 Shuffle 的输出置换逐元素相等（`perm_equal_to_conv_output_perm = 是`）。

{add_table(cp['add_mapping'])}

## 3. 可复现的优化记录（FX 图、ONNX、引擎）

- FX 原始图（折叠前）：`outputs/followup/fx/original_graph.txt`（含 generated code）、`original_nodes.json`；改写图（置换折叠后）：`fx/rewritten_graph.txt`、`rewritten_nodes.json`。
- 改写统计（`fx_summary.json`）：原始 {fxs['original_nodes']} 个节点，改写后 {fxs['rewritten_nodes']} 个；移除 ChannelShuffle {fxs['shuffle_removed']} 个，改写后图中剩余 {fxs['shuffle_left_in_rewritten_graph']} 个；改写卷积权重 {fxs['conv_weights_rewritten']} 个；输出置换调整 {fxs['conv_output_perm_sites']} 个。
- ONNX、引擎与 Inspector、nsys 文件均存放在 `outputs/followup/artifacts/`，下表为 sha256 与大小：

{artifact_table(man['files'])}

- 检查点（用于复现 FX 图与 ONNX）：
  - SGC 无注意力 `best.pth`，sha256 `{man['checkpoints']['SGC 无注意力 best.pth']['sha256'][:12]}…`
  - YOLOv5s `best.pth`，sha256 `{man['checkpoints']['YOLOv5s best.pth']['sha256'][:12]}…`

复现步骤（仓库根目录）：

```
python experiments/detection/project2_2611/no_train_analysis/followup_fx_records.py   # FX 明细、Conv 置换位置、FX 图
python experiments/detection/project2_2611/no_train_analysis/shuffle_fold_bench.py    # 置换折叠 ONNX 与 TRT FP16 引擎
python experiments/detection/project2_2611/no_train_analysis/followup_collect.py      # 存档、Inspector 与 nsys 汇总、sha256 清单
```

## 4. 不同输入尺寸与 Batch Size 下的延迟

测试方法：动态形状 ONNX（batch、H、W 可变），TRT FP16 引擎，optimization profile 为 min 1×320²、opt 4×640²、max 8×800²。
每个配置：逐次同步延迟取 50 次中位数；流水线吞吐取 50 次连续推理的平均。输入为随机张量（只测速度，不测精度）。
三个变体：`sgc_original`（BN 折叠，未置换）、`sgc_perm`（置换折叠，与前一节等价）、`yolov5s`（BN 折叠，对照）。

### 4.1 逐次同步延迟（ms）

SGC 原始（BN 折叠，未置换）：

{sweep_grid(sw['sgc_original'], 'sync_ms', '{:.2f}')}

SGC 置换折叠（等价）：

{sweep_grid(sw['sgc_perm'], 'sync_ms', '{:.2f}')}

YOLOv5s（对照）：

{sweep_grid(sw['yolov5s'], 'sync_ms', '{:.2f}')}

### 4.2 流水线吞吐（img/s）

置换折叠 SGC：

{sweep_grid(sw['sgc_perm'], 'pipe_img_per_s', '{:.0f}')}

YOLOv5s：

{sweep_grid(sw['yolov5s'], 'pipe_img_per_s', '{:.0f}')}

### 4.3 置换折叠相对原始的逐次同步延迟变化（负值 = 更快）

{speed_perm}

### 4.4 YOLOv5s 相对 SGC 置换折叠的逐次同步延迟变化（负值 = YOLOv5s 更快）

{speed_yolo}

要点与注意事项：

- 640×640、batch 1 下的对比：

{table640}
- 小 batch 时延迟由 kernel 发射与空隙主导，几乎不随输入尺寸增长（SGC 置换折叠 320→800 时 batch 1 仅从 2.22 增至 2.73 ms）；batch 增大后延迟近似线性增长。
- YOLOv5s 动态引擎构建时出现 `ptxas fatal: Memory allocation failure`，部分 tactic 被跳过（TRT 日志中可见），因此 YOLOv5s 的数值可能略偏保守；它与静态引擎（640、batch 1 约 1.9–2.3 ms）的差异需要在静态重建后确认。
- 三个动态引擎均以 opt 4×640² 优化，640 与 800 等尺寸的结果可能偏向 opt 形状；这是 TRT 动态形状的通用限制。
- 每个配置只测 1 轮（50 次同步 + 50 次流水线），单点误差约 ±5%，判断应以趋势为准。
- 本节只测速度，没有对不同尺寸与 batch 做精度验证；非 640 的输入尺寸未测量 mAP。

## 5. TensorRT Engine Inspector 与 Nsight Systems 结果

Kernel 汇总（nsys，每个引擎 100 次推理的平均，来自 `artifacts/nsys/kernel_summary.json`）：

{kernel_table(ks)}

定义：Reformat kernel 为 kernel 名含 `reformat` / `copyPacked` / `copyVectorized` 的算子；布局 kernel 为名含 `nchwTo*` / `nhwcTo*` 的算子。两者按名称关键词统计，与前几份报告中按类别统计的口径可能略有差异（例如布局 kernel 数）。

Inspector（逐层信息）：

- `inspector/sgc_shuffle_folded_detailed_layers.json`：DETAILED 置换折叠引擎的逐层信息，包括每层输入输出的格式/精度与 ONNX 来源（`Metadata`），是 `reformat_remaining_report_2026-11-21.md` 的数据来源。
- `inspector/sgc_lean_detailed_layers.json`：精简引擎（只输出 out0、float 网格）的同类信息。
- `inspector/*_layer_names.json`：非 DETAILED 引擎的层名列表（原始 325 层、置换折叠 249 层、无 Shuffle 上限 227 层）。

Nsight Systems 报告（`nsys/*.nsys-rep`）与 kernel 汇总 CSV（`nsys/*_kern_cuda_gpu_kern_sum.csv`）已存档，可直接用 Nsight Systems GUI 打开。

## 6. 未完成与下一步

1. PyTorch 原生（eager / CUDA Graph）的输入尺寸与 batch 扫描尚未做，本节只有 TRT 结果。
2. YOLOv5s 动态引擎需要在无内存压力的条件下重建（或改用静态引擎逐个形状测量），以排除 tactic 跳过的影响。
3. 不同输入尺寸与 batch 下的精度（mAP）尚未验证。
4. Reformat 与 Shuffle 相关的优化已经在 `shuffle_fold_report` 与 `reformat_remaining_report` 中讨论，本文件不再重复。
5. 置换折叠的规则目前只在 SGC 上验证，如果要推广到其它结构（例如 DFC 注意力分支 `x1 * gate`），需要在 `FoldEngine` 中补充对 `mul` 节点的处理规则。
"""
    MD.write_text(md, encoding="utf-8")
    print(f"已写入 {MD}")


if __name__ == "__main__":
    main()
