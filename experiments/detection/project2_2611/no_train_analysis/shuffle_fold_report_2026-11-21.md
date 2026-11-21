# ChannelShuffle 逐节点可折叠性分析与置换折叠（图优化）

日期：2026-11-21
对象：SGC 无注意力（`sgc_yolov5s_c3_sgdown_gbot_dw5`，run `2026-11-17_015_…_seed42_V1`，BN 已折叠）
工具：`torch.fx` 计算图（Detect 作为叶子）、`shuffle_fold.py`（分析与改写）、`shuffle_fold_bench.py`（ONNX/TRT）、`shuffle_fold_map_eval.py`（精度）

---

## 0. 结论

1. **30 个 ChannelShuffle 全部被等价消除**，图中不再有任何 ChannelShuffle 节点，**没有插入额外的 Gather**。
2. **数学功能不变**：PyTorch FP32 下，折叠图与折叠前的图最大绝对误差为 1.2e-4（相对 1.9e-7，即 float32 舍入）；测试集 1501 张上 mAP50 与 mAP50-95 与参考**逐位一致**，检测框数 15966 完全相同。
3. **ONNX 算子**：Reshape 69 → 9，Transpose 33 → 3（与"无 Shuffle 上限"的 ONNX 算子计数逐项相同）。
4. **TRT FP16 延迟**：同一会话内逐次同步 3.19 → 2.22 ms（−30%），流水线 3.22 → 2.29 ms（−29%）。与"无 Shuffle 上限"（2.38 ms，非等价）在测量噪声内相当。
5. **kernel**：每次推理 kernel 数 262 → 200（−24%），kernel 耗时 2.095 → 1.587 ms（−24%），布局转换 kernel 29 → 4（0.171 → 0.017 ms）。
6. **精度**：TRT FP16 折叠版相对原始 FP16 的 ΔmAP50 为 −0.0005（95% CI [−0.0022, +0.0008]），ΔmAP50-95 为 +0.0001（CI [−0.0007, +0.0010]），均不显著；TP@0.5 计数相同（8177）。

---

## 1. 逐节点消费者分析（30 个 ChannelShuffle）

约定：Shuffle 的物理重排在折叠后被移除。每个张量的"存储顺序"用置换 π 描述，ChannelShuffle(g) 把输出置换变为 ρ⁻¹∘π_in（ρ 与 `F.channel_shuffle` 一致）。消费者按下列规则吸收置换：

| 消费者 | 处理方式 |
|---|---|
| Conv2d（groups=1） | 输入置换吸收进权重 `W[:, π]`，**零成本** |
| Depthwise Conv | 输出置换 = 输入置换，权重沿通道维置换，**零成本** |
| SiLU / MaxPool / Upsample | 逐通道/空间算子，置换原样保留 |
| Concat | 输出置换 = 各输入置换拼接（加偏移），由下游 Conv 吸收 |
| Add（残差） | 两操作数置换必须相同；不同时，把需求反向传播到可自由选择输出顺序的 Conv（见 §2） |

### 1.1 消费者类型统计

| 消费者类型 | Shuffle 数量 | 折叠方式 |
|---|---|---|
| 仅 Conv（单一或两个 Conv） | 15 | 直接吸收进消费者 Conv 的输入权重（"单 Conv 消费者"） |
| 残差 Add | 11 | 把 Add 另一操作数的置换需求反向传播到 C3 的 `cv1` 卷积输出（8 个 Conv），再沿残差链（DWConv 与 Bottleneck）传播，无额外算子 |
| Concat（含经 Upsample 的分支） | 4 | 置换并入 Concat 的通道块，由下游 `cv1/cv2` 卷积吸收 |

### 1.2 逐节点明细

| Shuffle 节点 | C / g | 消费者 | 折叠方式 |
|---|---|---|---|
| backbone.stage2.0 | 64 / 16 | stage2.1.cv1, cv2（Conv） | 吸收进两个 Conv 的输入权重 |
| stage2.1.bottleneck.0.SGhostConv1 | 32 / 8 | SGhostConv2.conv（Conv） | 吸收进 Conv |
| stage2.1.bottleneck.0.SGhostConv2 | 32 / 8 | add（残差） | 残差链 + `cv1` 输出置换 |
| stage3.0 | 128 / 32 | stage3.1.cv1, cv2（Conv） | 吸收 |
| stage3.1.bottleneck.0.SGhostConv1 | 64 / 16 | SGhostConv2.conv | 吸收 |
| stage3.1.bottleneck.0.SGhostConv2 | 64 / 16 | add_1 | 残差链 |
| stage3.1.bottleneck.1.SGhostConv1 | 64 / 16 | SGhostConv2.conv | 吸收 |
| stage3.1.bottleneck.1.SGhostConv2 | 64 / 16 | add_2 | 残差链 |
| stage4.0 | 256 / 64 | stage4.1.cv1, cv2（Conv） | 吸收 |
| stage4.1.bottleneck.0.SGhostConv1 | 128 / 32 | SGhostConv2.conv | 吸收 |
| stage4.1.bottleneck.0.SGhostConv2 | 128 / 32 | add_3 | 残差链 |
| stage4.1.bottleneck.1.SGhostConv1 | 128 / 32 | SGhostConv2.conv | 吸收 |
| stage4.1.bottleneck.1.SGhostConv2 | 128 / 32 | add_4 | 残差链 |
| stage4.1.bottleneck.2.SGhostConv1 | 128 / 32 | SGhostConv2.conv | 吸收 |
| stage4.1.bottleneck.2.SGhostConv2 | 128 / 32 | add_5 | 残差链 |
| stage5.0 | 512 / 128 | stage5.1.cv1, cv2（Conv） | 吸收 |
| stage5.1.bottleneck.0.SGhostConv1 | 256 / 64 | SGhostConv2.conv | 吸收 |
| stage5.1.bottleneck.0.SGhostConv2 | 256 / 64 | add_6 | 残差链 |
| neck.reduce5 | 256 / 32 | Upsample（pass）、cat_39 | 经 Upsample 与 Concat 传播，由下游 Conv 吸收 |
| neck.c3_p4.m.0.GSConv1 | 128 / 16 | GSConv2.conv（Conv） | 吸收 |
| neck.c3_p4.m.0.GSConv2 | 128 / 16 | add_7 | 残差链 |
| neck.reduce4 | 128 / 16 | Upsample（pass）、cat_34 | 同 reduce5 |
| neck.c3_p3.m.0.GSConv1 | 64 / 8 | GSConv2.conv | 吸收 |
| neck.c3_p3.m.0.GSConv2 | 64 / 8 | add_8 | 残差链 |
| neck.down3 | 128 / 16 | cat_34 | Concat 传播 |
| neck.c3_n4.m.0.GSConv1 | 128 / 16 | GSConv2.conv | 吸收 |
| neck.c3_n4.m.0.GSConv2 | 128 / 16 | add_9 | 残差链 |
| neck.down4 | 256 / 32 | cat_39 | Concat 传播 |
| neck.c3_n5.m.0.GSConv1 | 256 / 32 | GSConv2.conv | 吸收 |
| neck.c3_n5.m.0.GSConv2 | 256 / 32 | add_10 | 残差链 |

### 1.3 残差 Add 为什么不需要额外算子

以 `stage4.1` 为例：C3 的 `cv1` 输出 x（Bottleneck 输入）先被置换为 π*，即 SGhostConv2 的 Shuffle 输出置换。此时：

- 残差支路 `DWConv2(x)` 是深度卷积，输出置换 = 输入置换 = π*；
- 主支路 `SGhostConv2` 的 Shuffle 输出置换也是 π*；
- 两者相同，Add 不需要任何重排。

Bottleneck 的输出仍然是置换 π*，因此下一个 Bottleneck 的输入也是 π*，**整条残差链都保持同一个置换**，直到 C3 的 `cv3`（Conv）吸收它。`cv1` 是唯一的"自由"节点（Conv 可以任意选择输出顺序）。共有 8 个 `cv1`（backbone stage2–5 与 neck c3_p4/p3/n4/n5）的输出被选择为对应的 π*，这就是报告中的 8 个"输出置换调整"。

迭代过程：传播收敛用了 9 轮，之后没有残余的 Add 不一致、没有 Detect 输入不一致、没有不支持的节点。

---

## 2. 图改写结果

| 项目 | 数值 |
|---|---|
| 移除的 ChannelShuffle | 30 |
| 改写权重的卷积 | 94（Conv 输入维置换 + 输出维置换 / DW 通道置换） |
| 调整输出顺序的 Conv（自由选择） | 8（C3 的 cv1） |
| 插入的 Gather（物化代价） | 0 |
| 数值校验（PyTorch FP32，随机输入 ×3） | 最大绝对误差 1.2e-4，相对 1.9e-7 |

改写在 `fx.GraphModule` 上完成，只改动权重与图结构，**不修改 `sgc_yolov5.py` 源码**。折叠版由 `shuffle_fold.fold_graph()` 生成，可以直接导出 ONNX。

---

## 3. ONNX 与 TRT

### 3.1 ONNX 算子（opset 17，静态 batch 1）

| 算子 | 原始 | 无 Shuffle 上限（非等价） | 置换折叠（等价） |
|---|---|---|---|
| Reshape | 69 | 9 | **9** |
| Transpose | 33 | 3 | **3** |
| Concat | 50 | 48 | 48 |
| Constant | 206 | 146 | 146 |
| Gather | 0 | 0 | 0 |

置换折叠版与"无 Shuffle 上限"的 ONNX 算子计数**逐项相同**，说明改写之后的图结构与去掉 Shuffle 的图一致，只是权重的排列不同。

### 3.2 TRT FP16 引擎结构与延迟（同一会话，3 次交替测量）

| 引擎 | 层数 | 名称含 shuffle 的层 | 名称含 Reformatting 的层 | 逐次同步（中位数） | 流水线（中位数） |
|---|---|---|---|---|---|
| 原始 SGC FP16 | 325 | 82 | 65 | **3.19 ms**（3.15–3.22） | 3.22 ms |
| 无 Shuffle 上限（非等价） | 227 | 0 | 29 | 2.38 ms（2.24–2.46） | 2.43 ms |
| **置换折叠 FP16（等价）** | **249** | **0** | **51** | **2.22 ms**（2.17–2.31） | **2.29 ms** |

- 置换折叠版的延迟比原始降低 **30%**（逐次同步），与"无 Shuffle 上限"在噪声内相当。
- 层数 249 与上限的 227 不同：两个引擎的图结构相同，但 TRT 的 tactic 选择与 Reformatting 插入存在构建期的 autotuning 差异，因此层数与 reformat 数量并不完全相同，延迟仍在同一区间。
- IProfiler 的逐层耗时：Reformatting 0.650 → 0.166 ms（同步开销使绝对值偏高，仅作相对参考）。

### 3.3 kernel（nsys，100 次推理，同一会话）

| 引擎 | kernel 数 / 次 | kernel 耗时 / 次 | 布局转换 kernel（nchwTonhwc 等）/ 次 | genericReformat / 次 |
|---|---|---|---|---|
| 原始 SGC FP16 | 262 | 2.095 ms | 29（0.171 ms） | 51（0.450 ms） |
| 无 Shuffle 上限 | 205 | 1.596 ms | 0 | 25（0.108 ms） |
| **置换折叠 FP16** | **200** | **1.587 ms** | **4（0.017 ms）** | **20（0.086 ms）** |

卷积 kernel 数基本不变（95 → 93），卷积耗时基本不变（1.001 → 1.046 ms）。减少的主要是布局转换与 Reformatting kernel。

---

## 4. 精度验证

测试集 1501 张，同一预处理（letterbox 640，batch 1），同一 NMS（conf 0.001，IoU 0.6，max_det 300）。ΔmAP 为相对 BN 折叠参考（PyTorch）的差值；括号为配对 bootstrap（B=300）的 95% 置信区间。

| 变体 | mAP50 | mAP50-95 | 检测框数 | ΔmAP50 [CI] | ΔmAP50-95 [CI] |
|---|---|---|---|---|---|
| PyTorch BN 折叠（参考） | 0.7967 | 0.4649 | 15966 | — | — |
| **PyTorch 置换折叠 FP32** | **0.7967** | **0.4649** | **15966** | 0.00000 [−5e-8, 0] | 0.00000 [−1e-8, 0] |
| TRT FP16 原始 | 0.7963 | 0.4652 | 15930 | −0.0004 [−0.0027, +0.0014] | +0.0003 [−0.0007, +0.0011] |
| TRT FP16 置换折叠 | 0.7957 | 0.4652 | 15914 | −0.0010 [−0.0029, +0.0001] | +0.0003 [−0.0004, +0.0011] |

- **PyTorch FP32 折叠版与参考完全一致**，证明改写是严格等价的。
- TRT FP16 折叠版相对原始 FP16：ΔmAP50 −0.0005（CI [−0.0022, +0.0008]），ΔmAP50-95 +0.0001（CI [−0.0007, +0.0010]），**均不显著**，TP@0.5 计数相同（8177）。
- FP16 引擎之间的微小差异来自权重置换改变了累加顺序，因此 FP16 舍入不同。它不是结构变化，但也不是比特级一致。

---

## 5. 局限与注意事项

1. **只对 SGC 做了改写与验证**。YOLOv5s 没有 ChannelShuffle，不适用。
2. **改写在 fx 图上完成**，源码 `sgc_yolov5.py` 未修改。要长期使用，需要在部署流程中调用 `shuffle_fold.fold_graph()`，或者把折叠逻辑固化为一个导出脚本。训练时仍然使用含 Shuffle 的原始结构，权重需要重新折叠后再部署。
3. **TRT 构建存在 autotuning 随机性**：同一 ONNX 两次构建的层数与 reformat 数量可能不同。延迟比较均在同一会话内交替测量，GPU 状态漂移（约 15%）已经通过同会话比较消除。
4. **剩余的 20 个 genericReformat kernel** 来自网络中其余的布局需求（例如网络输入与输出的格式转换），未在本报告中逐一定位。
5. 对 FP16 的数值差异只做了统计检验，没有逐检测框比较。
6. 这次改写不改变模型的数学功能，因此不存在"去掉 Shuffle 后精度未知"的问题，但仍然建议在部署前用同一测试集复核 mAP（本报告已做）。

---

## 6. 文件与复现

脚本（`no_train_analysis/`）：

- `shuffle_graph_analysis.py`：fx 追踪与消费者列表（第一步）
- `shuffle_fold.py`：置换传播、解析与图改写（`fold_graph()`、`verify()`）
- `shuffle_fold_bench.py`：ONNX 导出、TRT 构建与延迟对比
- `shuffle_fold_map_eval.py`：测试集 mAP 与配对 bootstrap

结果（`outputs/shuffle_fold/`）：

- `graph_analysis.json`：30 个 Shuffle 的消费者列表（分析阶段）
- `fold_report.json`：改写报告（含每个 Shuffle 的消费者与状态）
- `sgc_fold.onnx`、`sgc_fold_fp16.engine`：折叠版模型与 FP16 引擎
- `fold_bench_2026-11-21.json`：ONNX 算子计数、TRT 结构与延迟
- `nsys/`：三个引擎的 nsys 报告与 `kernel_summary.json`
- `fold_map_2026-11-21.json`、`stats_fold.pkl`：mAP 与逐图统计量
