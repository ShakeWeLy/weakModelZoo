# 剩余 Reformat kernel 分析（置换折叠 SGC，TRT FP16）

日期：2026-11-21
对象：置换折叠后的 SGC FP16 引擎（`shuffle_fold_report_2026-11-21.md` 中的 `sgc_fold_fp16`）
工具：TensorRT DETAILED profiling verbosity（inspector 的逐层输入/输出格式与 ONNX 来源 `Metadata`）、IProfiler（逐层耗时，偏高）、Nsight Systems（kernel 耗时，用于判断真实 GPU 代价）
脚本：`reformat_analysis.py`、`reformat_classify.py`、`reformat_poc.py`、`reformat_poc_bench.py`

---

## 0. 结论

1. **剩余的 Reformat kernel 不是瓶颈。** 在 nsys 中，DETAILED 构建的折叠引擎有 19 个 Reformat kernel，合计 0.075 ms，占 kernel 总耗时（1.596 ms）的 4.7%，约占墙钟延迟（2.12 ms）的 3.5%。即使全部消除，上限也只有约 0.08 ms。
2. **来源（IProfiler 占比）**：Concat 输入拷贝约 51%（其中 6 层来自 Detect 头）；网络输入与输出转换约 34%（输入 18%、输出 16%）；Detect 头格式/精度转换约 15%。
3. **两个假设被概念验证否定**：
   - "ONNX 中的 Cast 导致解码路径为 FP32"：去掉 Cast 后，FP32 层只从 33 降到 25，Reformat kernel 反而从 19 增到 25，延迟无改善（2.12 → 2.20 ms，在噪声内）。
   - "原始特征图输出导致额外转换"：只导出 out0 后，Reformat kernel 没有减少；它们的转换与 Detect 的其它张量合并在一起。
4. **真正的主要开销**在卷积（kernel 占 67%，约 1.08 ms）和逐元素融合层（占 25%，约 0.41 ms），以及 kernel 之间的调度空隙（墙钟 2.12 ms − kernel 合计 1.60 ms ≈ 0.5 ms）。剩余 Reformat 不应该作为优化的下一步。

---

## 1. 方法

### 1.1 DETAILED 构建的逐层结构

用 `ProfilingVerbosity.DETAILED` 重建折叠引擎（228 层，与非 DETAILED 构建的 249 层不同，这是 TRT 构建时 tactic 选择的随机差异）。inspector 给出每层的输入/输出张量名、格式/精度（如 `Half`、`Float`），以及 `Metadata` 中的 ONNX 来源节点，可以把 Reformat 层追溯到具体的 PyTorch 模块。

层类型统计：`CaskConvolution` 83、`PointWiseV2` 67、`NoOp` 28、`Reformat` 21、`Slice` 9、`CaskGemmConvolution` 6、`kgen` 3、`CaskPooling` 3、`Shuffle` 3、`Constant` 3、`Resize` 2。

### 1.2 分类规则

- 网络输入转换：层名为 Input Tensor 且输入为 `images`（FP32 NCHW → FP16 NHWC）；
- 网络输出转换：输出绑定为 `out*` 或 `onnx::Sigmoid_*`；
- Concat 输入拷贝：名称以 `copy` 结尾，且 ONNX 来源为 Concat；
- Detect 头：张量或 ONNX 来源涉及 `/detect/`；
- 内部布局转换：卷积或逐元素融合所需的格式转换（耗时为 0，不产生 kernel）。

### 1.3 耗时测量

- IProfiler：逐层耗时，同步开销使绝对值偏高，仅用于相对比较；
- Nsight Systems：CUDA kernel 的 GPU 耗时（100 次推理的平均），用于判断真实代价。

---

## 2. 逐层分类（IProfiler，DETAILED 折叠引擎）

| 类别 | 层数 | 有耗时层 | 耗时（ms，IProfiler） | 占比 |
|---|---|---|---|---|
| Concat 输入拷贝 | 13 | 13 | 0.209 | 51% |
| 网络输入转换（images） | 1 | 1 | 0.073 | 18% |
| 网络输出转换（out0 / 原始特征图） | 4 | 4 | 0.065 | 16% |
| Detect 头格式/精度转换 | 3 | 3 | 0.063 | 15% |
| 内部布局转换（卷积 / 逐元素） | 25 | 0 | 0.000 | 0% |
| 合计 | 46 | 21 | 0.409 | — |

其中 Concat 拷贝里有 6 个属于 Detect 头（例如 `/detect/Mul_3_output_0 copy → /detect/Concat_1`）。若把 Detect 相关的层全部汇总（Detect 格式转换 + Detect 的 Concat 拷贝），共 13 层，约 0.213 ms，占 52%。

### 2.1 主要张量与来源

- **Concat 输入拷贝（FP16）**：`/up/Resize_output_0`、`/up_1/Resize_output_0`（上采样输出拷贝进 Concat 输出缓冲），`/block.1_21/Mul_output_0`、`/m/MaxPool_output_0` 等（Concat_22/24/29 等）。TRT 用拷贝实现 Concat，当输入无法直接写入输出的对应通道块时就会产生。
- **Detect 头（FP32）**：`/detect/Mul_1…11_output_0 copy`（FP32）→ `/detect/Concat_1/3/5`，以及 `/detect/Reshape_k_copy_output` → `out0`（FP32）。
- **网络输入**：`images [Float] → Reformatted Input Tensor 0 to /block.0/Conv [Half]`，即 FP32 NCHW 到 FP16 NHWC。
- **网络输出**：`out0`、`onnx::Sigmoid_971` 等输出绑定需要从 FP16 NHWC 转换为 FP32。

---

## 3. GPU 耗时（Nsight Systems，100 次推理平均）

| 引擎 | kernel 总数 / 次 | kernel 总耗时 / 次 | Reformat kernel / 次 | Reformat 耗时 / 次 | 布局 kernel（nchwTonhwc 等）/ 次 |
|---|---|---|---|---|---|
| 置换折叠（非 DETAILED，上一轮） | 200 | 1.587 ms | 20 | 0.086 ms | 4（0.017 ms） |
| 置换折叠（DETAILED） | 197 | 1.596 ms | 19 | 0.075 ms（4.7%） | 3（0.007 ms） |
| 精简（out0 + float 网格，DETAILED） | 208 | 1.571 ms | 25 | 0.067 ms（4.3%） | 3（0.013 ms） |

Reformat kernel 按模板签名的分布（DETAILED 折叠引擎）：

| Reformat kernel 模板 | 次数 / 次推理 | 耗时 / 次推理 | 对应来源（推断） |
|---|---|---|---|
| `copyPacked<__half,__half>` | 6 | 0.042 ms | FP16 Concat 输入拷贝、NHWC 内部布局 |
| `copyVectorized<double,float,float>` | 8 | 0.017 ms | FP32 张量拷贝（Detect 头、输出） |
| `copyPacked<float,float>` | 2 | 0.008 ms | FP32 Concat 拷贝（Detect） |
| 其它 `copyPacked/Vectorized`（half↔float 混合） | 3 | 0.008 ms | 混合精度的拷贝（Detect 与输出） |
| 合计 Reformat kernel | 19 | 0.075 ms | — |

另有 3 个布局 kernel（`nhwcTonchw` / `nchwTonchw`，0.007 ms），属于输出格式转换，不计入上表。

Concat 拷贝约占 Reformat 耗时的一半，与 IProfiler 的分类一致。

---

## 4. 概念验证：两个假设及其结果

### 4.1 假设 A：ONNX 中的 Cast 导致解码路径为 FP32

`Detect._grid` 中的 `.float()` 在 ONNX 中生成了 3 个 `Cast → FLOAT` 节点，它们可能迫使 TRT 把解码（sigmoid、xy/wh 计算）设为 FP32。

验证：用 float32 直接生成网格（数值逐元素相同，PyTorch 中误差 0），导出的 ONNX 中 Cast 从 3 个降为 0。

结果：
- 全 FP32 输出的层从 40 降到 31，其中 Detect 相关从 33 降到 25。
- 剩余的 FP32 层主要是**形状相关的 Constant / Expand / Unsqueeze / Slice**，以及解码算术中使用的 **FP32 常量**（anchor、stride、网格的 buffer 通过 Constant 节点进入引擎）。
- Reformat kernel 从 19 增到 25，耗时 0.075 → 0.067 ms，延迟 2.12 → 2.20 ms（噪声内，无收益）。

结论：假设 A 只解释了一小部分，主因是 **FP32 常量与形状计算**，而不是 Cast 本身。要让解码真正走 FP16，需要把 anchor、stride 与网格也改为 FP16，这会改变坐标精度（与 `cuda层分析.md` 中 YOLOv5s 的 FP16 定位误差问题相关），风险不小，且收益上限只有约 0.08 ms。

### 4.2 假设 B：原始特征图输出导致额外的格式转换

导出时去掉 `out1/out2/out3`（训练用的原始特征图），只保留 `out0`。

结果：输出转换层从 4 个降到 3，但 Reformat kernel 没有减少（25 个），延迟无改善。原始特征图的转换与 Detect 其它张量的转换合在一起，删除它们并不会移除对应的 kernel。

结论：假设 B 不成立，但只导出 `out0` 仍然是部署上更干净的做法（数值完全不变，PyTorch 中误差为 0），只是不能带来速度收益。

---

## 5. 结论与建议

1. **不优化剩余 Reformat。** 它们合计 0.075–0.086 ms（约 4% 的 kernel 时间，约 3.5% 的墙钟延迟），即使全部消除，上限也只有约 0.08 ms。
2. **Concat 拷贝**是 Reformat 中最大的一部分，但它来自 TRT 的 Concat 实现（输入无法直接写入输出的通道块）。消除它需要改动网络结构（例如避免 Upsample 直接进入 Concat），收益上限约 0.04 ms，性价比低。
3. **Detect 的 FP32 路径**涉及 FP32 常量与形状计算。把它改为 FP16 会影响坐标精度，只有在确认 mAP 不受影响时才值得尝试，收益上限约 0.03–0.05 ms。
4. **更值得关注的方向**：
   - 卷积 kernel（约 1.08 ms，67%），如 `sm80_xmma_fprop_implicit_gemm` 与深度卷积（`sm50_xmma_convolution_depthwiseHMMA`）的 tactic 选择；
   - 逐元素融合层（约 0.41 ms，25%，72 个 kernel）；
   - kernel 之间的空隙（约 0.5 ms）。可以尝试用 CUDA Graph 捕获 TRT 的 `execute_async_v3` 调用，减少启动开销（未测量，需要单独实验）。

---

## 6. 文件

- `outputs/reformat/layers_verbose.json`：DETAILED 折叠引擎的 inspector 逐层信息（含 ONNX 来源）
- `outputs/reformat/layer_times.json`、`reformat_layers.json`、`classified.json`：逐层耗时与分类
- `outputs/reformat/poc_lean_layers.json`、`poc_lean_summary.json`、`poc_bench.json`：概念验证的引擎信息与延迟
- `outputs/reformat/nsys/`：DETAILED 折叠与精简引擎的 nsys 报告与 kernel 汇总 CSV
- 注意：概念验证的引擎文件在第一次运行时复用了一个旧的 `poc_lean_fp16_detailed.engine`（与当前 ONNX 不一致），已删除并重建，以下结果均基于重建后的引擎。
