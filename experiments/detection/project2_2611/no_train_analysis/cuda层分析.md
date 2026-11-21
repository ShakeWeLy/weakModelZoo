# TensorRT 部署分析：延迟、精度与 ChannelShuffle 消融

日期：2026-11-21
模型：SGC 无注意力（`sgc_yolov5s_c3_sgdown_gbot_dw5`，run `2026-11-17_015_…_seed42_V1`，best.pth）与 YOLOv5s（run `2026-10-30_002_YOLOv5s_V3`，best.pth），单类别（drone），640 输入。
硬件：RTX 3060 Laptop，TensorRT 10.13.3.9，PyTorch 2.6.0+cu124。

---

## 0. 结论摘要

1. **部署延迟**：TRT FP16 把 SGC 从 PyTorch（BN 折叠）eager 的 12.6 ms 降到 3.5 ms；YOLOv5s 从 7.4 ms 降到 1.9 ms。同一次运行中，SGC 的 TRT FP16 延迟约为 YOLOv5s 的 1.9 倍。
2. **精度**：BN 折叠与 TRT FP32 对两个模型的 mAP 都没有可检测的变化。TRT FP16 对 SGC 没有可检测的变化；对 YOLOv5s，mAP50-95 下降 0.31 个点（95% CI 不含 0），mAP50 不显著变化。
3. **误差来源（假设，未因果验证）**：YOLOv5s 的 FP16 误差集中在高置信框的坐标上，且其输出坐标落在 FP16 的量化网格上，符合"解码在 FP16 中完成导致定位精度下降，并只体现在严格 IoU 阈值上"的解释。SGC 不存在该现象，原因未查清。
4. **ChannelShuffle 消融**（仅估计执行成本上限，不能用于判断精度，也不是新模型结果）：去掉 Shuffle 后，TRT FP16 逐次同步延迟减少 1.17 ms（−32%），GPU kernel 数减少 57 个（−22%），布局转换 kernel 耗时减少约 0.54 ms。
5. **部署上的直接结论**：在这块 GPU 上，YOLOv5s 的 FP16 引擎同时在速度和精度（mAP50-95）上优于 SGC 的 FP16 引擎。SGC 的 FLOPs 减少没有转化为推理收益。SGC 与 YOLOv5s 的精度差来自单次训练（seed 42），需要多 seed 结果确认。

---

## 1. 部署延迟（同一次运行，batch 1，640×640，中位数，单位 ms）

| 方案 | SGC 无注意力 | YOLOv5s |
|---|---|---|
| PyTorch（BN 折叠）eager 逐次同步 | 12.63 | 7.40 |
| PyTorch（BN 折叠）eager 流水线 | 13.35 | 7.71 |
| PyTorch（BN 折叠）+ CUDA Graph | 5.13 | 4.89 |
| TRT FP32 逐次同步 / 流水线 | 4.91 / 4.63 | 4.16 / 3.94 |
| **TRT FP16 逐次同步 / 流水线** | **3.54 / 3.52** | **1.88 / 1.87** |

说明：

- 同一次运行内的数字可以比较；跨运行有波动（例如 PyTorch CUDA Graph 在不同运行中为 4.2–5.1 ms），GPU 温度与频率会影响结果。
- 重复测量的范围：TRT FP16 SGC 为 3.5–3.9 ms，YOLOv5s 为 1.6–1.9 ms。
- TRT 对默认流会有额外同步，但改用独立 CUDA 流没有可测收益（差异在 ±0.2 ms 内）。
- 作为参考，未折叠 PyTorch eager 的 SGC 约为 15.2 ms（见 `shuffle_module_bench.py` 结果）。

---

## 2. 检测精度（test 1501 张，同一预处理、同一 NMS 配置）

配置：letterbox 640，batch 1，conf 0.001，IoU 0.6，max_det 300。四个变体使用同一份预处理结果。ΔmAP 相对 PyTorch 原版；方括号为配对 bootstrap（按图像重采样 300 次）的 95% 置信区间，单位为 mAP 绝对值（0.001 = 0.1 个点）。

**SGC 无注意力**

| 变体 | mAP50 | mAP50-95 | P | R | ΔmAP50 [95% CI] | ΔmAP50-95 [95% CI] |
|---|---|---|---|---|---|---|
| PyTorch 原版 | 0.7967 | 0.4649 | 0.9218 | 0.7542 | — | — |
| PyTorch BN 折叠 | 0.7967 | 0.4649 | 0.9218 | 0.7542 | 0.0000 | 0.0000 |
| TRT FP32 | 0.7964 | 0.4648 | 0.9218 | 0.7535 | −0.0003 [−0.0015, +0.0001] | −0.0001 [−0.0005, +0.0003] |
| TRT FP16 | 0.7963 | 0.4652 | 0.9210 | 0.7535 | −0.0004 [−0.0027, +0.0014] | +0.0003 [−0.0007, +0.0011] |

**YOLOv5s**

| 变体 | mAP50 | mAP50-95 | P | R | ΔmAP50 [95% CI] | ΔmAP50-95 [95% CI] |
|---|---|---|---|---|---|---|
| PyTorch 原版 | 0.8054 | 0.4938 | 0.9204 | 0.7708 | — | — |
| PyTorch BN 折叠 | 0.8054 | 0.4938 | 0.9204 | 0.7708 | 0.0000 | 0.0000 |
| TRT FP32 | 0.8053 | 0.4938 | 0.9225 | 0.7695 | −0.0001 [−0.0001, +0.0001] | 0.0000 [−0.0004, +0.0003] |
| TRT FP16 | 0.8045 | 0.4908 | 0.9217 | 0.7688 | −0.0009 [−0.0048, +0.0040] | **−0.0031 [−0.0050, −0.0010]** |

交叉验证：官方 `analyze.py` 对 SGC PyTorch 原版得到 mAP50 0.7967、mAP50-95 0.4650，与评估循环一致（差 0.0001 来自 PyTorch 的 TF32 设置）。

---

## 3. 精度变化分析（含 YOLOv5s）

### 3.1 逐项结论

- **BN 折叠**：两个模型的 mAP 变化都在 1e-6 量级，可以视为无损。
- **TRT FP32**：两个模型都没有可检测变化。
- **TRT FP16**
  - SGC：没有可检测变化（mAP50-95 +0.0003，CI 含 0）。
  - YOLOv5s：mAP50-95 下降 0.31 个点，CI 不含 0，是真实存在的损失；mAP50 下降 0.09 个点，CI 含 0，不显著。
  - 两个模型的变化形态不同：YOLOv5s 只在 mAP50-95（严格 IoU 阈值）下降，mAP50 基本不变。这说明损失主要体现在**定位精度**，而不是检测与分类能力。

### 3.2 误差来源的分通道分析（40 张测试图，参考为 BN 折叠 PyTorch，TF32 关闭）

在高置信锚点（参考 obj > 0.25）上取各图最大绝对误差的最大值，单位为像素（坐标）或概率（obj/cls）：

| 模型 | 精度 | x | y | w | h | obj | cls |
|---|---|---|---|---|---|---|---|
| SGC | FP32 | 0.0067 | 0.0066 | 0.0270 | 0.0304 | 0.0017 | 0.0002 |
| SGC | FP16 | 0.0191 | 0.0234 | 0.0770 | 0.0832 | 0.0073 | 0.0005 |
| YOLOv5s | FP32 | 0.0048 | 0.0093 | 0.0229 | 0.0251 | 0.0013 | 0.0002 |
| YOLOv5s | FP16 | **0.1374** | **0.1366** | **0.1121** | **0.1053** | 0.0059 | 0.0006 |

要点：

- 在相关锚点上，YOLOv5s 的 FP16 xy 误差约为 SGC 的 7 倍，wh 误差约为 1.4 倍；YOLOv5s 的 FP32 误差则与 SGC 相当。
- **量化检查**：YOLOv5s FP16 输出中，|坐标| ≥ 256 的值 97.4% 落在 0.25 网格上，|坐标| ≥ 512 的值 100% 落在 0.5 网格上。这说明它的解码（`(sigmoid×2−0.5+grid)×stride`）在 FP16 中完成，大坐标的分辨率只有 0.25–0.5 像素。
- SGC 的同一检查为 0%，说明 SGC 的解码路径不在 FP16 量化网格上，具体原因（层精度选择或算子结构）未查清。
- 全部锚点的最大误差对两个模型都较大（例如 YOLOv5s FP32 的 w 误差达 2.0 px），但这些主要是低置信锚点，不影响 mAP。

### 3.3 解释（假设，未因果验证）

YOLOv5s 的 FP16 引擎在高置信框上引入约 0.1–0.14 像素的坐标误差。小目标的中位最长边约 14.5 px（见 `anchor_analysis.py` 的统计），0.1 像素约相当于框尺寸的 1%，对 IoU 的影响约 1–2%。这种影响足以改变在 IoU 0.75–0.95 阈值下的匹配结果，但不足以改变 IoU 0.5 下的匹配，因此 mAP50-95 下降而 mAP50 基本不变，与观察一致。

**注意**：这是一个与观察相符的解释，而不是因果证明。要验证它，可以把解码层强制为 FP32（TRT 逐层精度控制），重新构建引擎并重测 mAP50-95。如果损失消失，即可确认解码的 FP16 量化是主要原因。

### 3.4 其他注意事项

- 结论基于单次训练（seed 42）。SGC 的多 seed 标准差约为 1.5–2 个点，因此 SGC 与 YOLOv5s 之间约 2.6 个点的 mAP50-95 差距需要多 seed 结果确认。
- 误差分析用的是 40 张测试图，TRT 速度实验与 mAP 评估使用的是随机输入（速度）和完整测试集（精度）。

---

## 4. ChannelShuffle 消融（仅用于估计执行成本上限）

**重要**：去掉 ChannelShuffle 后模型不再与原模型等价，本实验**没有评估精度**，结果不能用于判断检测精度，也不能作为新模型的性能结论。它的唯一目的是估计 Shuffle 对 TRT 执行成本的上限。

ONNX 导出时把所有 `ChannelShuffle.forward` 替换为恒等映射，其他全部相同（同样的 BN 折叠、同样的 FP16 构建配置）。

### 4.1 结果

| 指标 | 原始 SGC FP16 | 去 Shuffle（消融） | 变化 |
|---|---|---|---|
| 逐次同步延迟（3 次交替中位数） | 3.70 ms | 2.53 ms | −1.17 ms（−32%） |
| 流水线延迟（中位数） | 3.55 ms | 2.65 ms | −0.90 ms（−25%） |
| GPU kernel 数 / 次推理（nsys，100 次） | 262 | 205 | −57（−22%） |
| GPU kernel 总耗时 / 次推理（nsys） | 2.169 ms | 1.673 ms | −0.50 ms（−23%） |
| 布局转换 kernel（genericReformat + nchwTonhwc）数 / 次 | 80（51 + 29） | 25 | −55 |
| 布局转换 kernel 耗时 / 次 | 0.648 ms（占 29.9%） | 0.107 ms（占 6.4%） | −0.54 ms |
| 卷积 / GEMM kernel 耗时 / 次 | 1.012 ms | 1.104 ms | +0.09 ms |
| 逐元素 kernel 耗时 / 次 | 0.458 ms | 0.412 ms | −0.05 ms |
| 其它 kernel 耗时 / 次 | 0.052 ms | 0.050 ms | ≈0 |
| 引擎层数（inspector） | 325 | 227 | −98 |
| 层名含 shuffle 的层 | 82 | 0 | — |
| 层名含 Reformatting 的层 | 65 | 29 | −36 |
| Shuffle 相关层耗时（IProfiler，含同步开销） | 0.877 ms | 0 | 偏高，仅作参考 |
| Reformatting 层耗时（IProfiler） | 0.664 ms | 0.229 ms | −0.44 ms |
| 引擎文件大小 | 13.0 MB | 12.8 MB | 基本不变 |

ONNX 算子差异：Reshape 69 → 9，Transpose 33 → 3，Constant 206 → 146，Concat 50 → 48（30 个 ChannelShuffle 对应 60 个 Reshape 与 30 个 Transpose）。

### 4.2 解读

1. **延迟上限**：去掉 Shuffle 后，延迟减少约 1.17 ms（逐次同步）或 0.90 ms（流水线），相当于原始引擎延迟的 25–32%。
2. **kernel 时间与调度间隙**：kernel 总耗时只减少 0.50 ms，剩余约 0.67 ms 的延迟下降来自 kernel 数减少带来的调度间隙（粗略估计：间隙从 1.53 ms 降到 0.86 ms，依赖 nsys 采集的 kernel 时间，仅作参考）。
3. **Shuffle 的主要成本是布局转换**：Shuffle 相关的 NCHW↔NHWC 转换 kernel 占原始引擎 kernel 时间的约 30%。去掉 Shuffle 后这部分只剩 0.107 ms（其中仍有 25 个 genericReformat，来自网络中其他的布局需求）。
4. **并非纯粹的 Shuffle 成本**：卷积 kernel 时间增加 0.09 ms，说明去掉 Shuffle 后 TRT 选择了不同的卷积实现。因此 1.17 ms 的数字包含了 kernel 选择变化的副作用，是上限估计而非精确代价。
5. **与 IProfiler 的一致性**：IProfiler 逐层求和的变化（5.19 → 4.09 ms）与延迟变化方向一致；Shuffle + Reformatting 占逐层求和的 29.6%，与 nsys 测得的布局转换占比 29.9% 接近。

### 4.3 对后续工作的含义

- 要保留等价性，可以考虑把通道置换折叠进消费者的权重，从而避免布局转换。但 SGC 中 shuffle 的输出还会与残差、门控等张量相加或相乘，需要逐个分析消费者，不能一概而论。这一点还未验证。
- 部署侧 `shuffle_ablation_build.py` 无法回答「去掉 Shuffle 后 mAP 会变多少」，需在训练期关闭混洗并重新训评。**已加 3 seed（42/43/44）**，与有 Shuffle 的 `015/016/017`（`sgc_yolov5s_c3_sgdown_gbot_dw5_nodfc_cbam_none*`，seed 42/43/44）一一对照；`model.params.use_shuffle = false`（`SGhostConv` / `GSConv` 不再调用 `ChannelShuffle`）。
  - 配置：`sgc_yolov5s_c3_sgdown_gbot_dw5_nodfc_cbam_none_no_shuffle`、`_seed43`、`_seed44`
  - Run：`2026-11-21_037/038/039_SGC_c3_sgdown_gbot_dw5_s_noDFC_cbamNone_noShuffle_seed{42,43,44}_V1`
  - 队列：`bash experiments/detection/project2_2611/run_ablation_sgc.sh`（训完后 `analyze.py --split test --checkpoint best --no-vis`，并与 015–017 对比 mAP）

---

## 5. 后续建议

1. **验证 YOLOv5s FP16 的误差假设**：强制解码层为 FP32，重新构建引擎并重测 mAP50-95，检验是否能消除 −0.31 个点的损失。
2. **多 seed 确认 SGC 与 YOLOv5s 的精度差距**（约 2.6 个点的 mAP50-95）。
3. **检查 SGC 解码路径的精度**：确认为什么其坐标不在 FP16 量化网格上，以及其 FP16 误差较小的原因。
4. **等价的 Shuffle 优化**（如果部署需要）：在保持等价的前提下折叠通道置换，再测延迟与 mAP。

---

## 6. 文件与复现

脚本（`no_train_analysis/`）：

- 部署：`trt_deploy_bench.py`（ONNX 导出、引擎构建、延迟）、`bn_fold.py`（BN 折叠）、`trt_stream_layer_check.py`（独立流与逐层 IProfiler）
- 精度：`trt_map_eval.py`（同一循环评估四个变体）、`trt_map_bootstrap.py`（配对 bootstrap）、`fp16_error_analysis.py`（分通道误差与量化检查）
- ChannelShuffle 消融：`shuffle_ablation_build.py`、`shuffle_ablation_bench.py`、`shuffle_ablation_nsys_driver.py`

结果（`outputs/`）：

- `trt/trt_deploy_2026-11-21.json`、`trt/trt_stream_layer_2026-11-21.json`
- `map_eval/map_eval_2026-11-21.json`、`map_eval/map_bootstrap_2026-11-21.json`、`map_eval/fp16_error_analysis.json`、`map_eval/stats_*.pkl`
- `shuffle_ablation/bench_2026-11-21.json`、`shuffle_ablation/onnx_op_counts.json`、`shuffle_ablation/nsys/`（nsys 报告与 kernel 汇总 CSV）

官方 `analyze.py` 对 SGC 的评估结果写入了该 run 的 `predictions/test/metrics.json`。
