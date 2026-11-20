| Name                                   | Backbone                     | Neck            | Attention / 残差                      |     Seed     |       mAP50       |     mAP50-95      | Params (M) | FLOPs (G) | 状态                                                         |
| :------------------------------------- | :--------------------------- | :-------------- | :------------------------------------ | :----------: | :---------------: | :---------------: | :--------: | :-------: | :----------------------------------------------------------- |
| YOLOv5s                                | 标准 YOLOv5 Conv + C3 + SPPF | 标准 YOLOv5 C3  | 无                                    |   42/43/44   |   0.808 ± 0.003   |   0.492 ± 0.003   |    7.02    |   15.94   | 3 seed 完成                                                  |
| c3topo_1                               | dense + Conv↓                | VOVGCSPYolo     | CBAM（full）                          |      42      |       0.799       |       0.483       |    6.11    |   14.34   | 单 seed                                                      |
| c3_sgdown                              | dense + SGhost↓              | VOVGCSPYolo     | CBAM（full）                          |      42      |       0.783       |       0.479       |    5.33    |   12.46   | 单 seed                                                      |
| c3_ghostbot_dw3                        | ghost-bot dw3 + Conv↓        | VOVGCSPYolo     | CBAM（full）+ DFC + dw 残差           |      42      |       0.774       |       0.458       |    5.52    |   12.64   | 单 seed（数值来自报告）                                      |
| c3_ghostbot_dw5                        | ghost-bot dw5 + Conv↓        | VOVGCSPYolo     | CBAM（full）+ DFC + dw 残差           |      42      |       0.777       |       0.463       |    5.52    |   12.68   | 单 seed                                                      |
| c3_ghostbot_dw7                        | ghost-bot dw7 + Conv↓        | VOVGCSPYolo     | CBAM（full）+ DFC + dw 残差           |      42      |       0.796       |       0.468       |    5.53    |   12.74   | 单 seed                                                      |
| c3_sgdown_gbot_dw3                     | ghost-bot dw3 + SGhost↓      | VOVGCSPYolo     | CBAM（full）+ DFC + dw 残差           |   42/43/44   |   0.765 ± 0.005   |   0.456 ± 0.006   |    4.73    |   10.76   | 3 seed（`2026-11-19_025`～`027`，test）                      |
| **c3_sgdown_gbot_dw5**                 | **ghost-bot dw5 + SGhost↓**  | **VOVGCSPYolo** | **CBAM（full）+ DFC + dw 残差**       | **42/43/44** | **0.765 ± 0.018** | **0.456 ± 0.016** |  **4.74**  | **10.80** | 3 seed 完成（s42 用 `2026-11-18_018_…_rerun_V2` test）       |
| c3_sgdown_gbot_dw5（P3 去 SAM）        | ghost-bot dw5 + SGhost↓      | VOVGCSPYolo     | P3 仅 CAM，P4/P5 CBAM + DFC + dw 残差 |      42      |       0.782       |       0.469       |    4.74    |   10.80   | 单 seed（`2026-11-17_006_…_cbamCamP3_V1`）                   |
| c3_sgdown_gbot_dw5（无 DFC）           | ghost-bot dw5 + SGhost↓      | VOVGCSPYolo     | CBAM（full）+ 无 DFC + dw 残差        |   42/43/44   |   0.785 ± 0.008   |   0.470 ± 0.006   |    4.73    |   10.75   | 3 seed（`007`+`2026-11-19_019`/`020`，test）                 |
| c3_sgdown_gbot_dw5（恒等残差）         | ghost-bot dw5 + SGhost↓      | VOVGCSPYolo     | CBAM（full）+ DFC + 恒等残差          |   42/43/44   |   0.765 ± 0.010   |   0.456 ± 0.007   |    4.73    |   10.74   | 3 seed（`008`+`2026-11-19_023`/`024`，test）                 |
| c3_sgdown_gbot_dw5（无 CBAM）          | ghost-bot dw5 + SGhost↓      | VOVGCSPYolo     | 无 CBAM + DFC + dw 残差               |   42/43/44   |   0.782 ± 0.009   |   0.467 ± 0.011   |    4.69    |   10.79   | 3 seed（`005`+`2026-11-19_021`/`022`，test）                 |
| c3_sgdown_gbot_dw5（无 DFC + 无 CBAM） | ghost-bot dw5 + SGhost↓      | VOVGCSPYolo     | 无 CBAM + 无 DFC + dw 残差            |   42/43/44   |   0.802 ± 0.015   |   0.482 ± 0.016   |    4.69    |   10.75   | 3 seed（`015`/`016_V2`/`017`，test；2026-11-18 完整 Det_Fly） |
| c3_sgdown_gbot_dw5（本征占比 0.625）   | ghost-bot dw5 + SGhost↓      | VOVGCSPYolo     | CBAM（full）+ DFC + dw 残差           |   42/43/44   |   0.773 ± 0.004   |   0.461 ± 0.007   |    5.12    |   11.90   | 3 seed 完成（test，1501 张）                                 |
| c3_sgdown_gbot_dw5（本征占比 0.75）    | ghost-bot dw5 + SGhost↓      | VOVGCSPYolo     | CBAM（full）+ DFC + dw 残差           |   42/43/44   |   0.777 ± 0.007   |   0.468 ± 0.004   |    5.45    |   12.69   | 3 seed 完成（test，1501 张）                                 |
| c3_sgdown_gbot_dw7                     | ghost-bot dw7 + SGhost↓      | VOVGCSPYolo     | CBAM（full）+ DFC + dw 残差           |   42/43/44   |   0.759 ± 0.043   |   0.453 ± 0.036   |    4.75    |   10.85   | 3 seed（s42=`…_030_…_V2`；s43 偏低见下节）                   |

### 待训（2026-11-20）：seed 45 + GhostNetV2 DFC

- **动机**：dw5 / 无 DFC+无 CBAM / dw7 的 seed 方差偏大；dw7 s43 坏收敛。补 **seed 45**；并增加 **仅改 DFC** 的基线对照。
- **DFC 实现**：历史 `dfc_type=strip`（默认）为 1×5 / 5×1 条带 depthwise **无池化**；`dfc_type=ghostv2` 对齐 GhostNetV2（`avg_pool2` → short_conv → sigmoid → nearest 上采样门控）。旧 checkpoint **不兼容** `ghostv2`。
- **队列**（`run_ablation_sgc.sh`）：`seed45`×3 + `sgc_yolov5s_c3_sgdown_gbot_dw5_dfc_ghostv2`（s42）。
- **汇总建议**：dw7 报告时用 **42/44/45** 或剔除 s43；无 DFC+无 CBAM 可报 **42/43/44/45** 四 seed 或 **42/43/45** 去掉 s44 离群。

| 配置 | `-m` | seed |
|:--|:--|:--:|
| dw5 基线 | `sgc_yolov5s_c3_sgdown_gbot_dw5_seed45` | 45 |
| dw7 | `sgc_yolov5s_c3_sgdown_gbot_dw7_seed45` | 45 |
| 无 DFC+无 CBAM | `sgc_yolov5s_c3_sgdown_gbot_dw5_nodfc_cbam_none_seed45` | 45 |
| dw5 + DFC GhostV2 | `sgc_yolov5s_c3_sgdown_gbot_dw5_dfc_ghostv2` | 42 |

### 服务器批次（2026-11-19）：dw3/dw7 与结构消融 43/44

- **来源**：`project2_2611/new/` 已迁入各 `SGC_YOLOv5_*/runs/`，并执行 `analyze.py --split test --checkpoint best --no-vis`（1501 张）。
- **Ghost 瓶颈 DW 扫描（SGhost↓）**

| 变体 | Seed 42 | Seed 43 | Seed 44 | test mAP50 | test mAP50-95 |
|:--|--:|--:|--:|--:|--:|
| **dw3** | 0.771 | 0.765 | 0.761 | **0.765 ± 0.005** | 0.456 ± 0.006 |
| **dw5（基线）** | 0.785 | 0.750 | 0.759 | **0.765 ± 0.018** | 0.456 ± 0.016 |
| **dw7** | 0.783 | **0.709** | 0.785 | **0.759 ± 0.043** | 0.453 ± 0.036 |

- **解读（DW）**
  1. **dw3 与 dw5 几乎等价**：三 seed 方差很小（±0.005），均值与 dw5 基线重合；缩小瓶颈 DW（5→3）**不带来精度收益，也未改善种子稳定性**。
  2. **dw7 仍不稳定**：seed 43 test mAP50 仅 **0.709**（比同配置 val 最佳 ~0.75 跌约 4pt），把三 seed 标准差拉回到 **0.043**（与早期 dw5 同量级）。seed 42/44 仍在 0.783–0.785，说明 **dw7 对初始化敏感，不是一致性的“更大感受野”收益**。
  3. **相对 Conv↓ 对照**（表上 `c3_ghostbot_dw*` 单 seed）：SGhost↓ 的 dw5/dw3 test 仍低于 Conv↓ dw7（0.796），轻量化路线仍在小目标/召回上付出代价。

- **结构消融 3 seed（相对 dw5 基线 0.765 / 0.456）**

| 消融 | mAP50（3 seed） | mAP50-95 | 相对基线 |
|:--|:--|:--|:--|
| 无 DFC | **0.785 ± 0.008** | 0.470 ± 0.006 | mAP50 **+0.020**（意外偏高，需结合延迟再论） |
| 无 CBAM | 0.782 ± 0.009 | 0.467 ± 0.011 | mAP50 +0.017 |
| 无 DFC + 无 CBAM | **0.802 ± 0.015** | **0.482 ± 0.016** | mAP50 **+0.037**，mAP50-95 +0.026 |
| 恒等残差 | 0.765 ± 0.010 | 0.456 ± 0.007 | 与基线重合（单 seed 0.754 属 seed 噪声） |

- **解读（消融）**
  1. 补齐 43/44 后，**去 DFC / 去 CBAM** 均值略高于 dw5 三 seed，但优势 **小于 dw5 seed 间 std（0.018）**。
  2. **无 DFC + 无 CBAM**（`015`/`016_V2`/`017`，`train=7000`）test 均值 **高于完整 dw5 基线**（mAP50 0.802 vs 0.765）；与「无效 14/3 训练」无关，early test=0 为误读。seed 44 达 **0.818**，seed 间 std 0.015，仍高于 YOLOv5s（0.003），但优于多数 SGhost 变体。
  3. **恒等残差** 三 seed 与基线持平；dw 残差主要影响延迟/动力学，非平均 test AP。

### 基线 3 seed：seed 42 重训（2026-11-18）

- **动机**：首训 `2026-11-13_029_…_V1` 在 test 上偏高（mAP50 **0.789**，mAP50-95 **0.470**），拉高 3 seed 汇总标准差（原 **0.766 ± 0.020** / **0.455 ± 0.013**）。
- **重训**：`sgc_yolov5s_c3_sgdown_gbot_dw5_seed42_rerun` → `2026-11-18_018_SGC_c3_sgdown_gbot_dw5_s_seed42_rerun_V2`（40 epoch，配置与基线一致）。
- **test 单 seed**（`analyze.py --split test --checkpoint best`，1501 张）：

| Seed | Run | mAP50 | mAP50-95 |
|:--:|:--|--:|--:|
| 42（rerun） | `…_018_…_rerun_V2` | 0.785 | 0.474 |
| 43 | `…_001_…_seed43_V1` | 0.750 | 0.446 |
| 44 | `…_002_…_seed44_V1` | 0.759 | 0.448 |

- **汇总**（样本标准差 ddof=1）：mAP50 **0.765 ± 0.018**，mAP50-95 **0.456 ± 0.016**。相对重训前，mAP50 标准差由 0.020 略降至 0.018，但仍远高于 YOLOv5s 同批 3 seed（**0.003**）；mAP50-95 标准差由 0.013 升至 0.016。说明 seed 42 偏高并非唯一因素，本模型在 test 上对初始化仍较敏感。

### 无 DFC + 无 CBAM：test 评估（2026-11-18 训练，2026-11-20 补评）

- **训练**：三 seed 均在 **完整 Det_Fly**（`train=7000 / val=1500`）上 40 epoch 完成；`train.log` 可查。早期 record 误写「train=14」系另一批失败 run，**与 `015`/`016`/`017` 无关**。
- **test**（`analyze.py --split test --checkpoint best --no-vis`）：

| Seed | Run | mAP50 | mAP50-95 |
|:--:|:--|--:|--:|
| 42 | `…_015_…_seed42_V1` | 0.797 | 0.465 |
| 43 | `…_016_…_seed43_V2` | 0.790 | 0.483 |
| 44 | `…_017_…_seed44_V1` | 0.818 | 0.497 |

- **汇总**：mAP50 **0.802 ± 0.015**，mAP50-95 **0.482 ± 0.016**（高于 dw5 基线 **0.765 / 0.456**）。参数 **4.69M**（少 CBAM）。**无需再入 `run_ablation_sgc.sh` 训练队列**。

### 通道冗余对比：Ghost 本征占比 0.5 / 0.625 / 0.75（2026-11-18）

- **对象**：`c3_sgdown_gbot_dw5` 的 SGhost 下采样输出（`backbone.stage{2..5}.0`）。r=0.5（基线，`ratio=2`）、0.625、0.75 各 3 个 seed（42/43/44），共 9 个 run。test 集随机 20 张（与 `redundancy_analysis.py` 相同抽样），逐图计算后平均。
- **脚本**：`redundancy_ir_analysis.py`，输出 `redundancy_ir/`（`ir_redundancy.json`、`ir_metrics.png`、`ir_corr_stage3.png`）。
- **指标**：全部通道的 NN\|r\|、dup%、mean\|r\|（与 `redundancy_analysis.py` 同口径）。幽灵相关指标：copy_frac（与任一本征 \|r\|>0.9 的幽灵比例）、R2_group（幽灵被其父组线性回归解释的方差）、dup_gg（幽灵之间的近似重复比例）。3 seed 标准差为样本标准差（ddof=1），与上表 test 指标口径一致。

**表 1 全部通道（三组通道数相同，可直接比较）**

| 本征占比 r | 参数量 (M) | test mAP50 | dup%：s2 / s3 / s4 / s5 | NN\|r\|：s2 / s3 / s4 / s5 | mean\|r\|：s2 / s3 / s4 / s5 |
|:--|--:|:--|:--|:--|:--|
| 0.5（基线） | 4.74 | 0.765 ± 0.018 | 0.71 / 0.66 / 0.80 / 0.82 | 0.91 / 0.91 / 0.93 / 0.94 | 0.45 / 0.31 / 0.37 / 0.35 |
| 0.625 | 5.12 | 0.773 ± 0.004 | 0.42 / 0.32 / 0.43 / 0.39 | 0.82 / 0.80 / 0.85 / 0.85 | 0.38 / 0.32 / 0.36 / 0.38 |
| 0.75 | 5.45 | 0.777 ± 0.007 | 0.41 / 0.30 / 0.42 / 0.37 | 0.82 / 0.81 / 0.85 / 0.84 | 0.37 / 0.32 / 0.36 / 0.36 |

**表 2 幽灵分支的父组结构**

| 本征占比 r | 父组结构 | copy_frac：s2 / s3 / s4 / s5 | best_par：s2 / s3 / s4 / s5 | R2_group：s2 / s3 / s4 / s5 | dup_gg：s2 / s3 / s4 / s5 |
|:--|:--|:--|:--|:--|:--|
| 0.5 | 1 本征 → 1 幽灵（深度卷积，单父） | 0.68 / 0.65 / 0.72 / 0.78 | 0.89 / 0.90 / 0.92 / 0.93 | 0.77 / 0.80 / 0.81 / 0.85 | 0.37 / 0.14 / 0.31 / 0.22 |
| 0.625 | 5 本征 → 3 幽灵（分组卷积，`groups=gcd`） | 0.34 / 0.28 / 0.38 / 0.36 | 0.80 / 0.78 / 0.84 / 0.84 | 0.81 / 0.84 / 0.90 / 0.92 | 0.26 / 0.19 / 0.29 / 0.30 |
| 0.75 | 3 本征 → 1 幽灵（分组卷积，`groups=幽灵数`） | 0.40 / 0.38 / 0.43 / 0.42 | 0.83 / 0.84 / 0.85 / 0.86 | 0.79 / 0.85 / 0.84 / 0.89 | 0.14 / 0.08 / 0.23 / 0.23 |

图：`redundancy_ir/ir_metrics.png`（每点为一个 seed，线为均值 ± 标准差）；`redundancy_ir/ir_corr_stage3.png`（stage3 同一测试图的 \|r\| 矩阵，青线为本征/幽灵分界）。

![各占比下的通道冗余指标](redundancy_ir/ir_metrics.png)

**结论**

1. **近似复制显著减少。** r 从 0.5 增至 0.625 / 0.75 后，幽灵与任一本征 \|r\|>0.9 的比例（copy_frac）从 0.65–0.78 降到 0.28–0.43。全部通道的近似重复比例在 stage3–5 下降约 0.3–0.4（dup 0.66–0.82 → 0.30–0.43），NN\|r\| 从 0.91–0.94 降到 0.80–0.85。
2. **冗余没有消失，而是从“单通道复制”变成“组合可预测”。** 分组卷积后的幽灵仍有 R2_group 0.79–0.92，基线单父为 0.77–0.85，r=0.625 时最高。幽灵基本是父组的线性组合。
3. **平均两两 \|r\| 在三组之间几乎不变**（0.31–0.45），再次说明只看平均值会低估结构性冗余。
4. **0.625 与 0.75 结果接近。** 0.75 的父组更小（3→1），幽灵之间的近似重复（dup_gg）最低（0.08–0.23）。
5. **精度没有显著收益，参数增加。** test mAP50 为 0.765 ± 0.018 → 0.773 ± 0.004 → 0.777 ± 0.007，差异小于基线 seed 间标准差（0.018）。参数量分别增加 8%（0.625）和 15%（0.75），来自本征主卷积和分组幽灵卷积的增长。
6. **混杂因素。** 本征通道数随 r 增加（stage3：64 → 80 → 96），copy 与 NN 对候选数量敏感。参数量同时变化，卷积结构也变了，不是只改冗余的纯对照。只用 20 张图，每组 3 个 seed。

**与轻量化的关系**：提高本征占比能降低“复制型”冗余，但没有带来显著精度收益，还增加了 8–15% 参数，对轻量化不利。复制型冗余在基线上并不是精度瓶颈。若目标是轻量化，应从减少算子数和计算量入手（见 2026-11-17 的延迟分析）。

**代码改动（需注意）**：为运行本分析，修复了 `src/models/detection/YOLO/SGC_YOLOv5/sgc_yolov5.py` 的导入：`CAM` 改为 `from src.module.attention.CAM.cam import CAM`。原因是提交 `821137d`（DSTFP）把 `src/module/attention/__init__.py` 改为只导出 `SelfAttention`，SGC 模型构建时报 `'module' object is not callable`。这同样影响服务器上的训练队列。`__init__.py` 未改动，本次修复尚未提交，需要确认后再入库。

**复现**：`python experiments/detection/project2_2611/redundancy_ir_analysis.py --num 20`（仓库根目录，CPU 约 100 s）。

**FLOPs (G) 口径**（2026-11-20 补全表中 `—`）：`thop.profile`，输入 1×3×640，`FLOPs (G) = 2 × GMACs`（与 `docs/SGC_YOLOv5_实验报告.md` §2.3 一致）。dw3/dw5/dw7 随瓶颈 `mid_dw_kernel_size` 略变（10.76 / 10.80 / 10.85）；无 CBAM 几乎不变；本征占比 0.625/0.75 为 11.90 / 12.69。

