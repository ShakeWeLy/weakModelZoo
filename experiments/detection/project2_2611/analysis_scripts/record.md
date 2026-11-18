| Name                                   | Backbone                     | Neck           | Attention / 残差                      |   Seed   |     mAP50     |   mAP50-95    | Params (M) | FLOPs (G) | 状态                                             |
| :------------------------------------- | :--------------------------- | :------------- | :------------------------------------ | :------: | :-----------: | :-----------: | :--------: | :-------: | :----------------------------------------------- |
| YOLOv5s                                | 标准 YOLOv5 Conv + C3 + SPPF | 标准 YOLOv5 C3 | 无                                    | 42/43/44 | 0.808 ± 0.003 | 0.492 ± 0.003 |    7.02    |   15.94   | 3 seed 完成                                      |
| c3topo_1                               | dense + Conv↓                | VOVGCSPYolo    | CBAM（full）                          |    42    |     0.799     |     0.483     |    6.11    |   14.34   | 单 seed                                          |
| c3_sgdown                              | dense + SGhost↓              | VOVGCSPYolo    | CBAM（full）                          |    42    |     0.783     |     0.479     |    5.33    |   12.46   | 单 seed                                          |
| c3_ghostbot_dw3                        | ghost-bot dw3 + Conv↓        | VOVGCSPYolo    | CBAM（full）+ DFC + dw 残差           |    42    |     0.774     |     0.458     |    5.52    |   12.64   | 单 seed（数值来自报告）                          |
| c3_ghostbot_dw5                        | ghost-bot dw5 + Conv↓        | VOVGCSPYolo    | CBAM（full）+ DFC + dw 残差           |    42    |     0.777     |     0.463     |    5.52    |   12.68   | 单 seed                                          |
| c3_ghostbot_dw7                        | ghost-bot dw7 + Conv↓        | VOVGCSPYolo    | CBAM（full）+ DFC + dw 残差           |    42    |     0.796     |     0.468     |    5.53    |   12.74   | 单 seed                                          |
| c3_sgdown_gbot_dw5                     | ghost-bot dw5 + SGhost↓      | VOVGCSPYolo    | CBAM（full）+ DFC + dw 残差           | 42/43/44 | 0.766 ± 0.020 | 0.455 ± 0.013 |    4.74    |   10.80   | 3 seed 完成                                      |
| c3_sgdown_gbot_dw5（P3 去 SAM）        | ghost-bot dw5 + SGhost↓      | VOVGCSPYolo    | P3 仅 CAM，P4/P5 CBAM + DFC + dw 残差 |    42    |     0.782     |     0.469     |    4.74    |     —     | 单 seed（`2026-11-17_006_…_cbamCamP3_V1`）       |
| c3_sgdown_gbot_dw5（无 DFC）           | ghost-bot dw5 + SGhost↓      | VOVGCSPYolo    | CBAM（full）+ 无 DFC + dw 残差        |    42    |     0.792     |     0.477     |    4.73    |     —     | 单 seed（`2026-11-17_007_…_noDFC_V1`）           |
| c3_sgdown_gbot_dw5（恒等残差）         | ghost-bot dw5 + SGhost↓      | VOVGCSPYolo    | CBAM（full）+ DFC + 恒等残差          |    42    |     0.754     |     0.449     |    4.73    |     —     | 单 seed（`2026-11-17_008_…_residIdentity_V1`）   |
| c3_sgdown_gbot_dw5（无 CBAM）          | ghost-bot dw5 + SGhost↓      | VOVGCSPYolo    | 无 CBAM + DFC + dw 残差               |    42    |     0.787     |     0.478     |    4.69    |     —     | 单 seed（`2026-11-17_005_…_cbamNone_V1`）        |
| c3_sgdown_gbot_dw5（无 DFC + 无 CBAM） | ghost-bot dw5 + SGhost↓      | VOVGCSPYolo    | 无 CBAM + 无 DFC + dw 残差            | 42/43/44 |      —        |      —        |    4.69    |     —     | **无效 run**：本地误用 train=14/val=3（非 7000/1500），test@42=0.000；须在完整 Det_Fly 上重训后重评 |
| c3_sgdown_gbot_dw5（本征占比 0.625）   | ghost-bot dw5 + SGhost↓      | VOVGCSPYolo    | CBAM（full）+ DFC + dw 残差           | 42/43/44 | 0.773 ± 0.004 | 0.461 ± 0.007 |    5.12    |     —     | 3 seed 完成（test，1501 张）                     |
| c3_sgdown_gbot_dw5（本征占比 0.75）    | ghost-bot dw5 + SGhost↓      | VOVGCSPYolo    | CBAM（full）+ DFC + dw 残差           | 42/43/44 | 0.777 ± 0.007 | 0.468 ± 0.004 |    5.45    |     —     | 3 seed 完成（test，1501 张）                     |
| c3_sgdown_gbot_dw7                     | ghost-bot dw7 + SGhost↓      | VOVGCSPYolo    | CBAM（full）+ DFC + dw 残差           |    42    |     0.783     |     0.479     |    4.75    |   10.86   | 单 seed（用 V2，V1 失败）                        |

### 无 DFC + 无 CBAM：test 评估记录（2026-11-17）

- **口径**：`analyze.py --split test --checkpoint best --no-vis`，与 `configs/global.toml` 的 `[test]`（conf=0.001、IoU=0.6、max_det=300）及同日其它消融一致。
- **已执行（seed 42）**：`sgc_yolov5s_c3_sgdown_gbot_dw5_nodfc_cbamnone` → run `_015_…_seed42` → **mAP50=0.0000，mAP50-95=0.0000**（`predictions/test/metrics.json`）。seed 43/44 未再跑（同批误训，结果必然为 0）。
- **根因**：`train.log` 记载 `train=14 / val=3`；本仓 `data/Det_Fly` 当前仅 1518 张图（test 1501 + 少量 train/val）。同日上午的 3-seed 本模型（`train=7000`）说明当时曾有完整集，晚间队列在残缺数据上误训约 3 min/40 epoch。
- **对照基线（完整 c3_sgdown_gbot_dw5，3 seed）**：mAP50 **0.766 ± 0.020**，mAP50-95 **0.455 ± 0.013**（`size_ap/multi_2026-11-17.json`：`ours_s42/s43/s44`）。
- **重训 + 重评**（仓库根目录，恢复 `train 7000 / val 1500 / test 1501` 后）：

```bash
# 删除无效 run 后按配置名重训并自动 test
for m in sgc_yolov5s_c3_sgdown_gbot_dw5_nodfc_cbam_none \
         sgc_yolov5s_c3_sgdown_gbot_dw5_nodfc_cbam_none_seed43 \
         sgc_yolov5s_c3_sgdown_gbot_dw5_nodfc_cbam_none_seed44; do
  python experiments/detection/project2_2611/train.py -m "$m" --analyze
done
```

- **服务器**：`server/manifest.csv` 已加入 `nodfc_cbamnone_s42/s43/s44`（`run_queue.py --analyze` 口径）。

### 通道冗余对比：Ghost 本征占比 0.5 / 0.625 / 0.75（2026-11-18）

- **对象**：`c3_sgdown_gbot_dw5` 的 SGhost 下采样输出（`backbone.stage{2..5}.0`）。r=0.5（基线，`ratio=2`）、0.625、0.75 各 3 个 seed（42/43/44），共 9 个 run。test 集随机 20 张（与 `redundancy_analysis.py` 相同抽样），逐图计算后平均。
- **脚本**：`redundancy_ir_analysis.py`，输出 `redundancy_ir/`（`ir_redundancy.json`、`ir_metrics.png`、`ir_corr_stage3.png`）。
- **指标**：全部通道的 NN\|r\|、dup%、mean\|r\|（与 `redundancy_analysis.py` 同口径）。幽灵相关指标：copy_frac（与任一本征 \|r\|>0.9 的幽灵比例）、R2_group（幽灵被其父组线性回归解释的方差）、dup_gg（幽灵之间的近似重复比例）。3 seed 标准差为样本标准差（ddof=1），与上表 test 指标口径一致。

**表 1 全部通道（三组通道数相同，可直接比较）**

| 本征占比 r | 参数量 (M) | test mAP50 | dup%：s2 / s3 / s4 / s5 | NN\|r\|：s2 / s3 / s4 / s5 | mean\|r\|：s2 / s3 / s4 / s5 |
|:--|--:|:--|:--|:--|:--|
| 0.5（基线） | 4.74 | 0.766 ± 0.020 | 0.71 / 0.66 / 0.80 / 0.82 | 0.91 / 0.91 / 0.93 / 0.94 | 0.45 / 0.31 / 0.37 / 0.35 |
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
5. **精度没有显著收益，参数增加。** test mAP50 为 0.766 ± 0.020 → 0.773 ± 0.004 → 0.777 ± 0.007，差异小于基线 seed 间标准差（0.020）。参数量分别增加 8%（0.625）和 15%（0.75），来自本征主卷积和分组幽灵卷积的增长。
6. **混杂因素。** 本征通道数随 r 增加（stage3：64 → 80 → 96），copy 与 NN 对候选数量敏感。参数量同时变化，卷积结构也变了，不是只改冗余的纯对照。只用 20 张图，每组 3 个 seed。

**与轻量化的关系**：提高本征占比能降低“复制型”冗余，但没有带来显著精度收益，还增加了 8–15% 参数，对轻量化不利。复制型冗余在基线上并不是精度瓶颈。若目标是轻量化，应从减少算子数和计算量入手（见 2026-11-17 的延迟分析）。

**代码改动（需注意）**：为运行本分析，修复了 `src/models/detection/YOLO/SGC_YOLOv5/sgc_yolov5.py` 的导入：`CAM` 改为 `from src.module.attention.CAM.cam import CAM`。原因是提交 `821137d`（DSTFP）把 `src/module/attention/__init__.py` 改为只导出 `SelfAttention`，SGC 模型构建时报 `'module' object is not callable`。这同样影响服务器上的训练队列。`__init__.py` 未改动，本次修复尚未提交，需要确认后再入库。

**复现**：`python experiments/detection/project2_2611/redundancy_ir_analysis.py --num 20`（仓库根目录，CPU 约 100 s）。

