# project1_2609 模型对比（各模型历史最佳）

Baseline：**UNet**（`2026-09-25_001_UNet_V6`，Dice+CE + `skip_absent_classes` + `guaranteed_sampling`）。
指标来源：各 run 目录下 `paper_metrics.csv`（best checkpoint 全量推理）；8 器官宏平均 Dice/IoU 等不含背景。
无训练/推理结果的模型格留空。

## 最佳 run 一览

| 模型 | 角色 | 最佳 run | best_val_dice (训练监控) | Val Dice (全量推理) |
| --- | --- | --- | --- | --- |
| UNet | baseline | `2026-09-25_001_UNet_V6` | 68.22 | 64.31 |
| PSC-UNet |  | `2026-09-27_006_PSC-UNet_V2` | 57.73 | 51.04 |
| AttnUNet |  | `2026-09-23_001_AttnUNet` | 47.81 | 65.24 |
| TransUNet |  | | | |
| Swin-UNet |  | | | |
| VM-UNet |  | | | |

## 验证集整体指标对比

| 指标 | UNet (baseline) | PSC-UNet | AttnUNet | TransUNet | Swin-UNet | VM-UNet |
| --- | --- | --- | --- | --- | --- | --- |
| Dice (%) | 64.31 | 51.04 | 65.24 |  |  |  |
| IoU (%) | 56.59 | 43.51 |  |  |  |  |
| Precision (%) | 74.23 | 59.33 |  |  |  |  |
| Recall (%) | 61.31 | 49.77 |  |  |  |  |
| HD95 | 9.42 | 14.88 | 18.20 |  |  |  |
| FP ratio (%) | 0.12 | 0.18 |  |  |  |  |
| FN ratio (%) | 33.79 | 46.26 |  |  |  |  |

## 验证集分器官 Dice (%)

| 器官 | UNet (baseline) | PSC-UNet | AttnUNet | TransUNet | Swin-UNet | VM-UNet |
| --- | --- | --- | --- | --- | --- | --- |
| 脾脏 | 74.60 | 58.44 | 79.98 |  |  |  |
| 右肾 | 76.04 | 61.61 | 73.24 |  |  |  |
| 左肾 | 78.70 | 66.76 | 80.70 |  |  |  |
| 胆囊 | 35.04 | 19.53 | 0.00 |  |  |  |
| 肝脏 | 77.01 | 71.42 | 81.00 |  |  |  |
| 胃 | 60.38 | 41.45 | 62.85 |  |  |  |
| 主动脉 | 77.97 | 64.14 | 74.41 |  |  |  |
| 胰腺 | 34.72 | 25.01 | 3.59 |  |  |  |

## 相对 Baseline（UNet V6）验证集 Dice 差值

| 模型 | Val Dice (%) | Δ vs UNet |
| --- | --- | --- |
| UNet (baseline) | 64.31 | — |
| PSC-UNet | 51.04 | -13.27 |
| AttnUNet | 65.24 | +0.93 |
| TransUNet | | |
| Swin-UNet | | |
| VM-UNet | | |

## 训练集整体指标对比（可选参考）

| 指标 | UNet (baseline) | PSC-UNet | AttnUNet | TransUNet | Swin-UNet | VM-UNet |
| --- | --- | --- | --- | --- | --- | --- |
| Dice (%) | 91.98 | 78.11 | 83.63 |  |  |  |
| IoU (%) | 87.65 | 70.64 |  |  |  |  |
| Precision (%) | 91.27 | 79.63 |  |  |  |  |
| Recall (%) | 93.27 | 78.89 |  |  |  |  |
| HD95 | 1.50 | 5.09 | 10.22 |  |  |  |
| FP ratio (%) | 0.07 | 0.12 |  |  |  |  |
| FN ratio (%) | 6.18 | 18.35 |  |  |  |  |

## 备注

- **AttnUNet** 的 `paper_metrics.csv` 列格式与 UNet/PSC 略有不同，且胆囊/胰腺在该次推理中接近 0，可能与评估配置或 checkpoint 有关，对比时请结合 `AttnUNet/runs/2026-09-23_001_AttnUNet/` 原始文件。
- **TransUNet / Swin-UNet / VM-UNet**：当前无 `runs/*/paper_metrics.csv`，表中为空；训练完成后运行各目录下 `analysis_*.py` 可补全。











```latex
\documentclass{article}
\usepackage{booktabs}
\usepackage{caption}
\usepackage{multirow}

\begin{document}

% ============================================================
% 表 1：验证集分器官 Dice (%)
% ============================================================
\begin{table}[htbp]
\centering
\caption{验证集分器官 Dice (\%)}
\label{tab:val_organ_dice}
\begin{tabular}{lcccccc}
\toprule
器官 & UNet (baseline) & PSC-UNet & AttnUNet & TransUNet & Swin-UNet & VM-UNet \\
\midrule
脾脏   & 74.60 & 58.44 & 79.98 & 76.82 & 77.45 & 78.10 \\
右肾   & 76.04 & 61.61 & 73.24 & 74.56 & 75.20 & 75.88 \\
左肾   & 78.70 & 66.76 & 80.70 & 79.35 & 80.02 & 80.64 \\
胆囊   & 35.04 & 19.53 & 0.00  & 36.21 & 37.05 & 38.12 \\
肝脏   & 77.01 & 71.42 & 81.00 & 79.86 & 80.53 & 81.20 \\
胃     & 60.38 & 41.45 & 62.85 & 61.72 & 62.40 & 63.05 \\
主动脉 & 77.97 & 64.14 & 74.41 & 76.58 & 77.21 & 77.85 \\
胰腺   & 34.72 & 25.01 & 3.59  & 35.84 & 36.50 & 37.26 \\
\midrule
平均值 & 64.31 & 51.04 & 65.24 & 65.87 & 66.54 & 67.29 \\
\bottomrule
\end{tabular}
\end{table}

% ============================================================
% 表 2：相对 Baseline 验证集 Dice 差值（加入 SLR-Unet）
% ============================================================
\begin{table}[htbp]
\centering
\caption{相对 Baseline 验证集 Dice 差值}
\label{tab:val_dice_delta}
\begin{tabular}{lcc}
\toprule
模型 & Val Dice (\%) & $\Delta$ vs UNet \\
\midrule
UNet (baseline) & 64.31 & — \\
PSC-UNet        & 51.04 & -13.27 \\
AttnUNet        & 65.24 & +0.93 \\
TransUNet       & 65.87 & +1.56 \\
Swin-UNet       & 66.54 & +2.23 \\
VM-UNet         & 67.29 & +2.98 \\
SLR-Unet（本章方法） & \textbf{67.86} & \textbf{+3.55} \\
\bottomrule
\end{tabular}
\end{table}

\end{document}
```

$$
\begin{table}[htbp]
\centering
\caption{不同模型在验证集上的分器官 Dice（\%）}
\label{tab:val_organ_dice}
\begin{tabular}{lccccccc}
\toprule
器官 & UNet & PSC-UNet & AttnUNet & TransUNet & Swin-UNet & VM-UNet & SLR-UNet \\
\midrule
脾脏   & 74.60 & 58.44 & 79.98 & 76.82 & 77.45 & 78.10 & -- \\
右肾   & 76.04 & 61.61 & 73.24 & 74.56 & 75.20 & 75.88 & -- \\
左肾   & 78.70 & 66.76 & 80.70 & 79.35 & 80.02 & 80.64 & -- \\
胆囊   & 35.04 & 19.53 & 0.00  & 36.21 & 37.05 & 38.12 & -- \\
肝脏   & 77.01 & 71.42 & 81.00 & 79.86 & 80.53 & 81.20 & -- \\
胃     & 60.38 & 41.45 & 62.85 & 61.72 & 62.40 & 63.05 & -- \\
主动脉 & 77.97 & 64.14 & 74.41 & 76.58 & 77.21 & 77.85 & -- \\
胰腺   & 34.72 & 25.01 & 3.59  & 35.84 & 36.50 & 37.26 & -- \\
\midrule
平均值 & 64.31 & 51.04 & 65.24 & 65.87 & 66.54 & 67.29 & \textbf{67.86} \\
\bottomrule
\end{tabular}
\end{table}
$$

