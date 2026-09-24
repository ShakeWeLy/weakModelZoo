# U-Net 训练实验

在 Synapse 2D 切片上训练 U-Net 基线，实验流程与 PSC-UNet 对齐。

## 运行

每次训练前在 `config.toml` 中设置唯一的 `[experiments].name`：

```bash
# 训练
python experiments/segment/project1_2609/UNet/train_unet.py

# 指定配置
python experiments/segment/project1_2609/UNet/train_unet.py --config experiments/segment/project1_2609/UNet/config.toml

# 快速试跑 1 个 epoch
python experiments/segment/project1_2609/UNet/train_unet.py --epochs 1 --quick

# 推理与评估（Dice / IoU / Precision / Recall / FP% / FN% / HD95）
python experiments/segment/project1_2609/UNet/analysis_unet.py
python experiments/segment/project1_2609/UNet/analysis_unet.py --split test
```

## 输出

```
UNet/runs/2026-09-24_001_UNet_V1/
├── config.yaml
├── history.csv
├── class_metrics.csv
├── summary.json
├── train.log
├── paper_metrics.csv
├── checkpoints/
│   ├── best.pth
│   └── last.pth
└── predictions/
    ├── train/
    ├── val/
    └── test/
        ├── predictions/
        ├── visualizations/
        ├── slice_metrics.csv
        ├── organ_slice_metrics.csv
        ├── organ_metrics.csv
        └── summary.json
```

## 说明

- 数据目录：`data/synapse_processed`
- 默认 **8 器官模式**：`class_nums=9`，`label_map=eval8`
- **类映射**（模型 ID → 器官）：`1 spleen, 2 rk, 3 lk, 4 gb, 5 liver, 6 stomach, 7 aorta, 8 pancreas`
- 损失函数：**加权 Dice Loss**（`dice_class_weights`）
- 优化器默认 **Adam + lr=1e-4**；`best.pth` 按 `val_eval_dice` 保存
- **Early stopping**：`early_stopping_patience=50`
- 超参与 PSC-UNet V3 对齐，便于公平对比
