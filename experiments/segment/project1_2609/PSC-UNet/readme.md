# PSC-UNet 训练实验

在 Synapse 2D 切片上训练 PSC-UNet（并行 ResNet-34 + Swin + DFCM/WBEM）。

## 运行

每次训练前在 `config.toml` 中设置唯一的 `[experiments].name`：

```bash
# 训练
python experiments/segment/project1_2609/PSC-UNet/train_psc_unet.py

# 指定配置
python experiments/segment/project1_2609/PSC-UNet/train_psc_unet.py --config experiments/segment/project1_2609/PSC-UNet/config.toml

# 快速试跑 1 个 epoch
python experiments/segment/project1_2609/PSC-UNet/train_psc_unet.py --epochs 1 --quick

# 推理与评估（Dice / IoU / Precision / Recall / FP% / FN% / HD95）
python experiments/segment/project1_2609/PSC-UNet/analysis_psc_unet.py
python experiments/segment/project1_2609/PSC-UNet/analysis_psc_unet.py --split test
```

## 输出

```
PSC-UNet/runs/2026-09-23_001_PSC-UNet/
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
        ├── visualizations/   # 含逐器官指标叠加
        ├── slice_metrics.csv
        ├── organ_slice_metrics.csv
        ├── organ_metrics.csv
        └── summary.json
```

## 说明

- 数据目录：`data/synapse_processed`
- 损失函数：多类 Dice Loss（忽略背景类）
- `image_size` 需能被 **32** 整除，且 `image_size/4` 能被 **7**（Swin window_size）整除，默认 224
- 灰度输入默认 `in_channels=1`；若设 `repeat_gray_to_rgb=true`，会自动 repeat 为 3 通道
- 每 `class_metrics_every` 个 epoch 记录各器官 Dice 到 `class_metrics.csv`
