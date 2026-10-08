# Swin-UNet 训练实验

在 Synapse 2D 切片上训练 Swin-UNet（Swin Transformer Block 替换 UNet 卷积块）。

## 模型

|              |            |
| ------------ | ---------- |
| **Swin-UNet** | **~42 M** |

## 运行

每次训练需唯一的 run 名称：可在 `config.toml` 里改 `[experiments].name`，或用 CLI 覆盖（不改 config 文件）：

```bash
# 训练
python experiments/segment/project1_2609/Swin-UNet/train_swin_unet.py

# 指定配置
python experiments/segment/project1_2609/Swin-UNet/train_swin_unet.py --config experiments/segment/project1_2609/Swin-UNet/config.toml

# 沿用 config，指定 run 名；目录已存在则清空重建（--no-overwrite 可禁止覆盖）
python experiments/segment/project1_2609/Swin-UNet/train_swin_unet.py -n 2026-10-08_002_Swin-UNet_V2

# 不写 -n：若 config 里的 name 已被占用，自动递增为 _V2、_V3… 并创建新 run
python experiments/segment/project1_2609/Swin-UNet/train_swin_unet.py

# 快速试跑 1 个 epoch
python experiments/segment/project1_2609/Swin-UNet/train_swin_unet.py --epochs 1 --quick

# 推理与评估（Dice / IoU / Precision / Recall / FP% / FN% / HD95）
python experiments/segment/project1_2609/Swin-UNet/analysis_swin_unet.py
python experiments/segment/project1_2609/Swin-UNet/analysis_swin_unet.py --config experiments/segment/project1_2609/Swin-UNet/config.toml
# 指定 run（须与训练实际生成的 runs/<name>/ 一致；未写 -n 训练时会自动 V2/V3…）
python experiments/segment/project1_2609/Swin-UNet/analysis_swin_unet.py --name 2026-10-06_001_Swin-UNet_V1
python experiments/segment/project1_2609/Swin-UNet/analysis_swin_unet.py --name latest
python experiments/segment/project1_2609/Swin-UNet/analysis_swin_unet.py --split test
```

## 输出

```
Swin-UNet/runs/2026-10-06_001_Swin-UNet_V1/
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
- **8 器官**：`class_nums=9`，`label_map=eval8`，训练/预测均为 9 类
- **类映射**（模型 ID → 器官）：`1 spleen, 2 rk, 3 lk, 4 gb, 5 liver, 6 stomach, 7 aorta, 8 pancreas`
- 损失函数：**加权 Dice Loss + Cross Entropy**
- 优化器默认 **Adam + lr=1e-4 + CosineAnnealingLR**；`best.pth` 按 `val_eval_dice` 保存
- **Early stopping**：`early_stopping_patience=80`
- `image_size` 需能被 **32** 整除，且 `image_size/patch_size` 能被 **window_size**（默认 7）整除，默认 224
- 灰度输入默认 `in_channels=1`；若设 `repeat_gray_to_rgb=true`，会自动 repeat 为 3 通道
- 每 `class_metrics_every` 个 epoch 记录各器官 Dice 到 `class_metrics.csv`
