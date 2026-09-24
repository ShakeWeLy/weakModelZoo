# PSC-UNet 训练实验

在 Synapse 2D 切片上训练 PSC-UNet（并行 ResNet-34 + Swin + DFCM/WBEM）。

## 模型


|              |            |
| ------------ | ---------- |
| **PSC-UNet** | **8.75 M** |


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

## 版本

| 配置 | 说明 |
| ---- | ---- |
| `config.toml`（V3） | **仅 8 器官**：`class_nums=9`，`label_map=eval8`，训练/预测均为 9 类 |
| `config_v2.toml`（V2） | 14 类训练，best 按 8 器官 `val_eval_dice` 选模型 |

## 说明

- 数据目录：`data/synapse_processed`
- **V3**：原始 GT 中非 8 器官（食管/IVC/门静脉/肾上腺等）在训练时映射为背景；模型输出 9 类
- **V3 类映射**（模型 ID → 器官）：`1 spleen, 2 rk, 3 lk, 4 gb, 5 liver, 6 stomach, 7 aorta, 8 pancreas`
- 损失函数：**加权 Dice Loss**（`dice_class_weights`）
- 优化器默认 **Adam + lr=1e-4**；`best.pth` 按 `val_eval_dice` 保存
- **Early stopping**：`early_stopping_patience=50`
- V3 下 `val_dice` 与 `val_eval_dice` 均为 8 器官均值
- `image_size` 需能被 **32** 整除，且 `image_size/4` 能被 **7**（Swin window_size）整除，默认 224
- 灰度输入默认 `in_channels=1`；若设 `repeat_gray_to_rgb=true`，会自动 repeat 为 3 通道
- 每 `class_metrics_every` 个 epoch 记录全部 13 个前景类 Dice 到 `class_metrics.csv`

