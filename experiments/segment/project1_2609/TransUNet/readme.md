# TransUNet 训练实验

在 Synapse 2D 切片上训练 TransUNet（UNet 编码器 + Transformer×12 瓶颈 + UNet 解码器），超参与 `UNet_V7` 对齐，便于公平对比。

## 运行

每次训练前在 `config.toml` 中设置唯一的 `[experiments].name`：

```bash
# 训练
python experiments/segment/project1_2609/TransUNet/train_transunet.py

# 指定配置
python experiments/segment/project1_2609/TransUNet/train_transunet.py --config experiments/segment/project1_2609/TransUNet/config.toml

# 快速试跑 1 个 epoch
python experiments/segment/project1_2609/TransUNet/train_transunet.py --epochs 1 --quick

# 推理与评估（Dice / IoU / Precision / Recall / FP% / FN% / HD95）
python experiments/segment/project1_2609/TransUNet/analysis_transunet.py
python experiments/segment/project1_2609/TransUNet/analysis_transunet.py --split test
```

## 输出

```
TransUNet/runs/2026-09-29_001_TransUNet_V1/
├── config.yaml
├── history.csv
├── class_metrics.csv
├── summary.json
├── train.log
├── paper_metrics.csv
├── checkpoints/
│   ├── best.pth
│   └── last.pth
├── summary.md
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
- 损失函数：**Dice + CE**（`dice_loss_weight=1.0`, `ce_loss_weight=1.0`）
- `skip_absent_classes=false`，与 UNet_V7 一致
- 优化器默认 **Adam + lr=1e-4**；`best.pth` 按 `val_eval_dice` 保存
- **Early stopping**：`early_stopping_patience=50`
- `image_size=224` 同时用于数据 resize 与 Transformer 位置编码初始化
