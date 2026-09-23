# AttnUNet 实验

基于 `AttentionGateUnet` 在 Synapse 腹部多器官分割数据集上的训练与评估。

## 运行

每次训练前在 `config.toml` 中设置唯一的 `[experiments].name`：

```bash
# 训练
python experiments/segment/project1_2609/AttnUNet/train_attn_unet.py

# 推理与评估（读取同名实验目录）
python experiments/segment/project1_2609/AttnUNet/analysis_attn_unet.py
```

## 输出

```
AttnUNet/runs/2026-09-23_001_AttnUNet/
├── config.yaml
├── history.csv
├── summary.json
├── train.log
├── paper_metrics.csv
├── checkpoints/
│   ├── best.pth
│   └── last.pth
└── predictions/
    ├── val/
    │   ├── predictions/
    │   ├── visualizations/
    │   ├── slice_metrics.csv
    │   └── summary.json
    └── test/
        └── ...
```

## 指标

评估指标：Dice 系数、HD95 距离（8 个器官：脾脏、左右肾、胆囊、肝脏、胃、主动脉、胰腺）。
