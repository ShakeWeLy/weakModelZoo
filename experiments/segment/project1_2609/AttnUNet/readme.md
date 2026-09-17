# AttnUNet 实验

基于 `AttentionGateUnet` 在 Synapse 腹部多器官分割数据集上的训练与评估。

## 运行

```bash
# 训练
python experiments/segment/project1_2609/AttnUNet/train_attn_unet.py

# 推理与评估
python experiments/segment/project1_2609/AttnUNet/analysis_attn_unet.py
```

## 输出

```
AttnUNet/outputs/
├── best_attn_unet.pth
├── train_log.csv
└── analysis/
    ├── val_summary.json
    ├── test_summary.json
    ├── paper_metrics.csv
    └── val/visualizations/
```

## 指标

评估指标：Dice 系数、HD95 距离（8 个器官：脾脏、左右肾、胆囊、肝脏、胃、主动脉、胰腺）。
