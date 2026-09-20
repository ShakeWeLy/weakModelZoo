# PSC-UNet 训练实验

在 Synapse 2D 切片上训练 PSC-UNet（并行 ResNet-34 + Swin + DFCM/WBEM）。

## 运行

```bash
# 项目根目录
python experiments/segment/project1_2609/PSC-UNet/train_psc_unet.py

# 指定配置
python experiments/segment/project1_2609/PSC-UNet/train_psc_unet.py --config experiments/segment/project1_2609/PSC-UNet/config.toml

# 快速试跑 1 个 epoch
python experiments/segment/project1_2609/PSC-UNet/train_psc_unet.py --epochs 1 --quick
```

## 说明

- 数据目录：`data/synapse_processed`
- 损失函数：多类 Dice Loss（忽略背景类）
- `image_size` 需能被 **32** 整除，且 `image_size/4` 能被 **7**（Swin window_size）整除，默认 224
- 灰度输入默认 `in_channels=1`；若设 `repeat_gray_to_rgb=true`，会自动 repeat 为 3 通道
- 输出：`experiments/segment/project1_2609/PSC-UNet/outputs/`
