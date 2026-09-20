# VM-UNet 训练实验

在 Synapse 2D 切片上训练 VM-UNet（VSSBlock 版 U-Net）。

## 运行

```bash
# 项目根目录
python experiments/segment/project1_2609/VM-UNet/train_vm_unet.py

# 指定配置 / 只跑 1 个 epoch
python experiments/segment/project1_2609/VM-UNet/train_vm_unet.py --epochs 1

# 无 mamba_ssm 时快速试跑（image_size=64）
python experiments/segment/project1_2609/VM-UNet/train_vm_unet.py --epochs 1 --quick
```

## 说明

- 数据目录：`data/synapse_processed`
- 损失函数：多类 Dice Loss（忽略背景类）
- `image_size` 需能被 32 整除（默认 224）
- 输出：`experiments/segment/project1_2609/VM-UNet/outputs/`
- 建议安装 `mamba_ssm` 后再用 `image_size=224` 正式训练；未安装时请加 `--quick`
