# TransUNet

## 简介

TransUNet 是 2021 年提出的医学图像分割网络 (论文: [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/abs/2102.04306), 2024 年扩展版发表于 Medical Image Analysis)。它的核心思路是: 用 CNN 与 Transformer 分工, 分别负责局部细节与全局上下文。

整体结构是 U 形:

1. **CNN 编码器**: 逐级下采样提取多尺度局部特征, 同时保留各级特征用于跳跃连接。官方实现为 ResNet-50 (R50), 最终下采样到 1/16。
2. **Transformer 瓶颈**: 把最深层特征展平为 token 序列 (如 224×224 输入对应 14×14 = 196 个 token), 经过若干层 ViT 编码器做全局自注意力, 再还原为空间特征。官方 `R50-ViT-B_16` 使用 12 层、768 维的 ViT-B/16。
3. **UNet 解码器**: 逐级上采样, 与编码器对应层的特征拼接, 恢复分辨率并输出逐像素类别。

Transformer 能建模长距离依赖, 弥补纯 CNN 感受野有限的问题; 跳跃连接则保留边界与小目标的细节。在 Synapse 多器官分割数据集上, 论文报告 TransUNet 的平均 Dice 约为 77.5%。

官方代码: [Beckschen/TransUNet](https://github.com/Beckschen/TransUNet), 仓库位于本地根目录 `TransUNet/`。官方预训练权重 (ImageNet-21k 的 `R50+ViT-B_16.npz`) 需要单独下载, 放在 `../model/vit_checkpoint/imagenet21k/` 下。

---

## 对比分析: 官方 TransUNet vs 本仓库实现

### 结论

本仓库的 `src/models/segment/TransUNet` 是"UNet + 12 层 Transformer 瓶颈"的自研变体。它保留了 TransUNet 的核心思路 (CNN 下采样提取局部特征, Transformer 在最深层做全局建模, UNet 解码), 但没有复现官方的几个关键要素: ImageNet 预训练的 R50+ViT-B/16 权重、pre-LN 的 ViT 结构、官方的 SGD + poly 训练配方。因此两边的数字不能直接比较。

### 参数规模 (实测)

使用 base 环境实例化两边模型 (随机初始化, 224×224 输入, 9 类), 仅统计参数量与前向形状, 未训练:

| 模型 | 总参数 | Transformer 部分 | 其余 (编码/解码) |
|---|---|---|---|
| 官方 R50-ViT-B_16 | 105.28M | 97.89M (768 维) | 解码器 7.39M |
| 本仓库 TransUNet (hidden=64, depth=12) | 182.40M | 151.15M (1024 维, FFN 4096) | 31.25M |

本仓库的 Transformer 宽度为 1024, 接近 ViT-L, 参数量约为官方的 1.7 倍。

### 逐项对比

| 维度 | 官方 `TransUNet/` | 本仓库实现 (`src/models/segment/TransUNet` + `project1_2610`) |
|---|---|---|
| CNN 编码器 | ResNetV2 (R50, 3-4-9 units, StdConv + GroupNorm), 预训练 | 4 层 DoubleConv + BN, 从零训练 |
| 预训练 | 加载 ImageNet-21k 的 `R50+ViT-B_16.npz` | 无 |
| Transformer 输入 | 1×1 卷积投影到 768 维, 14×14 = 196 个 token | 1024 通道直接作为 token, 无投影 |
| Transformer 结构 | pre-LN (先 LN 再注意力/FFN), 残差连接 | post-LN: `x = LN(x + MHA(x))`, 无 warmup |
| Dropout | embedding 与 MLP 均为 0.1 | 无 |
| 位置编码 | 可学习, 加载预训练并插值 | 可学习 (trunc_normal 0.02), 支持插值 |
| 输入通道 | 灰度复制为 3 通道 (匹配 ImageNet 预训练) | 1 通道 (`repeat_gray_to_rgb` 可选, 当前未开启) |
| 跳跃连接 | `n_skip=3`, 仅 1/8、1/4、1/2 尺度, 全分辨率无 skip | 4 级 skip, 包括全分辨率 (64 通道) |
| 解码器 | 双线性上采样 + 3×3 卷积, 通道 256/128/64/16, 3×3 分割头 | ConvTranspose + DoubleConv, 通道 512/256/128/64, 1×1 分割头 |
| 损失 | 0.5·CE + 0.5·Dice; Dice 含背景, 平方分母 | 1.0·CE + 1.0·Dice; Dice 不含背景, 线性分母, 可用 `skip_absent_classes` 开关 |
| 优化器 | SGD, lr 0.01, momentum 0.9, poly 衰减 | Adam, lr 1e-4, 无调度器 (与 UNet_V7 对齐) |
| 训练轮数 | 默认 150 epoch, 无验证集, 固定间隔保存 | 300 epoch, 留出 2 个病例做验证, 按 val Dice 保存 best, early stopping |
| 数据增强 | rot90/flip, ±20° 旋转 | 一致 (同源 `RandomGenerator`) |
| 评估口径 | 3D 体积级 Dice / HD95 (逐切片预测拼成体积后用 medpy 计算) | 2D 切片级, `summarize_organs` 池化统计 |

### 影响最大的几点 (按优先级)

1. **没有预训练, 差距最大。** 官方效果的主要来源之一是 ImageNet 预训练的 ViT 与 R50。医学小数据上从零训练 12 层 Transformer 通常明显弱于预训练版本。
2. **Post-LN 从零训练 12 层且无 warmup, 训练可能不稳定。** 官方使用 pre-LN, 深层更稳定。
3. **宽度 1024 偏大。** 参数量是官方的 1.7 倍, 显存与训练时间成本更高。若保留自研结构, 可考虑降到 768 维。
4. **评估口径不同。** 本仓库 `project1_2610` 按 2D 切片统计, 官方是 3D 体积级 DSC / HD95。之前 Swin-Unet 的 68.14% 是 3D 体积级结果, 与这里的切片级指标不能直接比较。
5. **损失定义不同。** Dice 去掉背景、线性分母、权重 1:1, 与官方 (含背景、平方分母、0.5:0.5) 不一致。Adam 对损失尺度不敏感, 因此权重比例的影响相对较小。
6. **命名误导。** `TransformerDecoderBlock` 实际是只含自注意力的 encoder 式块, `TransUNet.py` 文档字符串中也写成了 Decoder, 建议改名。

### 本仓库实现的优点

- 已接入统一框架: TOML 配置、TensorBoard、early stopping、逐类指标以及各类开关齐全。
- 4 级完整跳跃连接, 对胆囊、胰腺等细小器官可能更有利。
- 有验证集与 best checkpoint 选择, 比官方的固定间隔保存更规范。

### 运行官方代码的注意事项

- 官方代码基于 Python 3.7 + torch 1.4, 与当前 torch 2.6 环境存在差异。
- `train.py` / `test.py` 中硬编码了 `F:/myproject/weakUnet/...` 路径, 需改为相对路径。
- 官方 README 称 ViT 预训练链接已失效, 需使用其提供的网盘副本。
