# 预训练权重（可选，未入库，需手动放置）

MT-UNet 从零训练，不依赖 ImageNet 预训练，因此本目录**不是必需的**。仅当你想用官方在 Synapse 上训练好的权重做初始化或推理时，才需要把权重放到这里。

## 文件信息

- 文件名（建议）：`mtunet_synapse.pth`
- 来源：原作者在 Zenodo 发布的 Synapse 权重，<https://zenodo.org/records/18831591>
- 大小、MD5：待补充（下载后自行记录）

## 使用方式

`train_mtunet_Synapse.py` 通过 `--checkpoint` 加载已有权重，默认不加载：

```bash
python train_mtunet_Synapse.py --checkpoint pretrained_ckpt/mtunet_synapse.pth
```

也可以直接指定其他位置的 `.pth` 文件。

## 说明

- `*.pth` 已在 `MT-UNet/.gitignore` 中忽略，权重不会入库。
- 权重结构需与 `model/MTUNet.py` 中的 `MTUNet(out_ch=9)` 一致，否则 `load_state_dict` 会报错。
