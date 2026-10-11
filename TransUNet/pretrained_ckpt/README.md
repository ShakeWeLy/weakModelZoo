# 预训练权重（未入库，需手动放置）

本目录需要放置 ImageNet-21k 预训练的 R50+ViT-B/16 权重，用于 `train.py` 初始化编码器。权重文件体积较大（约 440 MiB），不纳入 Git 仓库。

## 文件信息

- 文件名：`R50+ViT-B_16.npz`
- 大小：461,217,452 字节
- MD5：`fb93b71c850df1b0390448d4bafc4262`

## 获取方式

1. 官方 README 中的 ViT 预训练下载链接已失效，可使用 README 提供的 Google Drive 备份，解压后找到 `R50+ViT-B_16.npz`。
2. 若本机已有副本（例如 `weakUnet/model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz`），直接复制即可。

## 放置位置

将文件放到本目录下，即：

```
TransUNet/pretrained_ckpt/R50+ViT-B_16.npz
```

## 校验

Linux / macOS：

```bash
md5sum "R50+ViT-B_16.npz"
```

Windows：

```bash
certutil -hashfile "R50+ViT-B_16.npz" MD5
```

输出的 MD5 应为 `fb93b71c850df1b0390448d4bafc4262`。

## 说明

- 若文件缺失，`train.py` 会报错 `未找到预训练权重`。
- 也可以通过 `--pretrained_path` 指定其他位置的 `.npz` 文件。
