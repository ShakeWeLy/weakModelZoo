# 预训练权重目录（不入库）

统一存放本项目用到的官方预训练权重。`*.pth` / `*.pt` 已在根目录 `.gitignore` 中忽略；本 README 与下载脚本保留入库。

## 目录结构

```text
pretrained_weights/
├── README.md
├── download_sam3.py
├── resnet/
│   ├── resnet18-f37072fd.pth      # 已从本机 torch hub 复制
│   └── resnet34-b627a593.pth
├── swin/
│   └── swin_tiny_patch4_window7_224.pth   # Swin-UNet 官方 ImageNet 预训练
├── transunet/
│   └── R50+ViT-B_16.npz                   # TransUNet 官方 ImageNet-21k 预训练
└── sam3/                          # ModelScope cache_dir
    └── facebook/sam3/
        └── sam3.pt
```

## ResNet

已从本机 `F:/models/torch/hub/checkpoints/` 复制。也可手动：

```bash
cp /path/to/resnet34-b627a593.pth pretrained_weights/resnet/
```

官方：https://download.pytorch.org/models/resnet34-b627a593.pth

## SAM3（ModelScope）

**正确 id：`facebook/sam3`**（`AI-ModelScope/sam3` 在 ModelScope 上不存在，会 404）。

```bash
# 仓库根目录执行（只下 sam3.pt 等必要文件，约 3.2GB）
python pretrained_weights/download_sam3.py
```

等价代码：

```python
from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download(
    "facebook/sam3",
    cache_dir="./pretrained_weights/sam3",
    allow_file_pattern=["sam3.pt", "config.json", "configuration.json", "README.md"],
)
```

若仓库需授权，先在 ModelScope / Hugging Face 申请访问并登录。

## Swin-UNet（官方 ImageNet Swin-Tiny）

已从 `Swin-Unet/pretrained_ckpt/swin_tiny_patch4_window7_224.pth` 复制。也可手动：

```bash
mkdir -p pretrained_weights/swin
cp /path/to/swin_tiny_patch4_window7_224.pth pretrained_weights/swin/
```

`experiments/segment/project1_2610/configs/models/swin_unet_official.toml` 默认 `load_pretrained=true`，读取上述路径。

## TransUNet（官方 ImageNet-21k R50+ViT-B/16）

已从 `TransUNet/pretrained_ckpt/R50+ViT-B_16.npz` 复制（约 440 MiB，MD5 `fb93b71c850df1b0390448d4bafc4262`）。也可手动：

```bash
mkdir -p pretrained_weights/transunet
cp /path/to/R50+ViT-B_16.npz pretrained_weights/transunet/
```

`experiments/segment/project1_2610/configs/models/transunet_official.toml` 默认 `load_pretrained=true`，读取上述路径。

## SLR-UNet（缺失自动下载）

`experiments/segment/project1_2610/configs/models/slr_unet.toml`：

- `resnet_weights`：本地没有时，`resnet_pretrained=true` 会走 torchvision 下载并尽量保存到该路径
- `sam_checkpoint`：本地没有且 `sam_auto_download=true` 时，自动 ModelScope 下载 `facebook/sam3` 的 `sam3.pt`
