# weakModelZoo

医学图像分割模型集合, 数据集为 weakUnet_Synapse (Synapse 多器官, 9 类)。

## 目录结构

```
weakModelZoo/
├── data/                               # 数据 (不入库)
│   ├── weakUnet_Synapse/
│   │   ├── train/                      # 2D 训练切片 (*.npz, keys: image, label)
│   │   └── test_vol_h5/                # 3D 测试体数据 (*.npy.h5)
│   └── weakUnet_Synapse_lists/
│       ├── train.txt                   # 2211 个训练切片名
│       └── test_vol.txt                # 12 个测试病例
├── Swin-Unet/                          # Swin-Unet (官方代码 + Synapse 适配)
├── TransUNet/                          # TransUNet (官方代码 + Synapse 适配)
├── MT-UNet/                            # MT-UNet (官方代码 + Synapse 适配)
├── experiments/segment/project1_2610/  # 统一分割训练框架
└── src/models/segment/                 # 各模型实现
```

数据标签: 0=背景, 1=脾脏, 2=右肾, 3=左肾, 4=胆囊, 5=肝脏, 6=胃, 7=主动脉, 8=胰腺。

## 环境

统一使用 conda base 环境, 在已激活 base 的终端中直接使用 `python`:

```bat
conda activate base
python --version
```

主要依赖: torch 2.6.0 (CUDA), tensorboardX, medpy, SimpleITK, h5py, ml_collections, scipy, tqdm。

## 命令参考

以下命令均在对应模型目录下执行。

| 模型 | 操作 | 目录 | 命令 |
| --- | --- | --- | --- |
| Swin-Unet | 训练 | `Swin-Unet/` | `python train.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --max_epochs 150 --batch_size 24 --base_lr 0.05 --output_dir ./model_out` |
| Swin-Unet | 测试 | `Swin-Unet/` | `python test.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --max_epochs 150 --batch_size 24 --base_lr 0.05 --output_dir ./model_out` |
| TransUNet | 训练 | `TransUNet/` | `python train.py --dataset Synapse --vit_name R50-ViT-B_16 --max_epochs 150 --batch_size 16` |
| TransUNet | 测试 | `TransUNet/` | `python test.py --dataset Synapse --vit_name R50-ViT-B_16 --max_epochs 150 --batch_size 16` |
| MT-UNet | 训练 (Synapse) | `MT-UNet/` | `python train_mtunet_Synapse.py --batch_size 12 --max_epochs 100` |
| MT-UNet | 训练 (Synapse, 加载已有权重) | `MT-UNet/` | `python train_mtunet_Synapse.py --batch_size 12 --max_epochs 100 --checkpoint pretrained_ckpt/mtunet_synapse.pth` |

说明:
- TransUNet 的测试参数 (`--vit_name`, `--max_epochs`, `--batch_size`) 必须与训练一致, 否则 `test.py` 拼出的 checkpoint 目录名不同, 找不到权重。
- Swin-Unet 测试会在 `test_vol` 的 12 个病例上计算 3D 体积级指标。
- MT-UNet 训练过程中每 `--n-skip` 个 epoch 在测试集上评估一次, 最优权重保存在 `MT-UNet/checkpoint/Synapse/mtunet/`, 没有单独的测试命令。
- MT-UNet 的 ACDC 训练脚本 (`train_mtunet_ACDC.py`) 指向 `../data/ACDC`, 该数据尚未按其格式准备 (缺少 `lists_ACDC` 与切片), 暂不可用。

## Swin-Unet

目录: `Swin-Unet/`

- 数据路径默认指向 `../data/weakUnet_Synapse`, 无需额外指定。
- 预训练权重: `Swin-Unet/pretrained_ckpt/swin_tiny_patch4_window7_224.pth` (本地保存, 不入库)。
- 输出: `./model_out/`。

测试 (`test.py`) 默认在 `test_vol` 的 12 个病例上评估 (3D 体积级 Dice / HD95)。

已记录结果 (`best_model.pth`, 来自 epoch 116, 12 例测试集):

| 指标      | 数值       |
| ------- | -------- |
| 平均 Dice | 0.6814   |
| 平均 HD95 | 36.44 px |

各器官 Dice: 脾脏 0.7860, 右肾 0.5900, 左肾 0.7343, 胆囊 0.6491, 肝脏 0.9193, 胃 0.3902, 主动脉 0.8340, 胰腺 0.5486。

## TransUNet

目录: `TransUNet/` (官方代码, 已适配本仓库数据与权重)

预训练权重: 需先放置 `TransUNet/pretrained_ckpt/R50+ViT-B_16.npz`, 详见 `TransUNet/pretrained_ckpt/README.md` (MD5 `fb93b71c850df1b0390448d4bafc4262`)。权重文件不入库。

- 本机为 RTX 3060 Laptop 6GB, 已验证 batch 2 可运行 (占用约 3.2GB); 远程服务器可按显存调大。
- 训练过程只输出训练 loss, 没有验证集; 结果由 `test.py` 在 12 例 `test_vol` 上给出。
- 输出: `TransUNet/model_out/` (不入库)。

## MT-UNet

目录: `MT-UNet/` (官方代码, 已适配本仓库 weakUnet_Synapse 数据)

- 从零训练, 不依赖 ImageNet 预训练; `pretrained_ckpt/` 为可选权重占位, 详见 `MT-UNet/pretrained_ckpt/README.md`。
- Synapse 数据: 切片从 `../data/weakUnet_Synapse/train/` 读取, 体数据从 `../data/weakUnet_Synapse/test_vol_h5/` 读取, 列表使用 `MT-UNet/dataset/lists_Synapse/` (`train.txt`, `test.txt`)。
- 输出: `MT-UNet/checkpoint/` 与 `MT-UNet/predictions/` (不入库)。
- 模型 `model/MTUNet.py` 中部分模块写死了 `.cuda()`, 需在 GPU 环境运行。

## 注意事项

- 权重文件 (`*.pth`, `*.bin`, `*.npz`) 和训练产物 (`model_out/`, `checkpoint/`, `predictions/`, `test_log/`, 日志) 均不入库, 已在 `.gitignore` 中排除。
- 所有路径均相对仓库解析, 不要写绝对路径; 需要覆盖时用命令行参数。
- Windows 下已修复的问题: DataLoader 的 `worker_init_fn` 需为模块级函数; npz 键名需用 `/` 分隔; 测试脚本的路径分隔符处理。
- Swin-Unet 与 TransUNet 的评估口径为 3D 体积级, 与 `project1_2610` 的 2D 切片级池化口径不同, 数值不可直接比较。
