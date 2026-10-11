# weakModelZoo

医学图像分割模型集合, 数据集为 weakUnet_Synapse (Synapse 多器官, 9 类)。

## 目录结构

```
weakModelZoo/
├── data/                       # 数据 (不入库)
│   ├── weakUnet_Synapse/
│   │   ├── train_npz/          # 2D 训练切片 (*.npz, keys: image, label)
│   │   └── test_vol_h5/        # 3D 测试体数据 (*.npy.h5)
│   └── weakUnet_Synapse_lists/
│       ├── train.txt           # 2211 个训练切片名
│       └── test_vol.txt        # 12 个测试病例
├── Swin-Unet/                  # Swin-Unet (官方代码 + Synapse 适配)
├── TransUNet/                  # TransUNet (官方代码 + Synapse 适配)
├── experiments/segment/project1_2610/   # 统一分割训练框架
└── src/models/segment/         # 各模型实现
```

数据标签: 0=背景, 1=脾脏, 2=右肾, 3=左肾, 4=胆囊, 5=肝脏, 6=胃, 7=主动脉, 8=胰腺。

## 环境

统一使用 conda base 环境, 解释器路径:

```
F:\Pydata\anacondai\python.exe
```

主要依赖: torch 2.6.0 (CUDA), tensorboardX, medpy, SimpleITK, h5py, ml_collections, scipy, tqdm。

## Swin-Unet

目录: `Swin-Unet/`

训练 (cmd, 在 `Swin-Unet` 目录下执行):

```bat
F:\Pydata\anacondai\python.exe train.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --max_epochs 150 --batch_size 24 --base_lr 0.05 --output_dir ./model_out
```

- 数据路径默认指向 `../data/weakUnet_Synapse`, 无需额外指定。
- 预训练权重: `Swin-Unet/pretrained_ckpt/swin_tiny_patch4_window7_224.pth` (本地保存, 不入库)。
- 输出: `./model_out/`。

测试 (`test.py`) 默认在 `test_vol` 的 12 个病例上评估 (3D 体积级 Dice / HD95)。

已记录结果 (`best_model.pth`, 来自 epoch 116, 12 例测试集):

| 指标 | 数值 |
|---|---|
| 平均 Dice | 0.6814 |
| 平均 HD95 | 36.44 px |

各器官 Dice: 脾脏 0.7860, 右肾 0.5900, 左肾 0.7343, 胆囊 0.6491, 肝脏 0.9193, 胃 0.3902, 主动脉 0.8340, 胰腺 0.5486。

## TransUNet

目录: `TransUNet/` (官方代码, 已适配本仓库数据与权重)

预训练权重: 需先放置 `TransUNet/pretrained_ckpt/R50+ViT-B_16.npz`, 详见 `TransUNet/pretrained_ckpt/README.md` (MD5 `fb93b71c850df1b0390448d4bafc4262`)。权重文件不入库。

训练 (cmd, 在 `TransUNet` 目录下执行):

```bat
F:\Pydata\anacondai\python.exe train.py --dataset Synapse --vit_name R50-ViT-B_16 --max_epochs 150 --batch_size 2
```

测试 (参数须与训练一致, 否则找不到 checkpoint):

```bat
F:\Pydata\anacondai\python.exe test.py --dataset Synapse --vit_name R50-ViT-B_16 --max_epochs 150 --batch_size 2
```

- 本机为 RTX 3060 Laptop 6GB, 已验证 batch 2 可运行 (占用约 3.2GB); 远程服务器可按显存调大。
- 训练过程只输出训练 loss, 没有验证集; 结果由 `test.py` 在 12 例 `test_vol` 上给出。
- 输出: `TransUNet/model_out/` (不入库)。

## 注意事项

- 权重文件 (`*.pth`, `*.bin`, `*.npz`) 和训练产物 (`model_out/`, `test_log/`) 均不入库, 已在 `.gitignore` 中排除。
- Windows 下已修复的问题: DataLoader 的 `worker_init_fn` 需为模块级函数; npz 键名需用 `/` 分隔; 测试脚本的路径分隔符处理。
- Swin-Unet 与 TransUNet 的评估口径为 3D 体积级, 与 `project1_2610` 的 2D 切片级池化口径不同, 数值不可直接比较。
