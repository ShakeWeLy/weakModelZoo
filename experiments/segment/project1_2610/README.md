# project1_2610：Synapse 多器官分割（统一训练框架）

由 project1_2609 重构而来：每个模型不再各自维护一套 train / analysis 脚本，而是共用 `trainer/` 中的同一套流程，模型差异只体现在 `configs/models/<model>.toml`。

## 目录

```text
project1_2610/
├── train.py                 训练入口
├── analyze.py               预测分析入口
├── configs/
│   ├── global.toml          全局配置：数据、默认超参、训练、测试、[quick]
│   └── models/<model>.toml  模型配置：[model] + [model.params]，可覆盖全局任意键
├── trainer/
│   ├── config.py            配置加载 / 深度合并 / --set 覆盖
│   ├── recorder.py          run 目录、文本日志、TensorBoard、预测 CSV、checkpoint、summary.json
│   ├── report.py            summary.md / paper_metrics
│   ├── data.py              weakUnet_Synapse（train_npz + test_vol_h5）
│   ├── labels.py            标签 0..8（背景 + 8 器官）
│   ├── models.py            模型注册表
│   ├── losses.py / optim.py / metrics.py
│   ├── trainer.py           Trainer
│   ├── predictor.py         Predictor（推理 + 指标 + 汇总）
│   └── visualize.py
└── <Model>/runs/<name>/     结果（布局同 2609）
```

## 配置合并顺序

`global.toml` → `models/<model>.toml` → `global.[quick]` → `model.[quick]`（仅 `--quick`）→ `--epochs` / `--set`

合并后的完整配置保存在 `runs/<name>/config.yaml`。

## 用法（项目根目录）

```bash
# 训练
python experiments/segment/project1_2610/train.py -m swin_unet
python experiments/segment/project1_2610/train.py -m unet --epochs 100 --set hyperparameters.learning_rate=3e-4
python experiments/segment/project1_2610/train.py -m swin_unet --quick -n _smoke --analyze   # 冒烟测试

# 预测分析（默认 best checkpoint，split 取 test.splits）
python experiments/segment/project1_2610/analyze.py -m swin_unet -n 2026-10-08_001_Swin-UNet_V1
python experiments/segment/project1_2610/analyze.py -m swin_unet --split val --checkpoint last --no-vis

# 查看训练曲线（TensorBoard）
tensorboard --logdir experiments/segment/project1_2610/UNet/runs/2026-10-08_001_UNet_V1/tensorboard
```

### 配置开关

`configs/global.toml` 与 `configs/models/*.toml` 中的布尔键即开关（`true` 开 / `false` 关），每个开关在文件中都有注释。常用开关：

| 开关 | 位置 | 作用 |
|---|---|---|
| `augment` | `[dataset]` | 训练集数据增强 |
| `use_class_weights` | `[train]` | 类别权重（关闭时所有类等权，忽略 `dice_class_weights` / `ce_class_weights`） |
| `skip_absent_classes` | `[train]` | Dice 是否跳过 batch 中缺失的类 |
| `guaranteed_sampling` | `[train]` | 保底采样（保证指定类别切片每个 epoch 出现） |
| `early_stopping` | `[train]` | 早停（关闭后训练满 `num_epochs`，`early_stopping_patience` 仅在开启时生效） |
| `amp` | `[train]` | 混合精度（仅 CUDA） |
| `repeat_gray_to_rgb` | `[model]` | 灰度输入是否复制为 3 通道 |
| `save_predictions` / `save_visualizations` | `[test]` | 是否保存预测掩码 / 可视化 |

- `-n` 指定 run 名：已存在时清空重建（`--no-overwrite` 改为报错）；不指定时使用 `experiments.name`，已占用则自动递增 `_V2`、`_V3`…
- `analyze` 的 dataset / model 配置来自 run 的 `config.yaml`，`[test]` 段来自当前 configs，改可视化数量等无需重训。

可选模型：`unet`、`attn_unet`、`transunet`、`swin_unet`、`psc_unet`、`vm_unet`。新增模型：在 `trainer/models.py` 的 `MODEL_REGISTRY` 加一项，再新建对应 toml。

## 结果目录

```text
<Model>/runs/<name>/
├── config.yaml  train.log  analyze.log
├── tensorboard/           标量 loss/*、dice/*、lr、time/*、class_dice/<split>/<类>；config 文本；hparams
├── summary.json  summary.md  paper_metrics.csv   （summary.json 含 last_epoch 最后一轮指标）
├── checkpoints/{best,last}.pth
└── predictions/<split>/{predictions/*.npy, visualizations/*.png,
                         slice_metrics.csv, organ_slice_metrics.csv, organ_metrics.csv, summary.json}
```

## 与 2609 的差异

- 训练期 `train_dice` / `val_dice` 为 **epoch 内累积** 的全局 Dice（`metric_class_ids` 上取均值），2609 为逐 batch Dice 的平均，数值不可直接对比；论文指标以 `analyze` 结果为准（两者口径一致）。
- 训练期逐类 Dice 直接取自当轮训练 / 验证的累积结果，不再额外遍历数据集。
- 只使用 `data/weakUnet_Synapse`（+ `data/weakUnet_Synapse_lists`），不再支持 synapse_processed / 14 类标签。
- analyze 的 train / val 与训练使用同一病例划分（`val_holdout_cases`、`split_seed`）；2609 中 analyze 的 train 包含了验证病例。
- `guaranteed_class_groups` 使用 0..8 标签（4=胆囊，8=胰腺）。
- HD95 单位为像素（224×224 分辨率）。
