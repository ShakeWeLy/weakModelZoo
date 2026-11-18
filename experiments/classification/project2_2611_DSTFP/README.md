# project2_2611_DSTFP：分类实验脚手架（DSTFP）

面向 **DSTFP**（voxel-level dual-modality Swin Transformer feature fusion）的分类实验目录。  
结构对齐 `experiments/classification/project1_ADNI`、`experiments/segment/project1_2610`、`experiments/detection/project2_2611`，并采用本项目约定的 **命名 / 结果路径** 规则。

> 当前状态：**初始化占位**。上游 `DSTFP/module/models/`（Swin4D / Swin3D / fusion）缺失，训练与评估入口会明确报 `NotImplementedError`，但配置、run 目录与命名规则可直接使用。

---

## 目录结构

```text
project2_2611_DSTFP/
├── README.md                      本说明（结构 + 约定）
├── train.py                       训练入口（占位：可 --init-only 建 run）
├── analyze.py                     评估入口（占位）
├── notes/
│   └── TODO.md                    接通真实训练的待办
├── configs/
│   ├── global.toml                全局：数据 / 超参 / 训练 / 测试 / [quick]
│   └── models/
│       ├── dstfp.toml             双模态 DSTFP（主实验）
│       ├── swin4d_fmri.toml       单模态 fMRI baseline（占位）
│       └── swin3d_fa.toml         单模态 FA/DTI baseline（占位）
├── trainer/
│   ├── config.py                  配置合并；model_output_dir = runs/<display_name>/
│   ├── recorder.py                run 目录、summary.json、_V 递增
│   ├── models.py                  模型注册表（指向 DSTFP/module/models，占位）
│   ├── data.py / trainer.py       数据与训练循环（占位）
│   ├── optim.py / metrics.py      优化器 / 指标键（占位）
│   └── __init__.py
└── runs/                          ★ 结果根目录（新布局）
    ├── DSTFP/
    │   └── 001_DSTFP_V1/          示例占位 run
    │       ├── summary.json
    │       ├── checkpoints/
    │       └── tensorboard/
    ├── swin4d_fmri/
    └── swin3d_fa/
```

---

## 与旧项目对照

| 项 | 旧约定（segment / detection / ADNI） | 本项目新约定 |
| --- | --- | --- |
| run `name` | `2026-10-08_001_PSC-UNet_V1`（日期在前） | `001_DSTFP_V1`（**序号 + 标签 + 版本**） |
| 日期放哪 | 写在 `name` 开头 | 写在 `description` 开头：`2026-11-12 \| ...` |
| 结果路径 | `<display_name>/runs/<name>/` | `runs/<display_name>/<name>/`（**runs 与模型名对调**） |
| 同实验复跑 | `..._V1` → `..._V2` | 不变：仍在同一 `runs/<display_name>/` 下递增 `_Vn` |

### 1) `name` 不再带日期

参考旧写法（`psc_unet.toml`）::

```toml
# 旧
name = "2026-10-08_001_PSC-UNet_V1"
description = "PSC-UNet | Dice + CE | Adam 1e-4 + cosine"
```

本项目写法::

```toml
# 新
name = "001_DSTFP_V1"
description = "2026-11-12 | DSTFP multimodal | ... | placeholder"
```

`config.py` 会在缺省时自动：

- 填 `experiments.date`（当天）
- 若 `description` 未以该日期开头，则前置 `YYYY-MM-DD | `

### 2) 结果目录：`runs` 与模型名对调

旧 detection 例子::

```text
project2_2611/SGC_YOLOv5_c3topo_2/runs/2026-11-12_022_SGC_c3topo_2_s_V1/
```

本项目目标形态（`runs/xxx/xxx_V1`、`xxx_V2`）::

```text
project2_2611_DSTFP/runs/DSTFP/001_DSTFP_V1/
project2_2611_DSTFP/runs/DSTFP/001_DSTFP_V2/   # 同配置复跑自动递增
```

对应代码：`trainer/config.py::model_output_dir` → `<project>/runs/<display_name>/`。

### 单次 run 内预期文件（接通训练后）

```text
runs/<display_name>/<name>/
├── config.yaml              合并后的完整配置
├── train.log
├── summary.json / summary.md
├── checkpoints/{best,last}.pth
└── tensorboard/
```

---

## 配置合并顺序

`global.toml` → `models/<model>.toml` → `global.[quick]` → `model.[quick]`（仅 `--quick`）→ `--epochs` / `--set`

布尔键即开关（`true`/`false`）。合并后的配置将写入 `runs/.../config.yaml`（训练接通后）。

---

## DSTFP 内容占位说明

| 实验配置 | arch | display_name | 上游对应（缺失实现） |
| --- | --- | --- | --- |
| `dstfp` | `dstfp` | `DSTFP` | `LitMultimodalClassifier` + `swin4d_ver7` + `swin3d_transformer` + `optimized_fusion` |
| `swin4d_fmri` | `swin4d_fmri` | `swin4d_fmri` | `LitClassifier` + `SwinTransformer4D` |
| `swin3d_fa` | `swin3d_fa` | `swin3d_fa` | `LitFA3DClassifier` + `SwinTransformer3D` |

源码线索：

- Lightning 组装：`DSTFP/module/pl_classifier.py`
- 训练入口参考：`DSTFP/module/main.py`（单模态）、`DSTFP/module/multi_main.py`（双模态）
- 数据：`DSTFP/module/utils/data_module.py`
- **网络实现目录 `DSTFP/module/models/` 当前不在仓库中**

---

## 用法

在仓库根目录执行：

```bash
# 只初始化 run 目录（推荐当前阶段）
python experiments/classification/project2_2611_DSTFP/train.py -m dstfp --init-only

# 尝试进入训练（会因占位 NotImplementedError 中止，但仍会建好 run）
python experiments/classification/project2_2611_DSTFP/train.py -m dstfp
python experiments/classification/project2_2611_DSTFP/train.py -m dstfp --quick
python experiments/classification/project2_2611_DSTFP/train.py -m swin4d_fmri --init-only
python experiments/classification/project2_2611_DSTFP/train.py -m swin3d_fa --init-only

# 指定 run 名 / 覆盖配置
python experiments/classification/project2_2611_DSTFP/train.py -m dstfp -n 001_DSTFP_V1 --init-only
python experiments/classification/project2_2611_DSTFP/train.py -m dstfp --set dataset.batch_size=2 --set train.device=cpu
```

命名规则：

- `-n` 指定 name：已存在时默认清空重建；`--no-overwrite` 改为报错
- 不指定 `-n`：使用 toml 里的 `experiments.name`；若目录已占用则自动 `_V2`、`_V3`…

---

## 与其它 experiments 项目的关系

```text
experiments/
├── classification/
│   ├── project1_ADNI/          # DFSC 等；旧布局 <Model>/runs/<name>
│   └── project2_2611_DSTFP/    # 本项目；新布局 runs/<Model>/<name>
├── segment/project1_2610/      # 分割统一框架；旧布局 + 旧 name（含日期）
└── detection/project2_2611/    # 检测；旧布局 <Model>/runs/<date>_<seq>_..._Vn
```

本 README 即以后新建分类实验时的默认模板约定：

1. **name** = `序号_标签_Vn`
2. **日期** → `description` 开头
3. **结果** = `runs/<display_name>/<name>/`
