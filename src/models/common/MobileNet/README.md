# MobileNet 系列（V1 → V3）

本目录实现 MobileNet V1 / V2 / V3，代码：`mobilenet.py`。目标是在移动端约束下，用更少的乘加运算做 ImageNet 级分类，并可作为检测 / 分割的轻量 backbone。

---

## 总览

| 版本 | 论文 | 核心增量 | 本仓库类 | 默认参数量（实测） |
|------|------|----------|----------|-------------------|
| V1 | [MobileNets (2017)](https://arxiv.org/abs/1704.04861) | 深度可分离卷积 + width multiplier | `MobileNetV1` | ≈4.23M |
| V2 | [MobileNetV2 (2018)](https://arxiv.org/abs/1801.04381) | 倒残差 + 线性瓶颈 | `MobileNetV2` | ≈3.50M |
| V3 | [MobileNetV3 (2019)](https://arxiv.org/abs/1905.02244) | NAS + SE + h-swish + 重新设计的头 | `MobileNetV3` | Large≈5.48M / Small≈2.71M |

`width_mult`（α）在本实现中用于**缩放通道数**（经 `_make_divisible` 对齐到 8 的倍数），不是缩放输入分辨率。

---

## 1. MobileNetV1：深度可分离卷积

### 解决什么问题
标准 3×3 卷积同时混合空间与通道，计算量约为 \(D_K^2 · C_{in} · C_{out} · H · W\)。移动端算力 / 功耗受限，需要在精度可接受的前提下大幅减算。

### 做了什么
把标准卷积拆成两步：

1. **Depthwise**：每个输入通道单独做 \(D_K×D_K\) 空间卷积（`groups=C_in`）
2. **Pointwise**：1×1 卷积做通道混合

再引入 **width multiplier α** 统一缩放各层通道，以及分辨率乘数（论文里的 ρ；本仓库未单独暴露，输入尺寸由调用方决定）。

### 基本块
```
DW 3×3, stride s  → BN → ReLU → PW 1×1 → BN → ReLU
```

### 结构要点（224 输入）
首层标准 3×3/s2，随后一系列 DW+PW；最后一层为 `1024→1024`（stride=1），再 GAP + FC。

### 本仓库细节
- 块：`MobileNetV1Block`，内部用仓库的 `DepthwiseConv` / `PointwiseConv`
- 激活显式为 `ReLU`（覆盖 `BaseConv` 默认 SiLU）
- `width_mult` 作用在通道配置表上

---

## 2. MobileNetV2：倒残差 + 线性瓶颈

### 相对 V1 的问题
V1 是「瘦」的流水线，中间没有残差；低维流形上的非线性（ReLU）容易丢掉信息。若强行在低维加残差，表达能力又不够。

### 做了什么
**Inverted Residual（倒残差）**：

1. 1×1 **expand** 升维（扩展比 t，常用 6）
2. 3×3 **depthwise**（可带 stride 下采样）
3. 1×1 **project** 降回低维，且 **不加激活（linear bottleneck）**
4. 仅当 `stride=1` 且输入输出通道相同时做 shortcut

直觉：在高维空间做非线性更安全，投影回低维时保持线性，减少信息塌缩。

### 基本块
```
[PW expand → BN → ReLU6]  # t=1 时跳过
DW 3×3 → BN → ReLU6
PW project → BN            # 无激活
(+ residual if applicable)
```

### 结构配置 `(t, c, n, s)`
| t | c | n | s |
|---|---|---|---|
| 1 | 16 | 1 | 1 |
| 6 | 24 | 2 | 2 |
| 6 | 32 | 3 | 2 |
| 6 | 64 | 4 | 2 |
| 6 | 96 | 3 | 1 |
| 6 | 160 | 3 | 2 |
| 6 | 320 | 1 | 1 |

末尾再 1×1 扩到 1280 → GAP → FC。首层通道 32，激活为 ReLU6。

---

## 3. MobileNetV3：搜索 + 模块级改进

### 相对 V2 的问题
V2 结构手调；激活、核大小、是否加注意力、分类头是否冗余，仍有优化空间。移动端还希望激活更硬件友好。

### 做了什么
1. **平台感知 NAS + NetAdapt**：搜出 Large / Small 两套具体结构
2. **Squeeze-and-Excitation（SE）**：部分块在 DW 后做通道重标定；门控用 **h-sigmoid**（`ReLU6(x+3)/6`），reduction 默认 4
3. **h-swish**：`x · h-sigmoid(x)`，后半网络用 HS，前半仍多用 ReLU，减轻浅层开销
4. **重新设计的最后阶段**：去掉计算重的一层，分类头改成「先 GAP，再 1×1/Linear 扩维」，显著省算力
5. DW **核可为 3 或 5**

### 基本块
```
[PW expand → BN → NL]     # expand==in 时跳过
DW k×k → BN
[SE]
NL                         # ReLU 或 Hardswish
PW project → BN            # 线性瓶颈
(+ residual if applicable)
```

### Large / Small
| | Large | Small |
|--|-------|-------|
| 首层 | 16 ch, Hardswish, s=2 | 同左 |
| 倒残差块数 | 15 | 11 |
| 特征末 1×1 | 960，无 SE | 576，带 SE |
| 分类头隐层 | 1280 | 1024 |
| 用途 | 精度优先 | 极度轻量 |

构造：

```python
from src.models.common.MobileNet import MobileNetV3, mobilenet_v3_large, mobilenet_v3_small

m = MobileNetV3(variant="large")   # 或 "small"
m = mobilenet_v3_large()
m = mobilenet_v3_small(width_mult=0.75)
```

---

## 三代对比（设计轴）

| 维度 | V1 | V2 | V3 |
|------|----|----|-----|
| 卷积单元 | DW + PW | 倒残差 DW+PW | 同左，核 3/5 |
| 残差 | 无 | 有（同通道且 s=1） | 同 V2 |
| 瓶颈 | 无特殊处理 | **线性** project | 同 V2 |
| 注意力 | 无 | 无 | **SE**（部分层） |
| 激活 | ReLU | ReLU6 | ReLU / **h-swish** |
| 结构来源 | 手工表 | 手工表 | **NAS + 手工 refinement** |
| 分类头 | GAP → FC | 1×1→1280 → GAP → FC | GAP → Linear→隐层(HS)→Dropout→FC |

---

## 使用示例

```python
import torch
from src.models.common.MobileNet import MobileNetV1, MobileNetV2, MobileNetV3

x = torch.randn(1, 3, 224, 224)
y1 = MobileNetV1(num_classes=1000, width_mult=1.0)(x)
y2 = MobileNetV2(num_classes=1000, width_mult=1.0)(x)
y3 = MobileNetV3(variant="large", num_classes=1000)(x)
# y*: [1, 1000]
```

---

## 本仓库实现约定

- 卷积积木复用 `src.module.conv` 的 `BaseConv` / `DepthwiseConv` / `PointwiseConv`
- V3 的 SE（`SqueezeExcite`）写在本文件内，门控为 `nn.Hardsigmoid`，与 CBAM 式 `CAM`（avg+max + sigmoid）不同
- 未加载官方预训练权重；数值对齐以结构与参数量级为准
- 导出：`MobileNetV1`、`MobileNetV2`、`MobileNetV3`、`mobilenet_v3_large`、`mobilenet_v3_small`
