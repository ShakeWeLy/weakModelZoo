"""推理期 BN 折叠（Conv2d + BatchNorm2d → 带偏置的 Conv2d）。

作用对象：src/module/conv/base.py 中的 BaseConv，其结构为
    self.block = nn.Sequential(Conv2d, [BatchNorm2d], [激活])
折叠后 block 中不再包含 BN，激活保持不变。数学上与 eval 模式下的 Conv→BN 等价。

只用于推理（model.eval()），不改动源码，不影响训练与 checkpoint。
"""

from __future__ import annotations

import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval


def fold_bn_inplace(model: nn.Module) -> int:
    """原地折叠所有 BaseConv 中的 Conv2d+BatchNorm2d，返回折叠的数量。必须在 model.eval() 下调用。"""
    if model.training:
        raise RuntimeError("BN 折叠只能在 eval 模式下进行（需要 running 统计量）")
    n = 0
    for _, m in list(model.named_modules()):
        blk = getattr(m, "block", None)
        if not isinstance(blk, nn.Sequential) or len(blk) < 2:
            continue
        if isinstance(blk[0], nn.Conv2d) and isinstance(blk[1], nn.BatchNorm2d):
            fused = fuse_conv_bn_eval(blk[0], blk[1])
            m.block = nn.Sequential(fused, *list(blk)[2:])
            n += 1
    return n


def count_bn(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
