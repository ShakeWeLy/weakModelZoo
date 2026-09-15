import torch
import torch.nn as nn

from ...conv import BaseConv
class SAM(nn.Module):
    '''
    Spatial Attention Module
    '''
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()

        if kernel_size not in (3, 7):
            raise ValueError("kernel_size 通常应为 3 或 7")

        self.conv = BaseConv(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bn=False,
            act=False,
        )
        self.sigmoid = nn.Sigmoid()

        x_avg = torch.mean(x, dim=1, keepdim=True)  # using mean instead of nn.AdaptiveAvgPool2d(1)
        x_max = torch.max(x, dim=1, keepdim=True).values  # return (values, indices), so we need to get the values

        attention = self.sigmoid(self.conv(torch.cat([x_avg, x_max], dim=1)))

        return attention * x

'''
当前有几个明显问题：
kernel_size 仅支持 3 和 7
ValueError: kernel_size 通常应为 3 或 7
使用 nn.AdaptiveAvgPool2d(1) 和 nn.AdaptiveMaxPool2d(1) 代替手动平均池化和最大池化
avg_pool = nn.AdaptiveAvgPool2d(1)
max_pool = nn.AdaptiveMaxPool2d(1)
x_avg = avg_pool(x)
x_max = max_pool(x)
使用 torch.mean() 和 torch.max() 代替手动平均池化和最大池化
x_avg = torch.mean(x, dim=1, keepdim=True)
x_max = torch.max(x, dim=1, keepdim=True).values
'''
