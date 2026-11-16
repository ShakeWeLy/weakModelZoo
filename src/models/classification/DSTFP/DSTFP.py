from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.module.transformer.swinTransformer import SwinTransformerBlock3D, SwinTransformerBlock4D
from src.module.attention import  AFL

class classification_head(nn.Module):
    def __init__(self, input_dim: int, class_num: int = 3):
        super().__init__()
        self.linear = nn.Linear(input_dim, class_num)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x: [B, C, D, W, H, T]
        '''
        x = x.flatten(2).transpose(1, 2)
        # x = x.flatten(2)
        x = x.mean(dim=1)
        return self.linear(x)  # [B, class_num]



class DSTFP(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, num_heads: int, window_size: int=7):
        super().__init__()
        self.stage = 4
        self.fmri_swin_transformer = nn.ModuleList([
            nn.Sequential(
                SwinTransformerBlock4D(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                ),
                SwinTransformerBlock4D(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=window_size // 2,
                ),
            )
            for _ in range(self.stage)
        ])

        self.dti_swin_transformer = nn.ModuleList([
            nn.Sequential(
                SwinTransformerBlock3D(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                ),
                SwinTransformerBlock3D(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=window_size // 2,
                ),
            )
            for _ in range(self.stage)
        ])

        self.afl = AFL(input_dim, embed_dim, num_heads)
 
        self.classification_head = classification_head(input_dim=embed_dim, class_num=3)

    def forward(self, fmri: torch.Tensor, dti: torch.Tensor) -> torch.Tensor:
        fmri = fmri.permute(0, 2, 3, 4, 5, 1)
        dti = dti.permute(0, 2, 3, 4, 1)


        # fmri = self.fmri_swin_transformer(fmri)  # !!错误写法
        # dti = self.dti_swin_transformer(dti)
        for blk in self.fmri_swin_transformer:
            fmri = blk(fmri)
        for blk in self.dti_swin_transformer:
            dti = blk(dti)

        fmri = fmri.permute(0, 5, 1, 2, 3, 4)
        dti = dti.permute(0, 4, 1, 2, 3)

        feature = self.afl(x_fmri=fmri, x_dti=dti)
        return self.classification_head(feature)


if __name__ == "__main__":
    x_fmri = torch.randn(1, 128, 14, 14, 14, 14)
    x_dti = torch.randn(1, 128, 14, 14, 14)
    model = DSTFP(input_dim=128,
                embed_dim=128,
                num_heads=8)
    out = model(x_fmri, x_dti)
    print(out.shape)