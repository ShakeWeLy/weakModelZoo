import torch
import torch.nn as nn


def global_max_pool(x: torch.Tensor, spatial_dims: int) -> torch.Tensor:
    """
    x: [B, C, *spatial]
    spatial_dims: 空间维度个数，如 2/3/4
    """
    return x.flatten(2).amax(dim=2, keepdim=True).view(
        *x.shape[:2], *([1] * spatial_dims)
    )


class AxialSelfAttention(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 32, num_heads: int = 8, axia_num: int=1):
        super().__init__()
        self.axia_num = axia_num # 012-->DHW
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(input_dim, embed_dim*3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: (B, C, D, H, W, T)
        Returns:
            x: (B, C, D, H, W, T)
        '''
        B, C, D, H, W, T = x.shape

        if self.axia_num == 0:      # 对 D 做注意力
            # (B, D, H, W, T, C) -> (B*H*W*T, D, C)
            x_ = x.permute(0, 2, 3, 4, 5, 1).reshape(B * H * W * T, D, C)
            N = D
        elif self.axia_num == 1:    # 对 H 做注意力
            # (B, D, H, W, T, C) -> (B*D*W*T, H, C)
            x_ = x.permute(0, 2, 3, 4, 5, 1).reshape(B, D, H, W, T, C)
            x_ = x_.permute(0, 1, 3, 4, 5, 2).reshape(B * D * W * T, H, C)
            N = H
        elif self.axia_num == 2:    # 对 W 做注意力
            # (B, D, H, W, T, C) -> (B*D*H*T, W, C)
            x_ = x.permute(0, 2, 3, 4, 5, 1).reshape(B, D, H, W, T, C)
            x_ = x_.permute(0, 1, 2, 4, 5, 3).reshape(B * D * H * T, W, C)
            N = W
        else:
            raise ValueError(f"axia_num must be 0/1/2, got {self.axia_num}")

        batch = x_.shape[0]

        # QKV + 多头注意力
        qkv = self.qkv(x_)                                           # (batch, N, 3*C)
        qkv = qkv.reshape(batch, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                             # (3, batch, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)                                  # (batch, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(batch, N, C)               # (batch, N, C)

        # 还原
        if self.axia_num == 0:
            out = out.reshape(B, H, W, T, D, C).permute(0, 4, 1, 2, 3, 5)
        elif self.axia_num == 1:
            out = out.reshape(B, D, W, T, H, C).permute(0, 1, 4, 2, 3, 5)
            # 注意这里得到的是 (B, D, H, W, T, C)
        elif self.axia_num == 2:
            out = out.reshape(B, D, H, T, W, C).permute(0, 1, 2, 4, 3, 5)
            # 得到 (B, D, H, W, T, C)

        return out.permute(0, 5, 1, 2, 3, 4).contiguous()   # (B, C, D, H, W, T)

class AFL(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 32, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm_fmri = nn.LayerNorm(input_dim)
        self.norm_dti = nn.LayerNorm(input_dim)
        # self.attention_fmri_d = AxialSelfAttention(embed_dim, num_heads, 0)
        self.attention_fmri_h = AxialSelfAttention(input_dim, embed_dim, num_heads, 1)
        self.attention_fmri_w = AxialSelfAttention(input_dim, embed_dim, num_heads, 2)
        # self.attention_dti_d = AxialSelfAttention(embed_dim, num_heads, 0)
        self.attention_dti_h = AxialSelfAttention(input_dim, embed_dim, num_heads, 1)
        self.attention_dti_w = AxialSelfAttention(input_dim, embed_dim, num_heads, 2)
        
        # maxpooling

        # fussion p
        self.learnable_weight = nn.Parameter(torch.full((embed_dim,), 0.5))


    def forward(
        self,
        x_fmri: torch.Tensor,
        x_dti: torch.Tensor
    ) -> torch.Tensor:
        '''
        Args:
            x_fmri: (B, C, D, H, W, T)
            x_dti: (B, C, D, H, W)
        Returns:
            x: (B, C, H, W)
        '''
        # 进行dti到fmri对齐
        _, _, _, _, _, T = x_fmri.shape
        x_dti = x_dti.unsqueeze(dim=-1).repeat(1,1,1,1,1,T)
        print(x_dti.shape)
        print(x_fmri.shape)

        # 进行归一化
        x_fmri = self.norm_fmri(x_fmri.permute(0,2,3,4,5,1)).permute(0,5,1,2,3,4)
        x_dti = self.norm_dti(x_dti.permute(0,2,3,4,5,1)).permute(0,5,1,2,3,4)
        
        # 进行注意力计算
        # f_x_fmri = self.attention_fmri_d(self.attention_fmri_h(self.attention_fmri_w(x_fmri)))
        # f_x_dti = self.attention_dti_d(self.attention_dti_h(self.attention_dti_w(x_dti)))
        f_x_fmri = self.attention_dti_w(self.attention_dti_h(x_fmri))  # 没有D纬度的注意力
        f_x_dti = self.attention_dti_w(self.attention_dti_h(x_dti))  # 没有D纬度的注意力

        print(f_x_fmri.shape)
        print(f_x_dti.shape)

        f_x_fmri_p = global_max_pool(f_x_fmri, 4)
        f_x_dti_p = global_max_pool(f_x_dti, 4)

        # fussion
        # self.learnable_weight = self.learnable_weight.view(1,self.embed_dim,1,1,1,1 )  # [B,C,D,W,H,T] 
        # print(self.learnable_weight.shape)  # 忘记赋值回去了!
        W = self.learnable_weight.view(1,self.embed_dim,1,1,1,1 )  # [B,C,D,W,H,T]
        print(W.shape)
        f_fusion = (f_x_fmri_p * W) + f_x_dti_p * (1 - W)
        return f_fusion

if __name__ == "__main__":
    x_fmri = torch.randn(1, 128, 10, 10, 10, 10)
    x_dti = torch.randn(1, 128, 10, 10, 10)
    model = AFL(input_dim=128, embed_dim=128, num_heads=8)
    f_fusion = model(x_fmri, x_dti)
    print(f_fusion.shape)