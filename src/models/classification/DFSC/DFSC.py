from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

# from src.module.attention.CAM.cam import CAM
from src.module.transformer import TransformerEncoderBlock
from src.module.attention.GAT import GATAttention


# def pearson_flat(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#     # x, y: 任意同形状 tensor
#     x = x.flatten()
#     y = y.flatten()
#     x = x - x.mean()
#     y = y - y.mean()
#     return (x * y).sum() / (x.norm() * y.norm() + 1e-8)

# def FSC_Personalized(x: torch.Tensor) -> torch.Tensor:
#     '''
#     DTI分别于fMRIs(T)的皮尔逊相关系数
#     x: [T+1, N, D]  # DTI+fMRIs(1+T), node, feature
#     return: [T, N, D]
#     '''
#     dti_matrix = x[0]
#     fmris_matrix = x[1:]
#     output_matrix = torch.zeros_like(fmris_matrix)
#     for i, fmri in enumerate(fmris_matrix):
#         pearson_matrix = pearson_flat(dti_matrix, fmri)
#         output_matrix[i] = pearson_matrix
#     return output_matrix

def filter_topk_global(x: torch.Tensor, ratio: float = 0.3) -> torch.Tensor:
    """
    x: [B, T+1, N, N]
    return: [B, T+1, N, N]，每个 N×N 内部保留全局 top-k，其余置 0
    """
    *lead, N1, N2 = x.shape
    assert N1 == N2, f"最后两维必须相等，实际 {x.shape}"

    total = N1 * N2
    k = max(1, int(total * ratio))

    flat = x.reshape(*lead, total)            # [B, T+1, N*N]
    topk_idx = flat.topk(k, dim=-1).indices   # [B, T+1, k]

    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(-1, topk_idx, True)

    return flat.masked_fill(~mask, 0.0).view(x.shape)


def FSC_Personalized(x: torch.Tensor) -> torch.Tensor:
    '''
    x: [B, T+1, N, D]  # DTI + fMRIs(1+T), node, feature (batch 维已向量化)
    return: [B, T, N, N]  # 每个样本、每个时间点，fMRI节点与DTI节点的D维特征相关性
    '''
    dti = x[:, 0]     # [B, N, D]
    fmris = x[:, 1:]  # [B, T, N, D]

    # 中心化
    dti_c = dti - dti.mean(dim=-1, keepdim=True)          # [B, N, D]
    fmri_c = fmris - fmris.mean(dim=-1, keepdim=True)     # [B, T, N, D]

    # 归一化
    dti_n = dti_c / (dti_c.norm(dim=-1, keepdim=True) + 1e-8)      # [B, N, D]
    fmri_n = fmri_c / (fmri_c.norm(dim=-1, keepdim=True) + 1e-8)   # [B, T, N, D]

    # 相关: [B, T, N, D] x [B, N, D] -> [B, T, N, N]
    corr = torch.einsum('btnd,bmd->btnm', fmri_n, dti_n)
    return corr


class SEattention(nn.Module):
    def __init__(self, in_features, dim):
        super(SEattention, self).__init__()
        self.in_features = in_features
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(self.in_features, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.in_features),
            nn.Sigmoid()
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        '''
        x: [B, L, D]  L=脑区数/序列长度, D=特征维度
        '''
        b, l, d = x.shape
        # print(x.shape)
        # 在 L 维上池化, 保留 D: [B, D]
        atten_score = x.mean(dim=1)
        # MLP 在 D 上: [B, D]
        atten = self.mlp(atten_score).view(b, 1, d)   # [B, 1, D]
        return x * atten + x    


class DFSC(nn.Module):
    def __init__(self, in_features, features_dim, se_dim=64, gat_heads=1, num_classes=2, transformer_embed_dim=20, transformer_heads=4, mlp_dim=128):
        super(DFSC, self).__init__()
        self.num_classes = num_classes
        self.d = features_dim
        self.gat_1 = GATAttention(in_features=in_features, out_features=features_dim, heads=gat_heads)
        self.gat_2 = GATAttention(in_features=in_features, out_features=features_dim, heads=gat_heads)
        self.se = SEattention(in_features=features_dim, dim=se_dim)
        self.transformer = TransformerEncoderBlock(embed_dim=transformer_embed_dim, num_heads=transformer_heads, ffn_hidden_channels=features_dim)
        # self.fc = nn.Linear(out_features, num_classes)
        self.mlp = nn.Sequential(
            nn.Linear(features_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, num_classes),
        )

    # def forward(self, x, adj):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T+1, N, N]  B=batch, T+1=时间步数量个(DTI+fMRIs(T个)), N=节点数, N×N为皮尔逊相关系数矩阵 
        Returns:
            [B, C]  C=类别数
        """
        b, t, n, _ = x.size()  # t = T+1, d = input features
        adj = filter_topk_global(x)
        # 将 B 与时间维合并为一个 batch, 一次性计算所有时间步 (权重共享, 与逐步调用等价)
        gat_1_score_matrix = self.gat_1(x.reshape(b*t, n, n), adj.reshape(b*t, n, n))  # [B*(T+1), N, D]
        gat_1_score_matrix = gat_1_score_matrix.reshape(b, t, n, self.d)  # [B, T+1, N, D]
        personalized_score_matrix = FSC_Personalized(gat_1_score_matrix)  # [B, T, N, N]
        # gat_2 的输入是 T 个时间步的特征, 同样合并 B 与 T 维
        t_pers = t - 1
        pers = personalized_score_matrix.reshape(b*t_pers, n, n)
        gat_2_score_matrix = self.gat_2(pers, pers)  # [B*T, N, D], 邻接与特征相同
        gat_2_score_matrix = gat_2_score_matrix.reshape(b, t_pers, n, self.d)  # [B, T, N, D]

        final_x = torch.cat([gat_1_score_matrix, gat_2_score_matrix], dim=1)  # [B, 2T+1, N, D]
        # print(final_x.shape)
        # print(t,d,self.d)
        l = (2*(t-1)+1)*n  # 
        final_x = final_x.view(b, l, self.d)  # [B, (2T+1)*N, D]
        self.se.in_features = l
        final_x = self.se(final_x)  # [B, (2T+1)*N, D]
        print(final_x.shape)
        out = self.transformer(final_x)
        out = out.mean(dim=1)  # [B, D]
        return self.mlp(out)  # [B, C] logits, 训练时配合 nn.CrossEntropyLoss; 推理需概率时对输出做 softmax


if __name__ == "__main__":
    x = torch.randn(size=(2, 3, 10, 10))  # [B, T+1, N, N]
    # adj = torch.randn(size=(2, 3, 10, 10))  # [B, T+1, N, N]
    dfsc = DFSC(in_features=10, features_dim=20, gat_heads=3, num_classes=2)
    # out = dfsc(x, adj)
    # out = dfsc(x, adj)
    out = dfsc(x)
    print(out.shape)  # [2, 2]