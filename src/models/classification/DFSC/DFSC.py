from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

# from src.module.attention.CAM.cam import CAM
from src.module.transformer import TransformerDecoderBlock
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
    x: [T+1, N, D]  # DTI + fMRIs(1+T), node, feature
    return: [T, N, N]  # 每个时间点，DTI节点与fMRI节点的D维特征相关性
    '''
    dti = x[0]        # [N, D]
    fmris = x[1:]     # [T, N, D]
    T, N, D = fmris.shape

    # 中心化
    dti_c = dti - dti.mean(dim=-1, keepdim=True)          # [N, D]
    fmri_c = fmris - fmris.mean(dim=-1, keepdim=True)     # [T, N, D]

    # 归一化
    dti_n = dti_c / (dti_c.norm(dim=-1, keepdim=True) + 1e-8)      # [N, D]
    fmri_n = fmri_c / (fmri_c.norm(dim=-1, keepdim=True) + 1e-8)   # [T, N, D]

    # 相关: [T, N, D] @ [N, D]^T -> [T, N, N]
    corr = torch.einsum('tnd,md->tnm', fmri_n, dti_n)
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
        self.transformer = TransformerDecoderBlock(embed_dim=transformer_embed_dim, num_heads=transformer_heads, ffn_hidden_channels=features_dim)
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
            x: [B, T+1, N, N]  B=batch, T+1=时间步数量个, N=节点数, N×N为皮尔逊相关系数矩阵 
        Returns:
            [B, C]  C=类别数
        """
        b, t, n, _ = x.size()  # t = T+1, d = input features
        gat_1_score_list = []
        adj = filter_topk_global(x)
        for i in range(t):
            score = self.gat_1(x[:,i], adj[:,i])  # [B, N, D]  # adjacency matrix is top 30% filtered feature matrix
            gat_1_score_list.append(score)
        gat_1_score_matrix = torch.stack(gat_1_score_list, dim=1)  # [B, T+1, N, D]
        personalized_score_list = []
        for i in range(b):
            # TODO: FSC_Personalized 函数需要修改输出情况
            personalized_score = FSC_Personalized(gat_1_score_matrix[i])  # [T, N, N]
            personalized_score_list.append(personalized_score)
        personalized_score_matrix = torch.stack(personalized_score_list, dim=0)  # [B, T, N, N]
        gat_2_score_list = []
        for i in range(t-1):  #  -1因为gat_2的输入是T个时间步的特征
            # print(personalized_score_matrix[:,i].shape)
            score = self.gat_2(personalized_score_matrix[:,i], personalized_score_matrix[:,i])  # [B, N, D]  # adjacency matrix same as feature matrix
            gat_2_score_list.append(score)
        gat_2_score_matrix = torch.stack(gat_2_score_list, dim=1)  # [B, T, N, D]

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
        return F.softmax(self.mlp(out), dim=1)


if __name__ == "__main__":
    x = torch.randn(size=(2, 3, 10, 10))  # [B, T+1, N, N]
    # adj = torch.randn(size=(2, 3, 10, 10))  # [B, T+1, N, N]
    dfsc = DFSC(in_features=10, features_dim=20, gat_heads=3, num_classes=2)
    # out = dfsc(x, adj)
    # out = dfsc(x, adj)
    out = dfsc(x)
    print(out.shape)  # [2, 2]