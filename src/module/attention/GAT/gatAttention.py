import torch
import torch.nn as nn
import torch.nn.functional as F


def atten_adjacency_matrix(att: torch.Tensor, adj: torch.Tensor):
    '''
    计算注意力邻接矩阵
    att: [N, N]  注意力矩阵
    adj: [N, N]  邻接矩阵
    return: [N, N]  注意力邻接矩阵
    '''
    atten_matrix = torch.zeros(size=(adj.size(0), adj.size(0)))
    atten_sum = torch.sum(adj, dim=1)
    for i in range(adj.size(0)):
        for j in range(adj.size(1)):
            if adj[i, j] != 0:
                atten_sum[j] += att[i, j]  # 计算邻接边的注意力分数之和
    for i in range(adj.size(0)):
        for j in range(adj.size(1)):
            if adj[i, j] != 0:
                atten_matrix[i, j] = att[i, j] / atten_sum[j]  # 计算注意力邻接矩阵
    return atten_matrix


def atten_matrix(x: torch.Tensor, n: torch.Tensor):
    '''
    计算注意力矩阵
    x: [N, N]  特征矩阵
    n: [N, N]  可训练参数
    return: [N, N]  注意力矩阵
    '''
    atten_matrix = torch.zeros(size=(x.size(0), x.size(0)))
    for i in range(x.size(0)):
        for j in range(x.size(0)): # 这里好像多计算了, 没有邻接的位置
            x_i = x[i]
            x_k = x[j]
            atten = concat_attention(x_i, x_k, n[i])  
            atten_matrix[i, j] = atten
    return atten_matrix


def concat_attention_adjacency_matrix(x: torch.Tensor, adj: torch.Tensor, n: torch.Tensor):
    '''
    计算注意力邻接矩阵, 邻接矩阵过滤
    x: [N, D]  特征矩阵
    adj: [N, D]  邻接矩阵
    n: [N, N, 2D] 可训练分数, NxN个, 每一个是2D(concatenate之后)向量
    return: [N, N]  注意力邻接矩阵
    '''
    concat_matrix = torch.zeros(size=(x.size(0), x.size(0)))
    atten_matrix = torch.zeros(size=(x.size(0), x.size(0)))  
    atten_sum = torch.sum(adj, dim=1)
    for i in range(adj.size(0)):
        for j in range(adj.size(0)):
            if adj[i, j] != 0:
                atten = concat_attention(x[i], x[j], n[i][j])
                concat_matrix[i, j] = atten
                atten_sum[j] += atten
        for j in range(adj.size(0)):
            if adj[i, j] != 0:
                atten_matrix[i, j] = concat_matrix[i, j] / atten_sum[j]
    return atten_matrix


def concat_attention(x_i: torch.Tensor, x_k: torch.Tensor, n: torch.Tensor):
    '''
    计算注意力分数
    x_i: [in_features]  特征向量
    x_k: [in_features]  特征向量
    n: torch.Tensor  偏置
    return: [1]  注意力分数
    '''
    x = torch.cat([x_i, x_k])
    x = F.leaky_relu(x)
    atten = torch.dot(x, n)
    return torch.exp(atten)


class GATAttention(nn.Module):
    def __init__(self, in_features, out_features, heads=1):
        super(GATAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.x_w = nn.Linear(in_features=in_features, out_features=out_features)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, a):
        '''
        x: [B, N, N]  特征矩阵
        a: [B, N, N]  邻接矩阵
        return: [B, N, out_features]  注意力权重
        '''
        x = self.x_w(x)  # [B, N, out_features]
        self.n = nn.Parameter(torch.zeros(size=(x.size(0), self.heads, self.in_features, self.in_features, self.out_features*2)))  # [B, heads, N, N, 2D]
        out = torch.zeros(size=(x.size(0), x.size(1), self.out_features))  # [B, N, out_features]
        for i in range(x.size(0)):
            k = torch.zeros(size=(x.size(1), self.out_features))  # [N, out_features]
            for k in range(self.heads):
                atten_matrix = concat_attention_adjacency_matrix(x[i], a[i], self.n[i][k])
                atten_matrix = self.softmax(atten_matrix)
                e = torch.matmul(atten_matrix, x[i])
                e = self.leaky_relu(e)
                k += e  # [N, out_features]
            out[i] = k
        return out  # [B, N, out_features]

# if __name__ == "__main__":
#     x = torch.randn(size=(10, 10))
#     adj = torch.randn(size=(10, 10))
#     n = torch.randn(size=(10, 10, 20))  # 20, 因为是x行作为特征计算注意力, 所以是20
#     atten_matrix = concat_attention_adjacency_matrix(x, adj, n)
#     print(atten_matrix)

if __name__ == "__main__":
    import time
    x = torch.randn(size=(2, 10, 10))  # [B, N, N]
    adj = torch.randn(size=(2, 10, 10))  # [B, N, N]
    start = time.perf_counter()
    gat = GATAttention(in_features=10, out_features=20, heads=3)
    out = gat(x, adj)
    time.perf_counter()
    print(time.perf_counter() - start)
    # print(out)
    print(out.shape)  # [2, 10, 20]