import torch
import torch.nn as nn

from src.models.common.ResNet import ResNet18
from src.module.conv import BaseConv






class Transformer(nn.Module):
    pass



class DETR(nn.Module):
    def __init__(self, num_classes: int, num_queries: int, hidden_dim: int, nheads: int, num_encoder_layers: int, num_decoder_layers: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.nheads = nheads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward

        self.backbone = ResNet18()
        self.input_proj = BaseConv(3, hidden_dim, kernel_size=1)  # !project the input channels to the hidden dimension
        self.transformer = Transformer(num_encoder_layers, num_decoder_layers, hidden_dim, nheads, dim_feedforward)
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

    def forward(self):
        pass

