import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, GATConv, GATv2Conv


class PoolingLayer(nn.Module):
    def __init__(self, d, global_hidden_dim):
        """ Pool node or edge features to graph level features. """
        super().__init__()
        self.lin = nn.Linear(4 * d, global_hidden_dim)
        self.lin2 = nn.Linear(global_hidden_dim, global_hidden_dim)

    def forward(self, features, batch):
        assert features.shape[0] == len(batch)
        m = global_mean_pool(features, batch)
        ma = global_max_pool(features, batch)
        mi = -global_max_pool(-features, batch)
        std = global_mean_pool(features ** 2, batch) - global_mean_pool(features, batch) ** 2
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        out = F.relu(out)
        out = self.lin2(out)
        return out


class TransformerLayer(nn.Module):
    """ Transformer that updates node, edge and global features
        d_x: node features
        d_e: edge features
        dz : global features
        n_head: the number of heads in the multi_head_attention
        dim_feedforward: the dimension of the feedforward network model after self-attention
        dropout: dropout probablility. 0 to disable
        layer_norm_eps: eps value in layer normalizations.
    """
    def __init__(self, dx: int, de: int, n_head: int, dim_ff_x, dim_ff_e, dropout, layer_norm_eps,
                 use_v2: bool = True):
        super().__init__()

        conv = GATv2Conv if use_v2 else GATConv
        assert dx % n_head == 0
        self.self_attn = conv(in_channels=dx, out_channels=int(dx / n_head), heads=n_head, edge_dim=de)

        self.linX1 = nn.Linear(dx, dim_ff_x)
        self.linX2 = nn.Linear(dim_ff_x, dx)
        self.normX1 = nn.LayerNorm(dx, eps=layer_norm_eps)
        self.normX2 = nn.LayerNorm(dx, eps=layer_norm_eps)
        self.dropoutX1 = nn.Dropout(dropout)
        self.dropoutX2 = nn.Dropout(dropout)
        self.dropoutX3 = nn.Dropout(dropout)

        self.linE1 = nn.Linear(de, dim_ff_e)
        self.linE2 = nn.Linear(dim_ff_e, de)
        self.normE1 = nn.LayerNorm(de, eps=layer_norm_eps)
        self.normE2 = nn.LayerNorm(de, eps=layer_norm_eps)
        self.dropoutE1 = nn.Dropout(dropout)
        self.dropoutE2 = nn.Dropout(dropout)
        self.dropoutE3 = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        new_x = self.self_attn(x, edge_index, edge_attr)
        new_x_d = self.dropoutX1(new_x)
        x = self.normX1(x + new_x_d)

        ff_output_x = self.linX2(self.dropoutX2(F.relu(self.linX1(x))))
        ff_output_x = self.dropoutX3(ff_output_x)
        x = self.normX2(x + ff_output_x)

        return x
