import torch.nn as nn
import torch.nn.functional as F
import layers


class SparseGraphTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp_in_x = nn.Sequential(nn.LazyLinear(cfg.x_hidden_mlp), nn.ReLU(),
                                      nn.Linear(cfg.x_hidden_mlp, cfg.x_hidden), nn.ReLU())

        self.mlp_in_e = nn.Sequential(nn.LazyLinear(cfg.e_hidden_mlp), nn.ReLU(),
                                      nn.Linear(cfg.e_hidden_mlp, cfg.e_hidden), nn.ReLU())

        self.tf_layers = nn.ModuleList([layers.TransformerLayer(dx=cfg.x_hidden,
                                                                de=cfg.e_hidden,
                                                                n_head=cfg.num_heads,
                                                                dim_ff_x=cfg.x_hidden_mlp,
                                                                dim_ff_e=cfg.e_hidden_mlp,
                                                                dropout=cfg.dropout,
                                                                layer_norm_eps=cfg.layer_norm_eps)
                                        for _ in range(cfg.num_layers)])

        self.node_pooling = layers.PoolingLayer(cfg.x_hidden, cfg.global_hidden_dim)
        self.edge_pooling = layers.PoolingLayer(cfg.e_hidden, cfg.global_hidden_dim)
        self.final_linear = nn.Linear(cfg.global_hidden_dim, cfg.global_dim_out)

    def forward(self, data):
        # Initial MLPs
        x = self.mlp_in_x(data.x)
        edge_attr_embedding = self.mlp_in_e(data.edge_attr)

        # Pool the edge features
        edge_batch = data.batch[data.edge_index[0]]
        e_out = self.edge_pooling(edge_attr_embedding, edge_batch)

        for layer in self.tf_layers:
            x = layer(x, data.edge_index, edge_attr_embedding)

        x_out = self.node_pooling(x, data.batch)

        out = F.relu(x_out + e_out)
        out = self.final_linear(out)
        return out



