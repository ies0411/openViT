import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.embedding_dimension = model_cfg.embedding_dimension
        self.hidden_dimension = model_cfg.hidden_dimension
        self.drop_rate = model_cfg.drop_rate
        self.norm_1 = nn.LayerNorm(self.embedding_dimension)
        self.norm_2 = nn.LayerNorm(self.embedding_dimension)
        self.msa = MultiheadedSelfAttention(model_cfg)
        self.dropout = nn.Dropout(self.drop_rate)
        self.mlp_module = nn.Sequential(
            nn.Linear(self.embedding_dimension, self.hidden_dimension),
            nn.GELU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.hidden_dimension, self.embedding_dimension),
            nn.Dropout(self.drop_rate),
        )

    def forward(self, x):
        z = self.norm_1(x)  # batch_size, patch개수 + 1, embedding_dimension
        z, att = self.msa(
            z
        )  # z : batch_size, patch개수 + 1 ,embedding_dimension(head_number x head_dimension)
        z = self.dropout(z)
        x = x + z
        z = self.norm_2(x)
        z = self.mlp_module(z)
        x = (
            x + z
        )  # x : batch_size, patch개수 + 1 ,embedding_dimension(head_number x head_dimension)

        return x, att
