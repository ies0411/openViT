

import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self, latent_vector_dim, head_num, mlp_hidden_dim, drop_rate=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(latent_vector_dim)
        self.ln2 = nn.LayerNorm(latent_vector_dim)
        self.msa = MultiheadedSelfAttention(
            latent_vector_dim=latent_vector_dim,
            head_num=head_num,
            drop_rate=drop_rate,
        )
        self.dropout = nn.Dropout(drop_rate)
        self.mlp = nn.Sequential(
            nn.Linear(latent_vec_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(mlp_hidden_dim, latent_vec_dim),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        z = self.ln1(x)
        z, att = self.msa(z)
        z = self.dropout(z)
        x = x + z
        z = self.ln2(x)
        z = self.mlp(z)
        x = x + z

        return x, att