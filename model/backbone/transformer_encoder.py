

import torch
import torch.nn as nn


class MultiheadedSelfAttention(nn.Module):
    def __init__(self, latent_vector_dim, head_num, drop_rate):
        super().__init__()
        self.latent_vector_dim = latent_vector_dim
        self.head_num = head_num
        self.query = nn.Linear(latent_vector_dim, latent_vector_dim)
        self.key = nn.Linear(latent_vector_dim, latent_vector_dim)
        self.value = nn.Linear(latent_vector_dim, latent_vector_dim)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        q = self.query(x)  # toal head에 한 query
        k = self.key(x)  # toal head에 한 key
        v = self.value(x)  # toal head에 한 value
        q = q.view(x.size(0), self.head_num, -1, self.head_dim)
        k = k.view(x.size(0), self.head_num, self.head_dim, -1)
        v = v.view(x.size(0), self.head_num, -1, self.head_dim)
        att = torch.softmax(
            q @ k / torch.sqrt(self.head_dim * torch.ones(1)).to(device), dim=-1
        )
        batch_size = x.size(0)
        x = self.dropout(attention) @ v
        x = x.reshape(batch_size, -1, self.latent_vector_dim)


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