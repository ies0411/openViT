import torch
import torch.nn as nn


class LinearProjection(nn.Module):

    def __init__(self, patch_vec_size, num_patches, latent_vec_dim, drop_rate):
        super().__init__()
        self.linear_proj = nn.Linear(patch_vec_size, latent_vec_dim)
        self.cls_token = nn.Parameter(torch.randn(1, latent_vec_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, latent_vec_dim))
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.cat([self.cls_token.repeat(batch_size, 1, 1), self.linear_proj(x)], dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        return x

class LinearEmbeddingMudule(
    nn.Module
):  # patch_numx(patchxpatchximage_channel)  -> patch_num x D(latent_vector_dimentsion)
    def __init__(
        self, patch_vec_size, patches_num, latent_vector_dimension, drop_rate=0.1
    ):
        self.linear_proj = nn.Linear(patch_vec_size, latent_vector_dimension)
        self.class_token = nn.Parameter(torch.randn(1, latent_vector_dimension))
        self.pos_embedding = PositionalEncoding(latent_vector_dimension)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.cat(
            [self.cls_token.repeat(batch_size, 1, 1), self.linear_proj(x)], dim=1
        )
        x += self.pos_embedding
        x = self.dropout(x)
        return x