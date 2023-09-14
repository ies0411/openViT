import torch
import torch.nn as nn


# reference : https://github.com/hyunwoongko/transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        # embedding vector dimension = positionalending dimension
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        """
        seq_len, batch_size,embedding_dim
        """
        return x + self.encoding[: x.size(0)]
        # return self.dropout(x)


class EmbeddingMudule(
    nn.Module
):  # patch_numx(patchxpatchximage_channel)  -> patch_num x D(latent_vector_dimentsion)
    def __init__(
        self, patch_vec_size, patches_num, latent_vector_dimension, drop_rate=0.1
    ):
        super(EmbeddingMudule, self).__init__()
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


class MultiheadedSelfAttention(nn.Module):
    def __init__(self, latent_vec_dim, num_heads, drop_rate):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_heads = num_heads
        self.latent_vec_dim = latent_vec_dim
        self.head_dim = int(latent_vec_dim / num_heads)
        self.query = nn.Linear(latent_vec_dim, latent_vec_dim)
        self.key = nn.Linear(latent_vec_dim, latent_vec_dim)
        self.value = nn.Linear(latent_vec_dim, latent_vec_dim)
        self.scale = torch.sqrt(self.head_dim * torch.ones(1)).to(device)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        batch_size = x.size(0)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).permute(
            0, 2, 3, 1
        )  # k.t
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attention = torch.softmax(q @ k / self.scale, dim=-1)
        x = self.dropout(attention) @ v
        x = x.permute(0, 2, 1, 3).reshape(batch_size, -1, self.latent_vec_dim)

        return x, attention


class TransformerEncoder(nn.Module):
    def __init__(self, latent_vec_dim, num_heads, mlp_hidden_dim, drop_rate):
        super().__init__()
        self.ln1 = nn.LayerNorm(latent_vec_dim)
        self.ln2 = nn.LayerNorm(latent_vec_dim)
        self.msa = MultiheadedSelfAttention(
            latent_vec_dim=latent_vec_dim, num_heads=num_heads, drop_rate=drop_rate
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


class ViT(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super(ViT, self).__init__()
        self.cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset

        self.embedding = EmbeddingMudule(
            self.cfg.PATCH_SIZE, self.cfg.PATCH_NUM, self.cfg.LATENT_DIMENSION
        )
        self.transformer = TransformerEncoder()

        # self.class_names = dataset.class_names
        # self.register_buffer('global_step', torch.LongTensor(1).zero_())

    def forward(self, input_data):
        print("todo")
