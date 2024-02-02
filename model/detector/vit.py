import torch
import torch.nn as nn
from template import DetectorTemplate


# from .detector3d_template import Detector3DTemplate
class ViT(DetectorTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

        # self.embedding = EmbeddingMudule(
        #     self.cfg.PATCH_SIZE, self.cfg.PATCH_NUM, self.cfg.LATENT_DIMENSION
        # )
        # self.transformer = TransformerEncoder()

        # self.class_names = dataset.class_names
        # self.register_buffer('global_step', torch.LongTensor(1).zero_())

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {"loss": loss}
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts


# # reference : https://github.com/hyunwoongko/transformer
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000, dropout=0.1):
#         # embedding vector dimension = positionalending dimension
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         # same size with input matrix (for adding with input matrix)
#         self.encoding = torch.zeros(max_len, d_model)
#         self.encoding.requires_grad = False  # we don't need to compute gradient

#         pos = torch.arange(0, max_len)
#         pos = pos.float().unsqueeze(dim=1)
#         # 1D => 2D unsqueeze to represent word's position

#         _2i = torch.arange(0, d_model, step=2).float()
#         # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
#         # "step=2" means 'i' multiplied with two (same with 2 * i)

#         self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
#         self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
#         # compute positional encoding to consider positional information of words

#     def forward(self, x):
#         """
#         seq_len, batch_size,embedding_dim
#         """
#         return x + self.encoding[: x.size(0)]
#         # return self.dropout(x)


class EmbeddingMudule(nn.Module):
    def __init__(
        self, patch_vector_size, patches_num, latent_vector_dimension, drop_rate=0.1
    ):
        super(EmbeddingMudule, self).__init__()
        self.linear_projection = nn.Linear(patch_vector_size, latent_vector_dimension)
        self.class_token = nn.Parameter(torch.randn(1, latent_vector_dimension))
        self.positional_emdedding = nn.Parameter(
            torch.randn(1, patches_num + 1, latent_vector_dimension)
        )
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        # x : B, patch개수, 3 x patch_w x patch_h
        # repeat == expand
        # linear_proj : B, patch개수, latent_vector_dimension,  cls_token : B, 1, 1
        # B, patch개수 + 1, latent_vector_dimension
        batch_size = x.size(0)
        x = torch.cat(
            [self.class_token.repeat(batch_size, 1, 1), self.linear_projection(x)],
            dim=1,
        )
        # B, patch개수 + 1, latent_vector_dimension
        x += self.positional_emdedding
        x = self.dropout(x)
        return x


class MultiheadedSelfAttention(nn.Module):
    def __init__(self, latent_vector_dim, head_num, drop_rate, device):
        super().__init__()
        self.device = device
        self.head_num = head_num
        self.latent_vector_dim = latent_vector_dim
        self.head_dim = int(latent_vector_dim / head_num)
        self.query = nn.Linear(latent_vector_dim, latent_vector_dim)
        self.key = nn.Linear(latent_vector_dim, latent_vector_dim)
        self.value = nn.Linear(latent_vector_dim, latent_vector_dim)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        # B, patch개수 + 1, latent_vector_dim
        batch_size = x.size(0)
        q = self.query(x)  # q : B, patch개수 + 1, latent_vector_dim
        k = self.key(x)  # k : B, patch개수 + 1, latent_vector_dim
        v = self.value(x)  # v : B, patch개수 + 1, latent_vector_dim
        q = q.view(batch_size, -1, self.head_num, self.head_dim).permute(
            0, 2, 1, 3
        )  # B, patch개수 + 1 ,head number, head dimension -> B, head_number, patch개수 + 1, head_dimension(int(latent_vector_dim / num_heads)
        k = k.view(batch_size, -1, self.head_num, self.head_dim).permute(
            0, 2, 3, 1
        )  # B, patch개수 + 1 ,head number, head dimension -> B, head_number, head_dimension(int(latent_vector_dim / num_heads), patch개수 + 1
        v = v.view(batch_size, -1, self.head_num, self.head_dim).permute(
            0, 2, 1, 3
        )  # B, patch개수 + 1 ,head number, head dimension -> B, head_number, patch개수 + 1, head_dimension(int(latent_vector_dim / num_heads)
        attention = torch.softmax(
            q @ k / torch.sqrt(self.head_dim * torch.ones(1)), dim=-1
        )  # B, head_number, patch개수 + 1, patch개수 + 1
        x = (
            self.dropout(attention) @ v
        )  # B, head_number,  patch개수 + 1,  patch개수 + 1 x  B, head_number, patch개수 + 1, head_dimension => B, head_number, patch개수 + 1, head_dimension
        x = x.permute(0, 2, 1, 3).reshape(
            batch_size, -1, self.latent_vector_dim
        )  # B, patch개수 + 1, head_number, head_dimension -> B, patch개수 + 1 ,latent_vector_dim(head_number x head_dimension)
        return x, attention


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
            nn.Linear(latent_vector_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(mlp_hidden_dim, latent_vector_dim),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        # B, patch개수 + 1, latent_vector_dim
        z = self.ln1(x)  # B, patch개수 + 1, latent_vector_dim
        z, attention_vector = self.msa(
            z
        )  # z : B, patch개수 + 1 ,latent_vector_dim(head_number x head_dimension)
        z = self.dropout(z)
        x = x + z
        z = self.ln2(x)
        z = self.mlp(z)
        x = x + z  # B, patch개수 + 1 ,latent_vector_dim

        return x, attention_vector


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
