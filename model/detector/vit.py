import torch
import torch.nn as nn
from template import DetectorTemplate


# from .detector3d_template import Detector3DTemplate
class ViT(DetectorTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        # self.cfg = model_cfg
        # self.num_class = num_class
        # self.dataset = dataset

        self.module_list = self.build_networks()

        # self.embedding = EmbeddingMudule(
        #     self.cfg.PATCH_SIZE, self.cfg.PATCH_NUM, self.cfg.LATENT_DIMENSION
        # )
        # self.transformer = TransformerEncoder()

        # self.class_names = dataset.class_names
        # self.register_buffer('global_step', torch.LongTensor(1).zero_())
    def build_networks(self):
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size,
            'depth_downsample_factor': self.dataset.depth_downsample_factor
        }
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)
        return model_info_dict['module_list']


    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

# class SECONDNet(Detector3DTemplate):
#     def __init__(self, model_cfg, num_class, dataset):
#         super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
#         self.module_list = self.build_networks()

#     def build_networks(self):
#         model_info_dict = {
#             'module_list': [],
#             'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,
#             'num_point_features': self.dataset.point_feature_encoder.num_point_features,
#             'grid_size': self.dataset.grid_size,
#             'point_cloud_range': self.dataset.point_cloud_range,
#             'voxel_size': self.dataset.voxel_size,
#             'depth_downsample_factor': self.dataset.depth_downsample_factor
#         }
#         for module_name in self.module_topology:
#             module, model_info_dict = getattr(self, 'build_%s' % module_name)(
#                 model_info_dict=model_info_dict
#             )
#             self.add_module(module_name, module)
#         return model_info_dict['module_list']

#     def forward(self, batch_dict):
#         for cur_module in self.module_list:
#             batch_dict = cur_module(batch_dict)

#         if self.training:
#             loss, tb_dict, disp_dict = self.get_training_loss()

#             ret_dict = {
#                 'loss': loss
#             }
#             return ret_dict, tb_dict, disp_dict
#         else:
#             pred_dicts, recall_dicts = self.post_processing(batch_dict)
#             return pred_dicts, recall_dicts

#     def get_training_loss(self):
#         disp_dict = {}

#         loss_rpn, tb_dict = self.dense_head.get_loss()
#         tb_dict = {
#             'loss_rpn': loss_rpn.item(),
#             **tb_dict
#         }

#         loss = loss_rpn
#         return loss, tb_dict, disp_dict






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
