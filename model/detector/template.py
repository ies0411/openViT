import torch
import torch.nn as nn

from ..backbone import embedding
import ..backbone


class DetectorTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.module_topology = [
          'embedding', 'backbone', 'head'
            # 'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            # 'backbone_2d', 'dense_head',  'point_head', 'roi_head'
        ]

        # self.embedding = EmbeddingMudule(
        #     self.cfg.PATCH_SIZE, self.cfg.PATCH_NUM, self.cfg.LATENT_DIMENSION
        # )
        # self.transformer = TransformerEncoder()

        # self.class_names = dataset.class_names
        # self.register_buffer('global_step', torch.LongTensor(1).zero_())

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def build_embedding(self, model_info_dict):
        if self.model_cfg.get('EMBEDDING', None) is None:
            return None, model_info_dict

        embedding_module = embedding.__all__[self.model_cfg.EMBEDDING.NAME](
            model_cfg=self.model_cfg.EMBEDDING,
            # num_point_features=model_info_dict['num_rawpoint_features'],
            # point_cloud_range=model_info_dict['point_cloud_range'],
            # voxel_size=model_info_dict['voxel_size'],
            # grid_size=model_info_dict['grid_size'],
            # depth_downsample_factor=model_info_dict['depth_downsample_factor']
        )
        # model_info_dict['num_point_features'] = vfe_module.get_output_feature_dim()
        model_info_dict['module_list'].append(embedding_module)
        return embedding_module, model_info_dict

    def build_networks(self):
        model_info_dict = {
            'module_list': [],
            # 'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,
            # 'num_point_features': self.dataset.point_feature_encoder.num_point_features,
            # 'grid_size': self.dataset.grid_size,
            # 'point_cloud_range': self.dataset.point_cloud_range,
            # 'voxel_size': self.dataset.voxel_size,
            # 'depth_downsample_factor': self.dataset.depth_downsample_factor
        }
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    def build_backbone(self, model_info_dict):
        if self.model_cfg.get('BACKBONE', None) is None:
            return None, model_info_dict

        backbone_module = backbone.__all__[self.model_cfg.BACKBONE.NAME](
            model_cfg=self.model_cfg.BACKBONE_3D,
            # input_channels=model_info_dict['num_point_features'],
            # grid_size=model_info_dict['grid_size'],
            # voxel_size=model_info_dict['voxel_size'],
            # point_cloud_range=model_info_dict['point_cloud_range']
        )
        model_info_dict['module_list'].append(backbone_module)
        model_info_dict['num_point_features'] = backbone_module.num_point_features
        model_info_dict['backbone_channels'] = backbone_module.backbone_channels \
            if hasattr(backbone_module, 'backbone_channels') else None
        return backbone_module, model_info_dict