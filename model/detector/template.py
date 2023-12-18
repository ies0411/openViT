import torch
import torch.nn as nn


class DetectorTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'dense_head',  'point_head', 'roi_head'
        ]

        @property
        def mode(self):
            return 'TRAIN' if self.training else 'TEST'
