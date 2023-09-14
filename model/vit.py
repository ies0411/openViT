import torch
import torch.nn as nn


# FIXME : data aug, transform
class ViT(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        # self.class_names = dataset.class_names
        # self.register_buffer('global_step', torch.LongTensor(1).zero_())

    def preprocessing_data(self):
        print("todo")
