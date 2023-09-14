import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from ..dataset import DatasetTemplate


class Cifa10Dataset(Dataset):
    def __init__(self, dataset_cfg, class_names, root_path, training):
        self.patch_size = dataset_cfg.patch_size
        self.image_size = dataset_cfg.image_size
        self.batch_size = dataset_cfg.batch_size
        self.train_set = None
        self.test_set = None
        self.class_names = class_names
        self.labels = None
        self.path = root_path
        self.training = training

    def prepare_data(self):
        self.train_set = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True
        ).data
        self.labels = trainset.targets

        self.test_set = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True
        ).data

        self.class_names = self.train_set.classes

    def augmentor(self, data):
        print("todo")

    def __len__(self):
        return self.train_set.shape[0]

    def __getitem__(self, index):
        input_data = self.train_set[index]
        label = self.labels[index]

        return input_data, label
