import torchvision
from torch.utils.data import Dataset

from ..utils.data_processing import generate_patch


class Cifa10Dataset(Dataset):
    def __init__(self, dataset_cfg, class_names, root_path, training):
        self.patch_size = dataset_cfg.patch_size
        self.image_size = dataset_cfg.image_size
        self.batch_size = dataset_cfg.batch_size
        self.cfg = dataset_cfg
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
        if self.training:
            input_data = self.train_set[index]
            label = self.labels[index]
        else:
            input_data = self.test_set[index]
            label = None

        if self.cfg.PATCH:
            generate_patch(self.cfg.PATCH.PATCH_SIZE, input_data)

        return input_data, label
