from functools import partial

from torch.utils.data import DataLoader

from ..utils.utils import worker_init_fn
from .cifar10.cifar10_dataset import Cifa10Dataset

__all__ = {'Cifa10Dataset': Cifa10Dataset, 'ImagenetDataset': ImagenetDataset}


def build_dataloader(dataset_cfg, class_names=None, root_path=None, training=False):
    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=dataset_cfg.BATCH_SIZE,
        pin_memory=dataset_cfg.TRUE,
        num_workers=dataset_cfg.WORKERS,
        shuffle=training,
        collate_fn=dataset.collate_batch,
        drop_last=False,
        sampler=None,
        timeout=0,
        worker_init_fn=partial(worker_init_fn, seed=666),
    )

    return dataset, dataloader
