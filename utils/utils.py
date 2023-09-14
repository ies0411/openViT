import argparse
import logging

import yaml

logger = logging.getLogger('vit.utils')


def parse_config():
    parser = argparse.ArgumentParser(description='ViT')
    parser.add_argument(
        '--cfg_file', type=str, default=None, help='specify the config for training'
    )
    parser.add_argument(
        '--ckpt', type=str, default=None, help='checkpoint to start from'
    )
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    args = parser.parse_args()
    return args


def get_cfg(args):
    with open(args.cfg_file, 'r') as f:
        try:
            config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.safe_load(f)
    return config


def worker_init_fn(worker_id, seed=666):
    if seed is not None:
        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)
        torch.cuda.manual_seed(seed + worker_id)
        torch.cuda.manual_seed_all(seed + worker_id)
