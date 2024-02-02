import logging
import os

from ..datasets import build_dataloader
from ..model import build_model
from ..utils.utils import get_cfg, parse_config
import torch

logger = logging.getLogger("vit.train")


def main():
    args = parse_config()
    cfg = get_cfg(args)
    logger.info("*******ARGS*******")
    logger.info(f"args : {args}")

    gpu_list = (
        os.environ["CUDA_VISIBLE_DEVICES"]
        if "CUDA_VISIBLE_DEVICES" in os.environ.keys()
        else "ALL"
        if torch.cuda.is_available()
        else "CPU"
    )
    logger.info("*******GPU ENV*******")

    logger.info(f"CUDA DEVICES={gpu_list}")

    logger.info("*******DATA LOADER*******")
    train_set, data_loader = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=cfg.ROOT_PATH,
        training=True,
    )

    logger.info("*******MAKE MODEL*******")
    model = build_model(model_cfg=cfg.MODEL, num_class=num_class, dataset=train_set)


if __name__ == "__name__":
    main()
