from .vit import ViT

__all__ = {
    'ViT': ViT,
    # swinVit
}


def build_model(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
