

__all__ = ['build_backbone']


def build_backbone(config, model_type):
    name = config.pop('name')
    if model_type == 'det':
        from .mobilenet_v3 import MobileNetV3
    else:
        raise NotImplementedError
    model = MobileNetV3(**config)
    return model