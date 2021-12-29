

__all__ = ['build_backbone']


def build_backbone(config, model_type):
    name = config.pop('name')
    if model_type == 'det':
        from .det_mbnetv3 import MobileNetV3
    elif model_type == "rec" or model_type == "cls":
        from .rec_mbnetv3 import MobileNetV3
        from .rec_mv1_enhance import MobileNetV1Enhance
    else:
        raise NotImplementedError
    model = eval(name)(**config)
    return model