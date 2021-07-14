

__all__ = ['build_backbone']


def build_backbone(cfg, mtype):
    from .mobilenet_v3 import MobileNetV3

    model = MobileNetV3(**cfg)
    return model