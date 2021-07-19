
__all__ = ['build_neck']


def build_neck(config):
    from .db_fpn import DBFPN
    name = config.pop('name')
    module_class = DBFPN(**config)
    return module_class
