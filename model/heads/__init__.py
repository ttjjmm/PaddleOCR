

__all__ = ['build_head']


def build_head(config):
    from .db_head import DBHead
    from .cls_head import ClsHead
    from .ctc_head import CTCHead
    module_name = config.pop('name')

    module_class = eval(module_name)(**config)

    return module_class



