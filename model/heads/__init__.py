

__all__ = ['build_head']


def build_head(config):
    from .db_head import DBHead
    name = config.pop('name')
    module_class = DBHead(**config)

    return module_class



