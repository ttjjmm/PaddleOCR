
__all__ = ['build_neck']


def build_neck(config):
    from .db_fpn import DBFPN
    from .rnn import SequenceEncoder
    module_name = config.pop('name')
    module_class = eval(module_name)(**config)
    return module_class
