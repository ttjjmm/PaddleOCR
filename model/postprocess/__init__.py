import copy

__all__ = ['build_postprocess']



def build_postprocess(config, global_config=None):

    from .db_postprocess import DBPostProcess
    from .cls_postprocess import ClsPostProcess

    config = copy.deepcopy(config)
    module_name = config.pop('name')
    if global_config is not None:
        config.update(global_config)

    module_class = eval(module_name)(**config)

    return module_class

