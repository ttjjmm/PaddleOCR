import copy

def build_loss(config):
    from .db_loss import DBLoss

    support_dict = [
        'DBLoss', 'EASTLoss', 'SASTLoss', 'CTCLoss', 'ClsLoss', 'AttentionLoss',
        'SRNLoss', 'PGLoss']

    config = copy.deepcopy(config)
    module_name = config.pop('name')
    assert module_name in support_dict, Exception('loss only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class