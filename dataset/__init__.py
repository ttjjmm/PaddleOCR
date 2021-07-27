import copy
from torch.utils.data import DataLoader
from .simple_dataset import SimpleDataSet

def build_dataloader(config, mode, logger, seed=None):
    config = copy.deepcopy(config)

    support_dict = ['SimpleDataSet', 'LMDBDataSet', 'PGDataSet']
    module_name = config[mode]['dataset']['name']

    assert module_name in support_dict, Exception('DataSet only support {}'.format(support_dict))
    assert mode in ['Train', 'Eval', 'Test'], 'Mode should be Train, Eval or Test.'
    dataset = eval(module_name)(config, mode, logger, seed)
    loader_config = config[mode]['loader']
    batch_size = loader_config['batch_size']
    drop_last = loader_config['drop_last']
    shuffle = loader_config['shuffle']
    num_workers = loader_config['num_workers']
    pin_memory = loader_config['pin_memory']

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        drop_last=drop_last,
                        pin_memory=pin_memory,
                        collate_fn=dataset.collate_fn)

    return loader













