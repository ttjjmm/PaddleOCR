import copy
import torch

class Trainer(object):
    def __init__(self):
        self.model = None


    def _init_optimizer(self, cfg_optim):
        config = copy.deepcopy(cfg_optim)
        name = config.pop('name')
        Optimizer = getattr(torch.optim, name)
        self.optimizer = Optimizer(params=self.model.parameters(), **config)

    def _init_scheduler(self, ):
        pass