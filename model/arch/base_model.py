import torch
import torch.nn as nn

from model.backbones import build_backbone
from model.necks import build_neck
from model.heads import build_head


class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        # input image channels
        in_channels = config.get('in_channels', 3)
        model_type = config['model_type']
        config['Backbone']['in_channels'] = in_channels
        self.backbone = build_backbone(config['Backbone'], model_type)
        in_channels = self.backbone.out_channels
        if 'Neck' not in config or config['Neck'] is None:
            self.use_neck = False
        else:
            self.use_neck = True
            config['Neck']['in_channels'] = in_channels
            self.neck = build_neck(config['Neck'])
            in_channels = self.neck.out_channels
        config["Head"]['in_channels'] = in_channels
        self.head = build_head(config["Head"])


    def forward(self, x):
        x = self.backbone(x)
        if self.use_neck:
            x = self.neck(x)
        x = self.head(x)

        return x