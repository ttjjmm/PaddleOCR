import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CTCHead, self).__init__()
        self.fc = nn.Linear(
            in_channels,
            out_channels)
        self.out_channels = out_channels

    def forward(self, x):
        predicts = self.fc(x)
        if not self.training:
            predicts = F.softmax(predicts, dim=2)
        return predicts














