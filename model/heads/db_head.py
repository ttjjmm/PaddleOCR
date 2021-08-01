import torch
import torch.nn as nn
import torch.nn.functional as F


class SubHead(nn.Module):
    def __init__(self, in_channels):
        super(SubHead, self).__init__()

        self.conv1 =  nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels // 4,
                kernel_size=(3, 3),
                padding=(1, 1),
                bias=False)
        self.conv_bn1 = nn.BatchNorm2d(in_channels // 4)

        self.conv2 = nn.ConvTranspose2d(
                in_channels=in_channels // 4,
                out_channels=in_channels // 4,
                kernel_size=(2, 2),
                stride=(2, 2),
                bias=False)
        self.conv_bn2 = nn.BatchNorm2d(in_channels // 4)

        self.conv3 =  nn.ConvTranspose2d(
                in_channels=in_channels // 4,
                out_channels=1,
                kernel_size=(2, 2),
                stride=(2, 2))

    def forward(self, x):
        x = self.conv_bn1(self.conv1(x))
        x = F.relu(x, inplace=True)
        x = self.conv_bn2(self.conv2(x))
        x = F.relu(x, inplace=True)
        x = F.sigmoid(self.conv3(x))
        return x


class DBHead(nn.Module):
    """
        Differentiable Binarization (DB) for text detection:
            see https://arxiv.org/abs/1911.08947
        args:
            params(dict): super parameters for build DB network
    """
    def __init__(self, in_channels, k=50, **kwargs):
        super(DBHead, self).__init__()
        self.k = k
        self.binarize = SubHead(in_channels)
        self.thresh = SubHead(in_channels)

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def forward(self, x):
        shrink_maps = self.binarize(x)
        if not self.training:
            return {'maps': shrink_maps}
        threshold_maps = self.thresh(x)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        y = torch.cat([shrink_maps, threshold_maps, binary_maps], dim=1)
        return {'maps': y}




if __name__ == '__main__':
    pass



