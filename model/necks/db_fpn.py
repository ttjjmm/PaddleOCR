import torch
import torch.nn as nn
import torch.nn.functional as F


class DBFPN(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(DBFPN, self).__init__()
        self.out_channels = out_channels
        # weight_attr = paddle.nn.initializer.KaimingUniform()

        self.in2_conv = nn.Conv2d(
            in_channels=in_channels[0],
            out_channels=self.out_channels,
            kernel_size=(1, 1),
            bias=False)
        self.in3_conv = nn.Conv2d(
            in_channels=in_channels[1],
            out_channels=self.out_channels,
            kernel_size=(1, 1),
            bias=False)
        self.in4_conv = nn.Conv2d(
            in_channels=in_channels[2],
            out_channels=self.out_channels,
            kernel_size=(1, 1),
            bias=False)
        self.in5_conv = nn.Conv2d(
            in_channels=in_channels[3],
            out_channels=self.out_channels,
            kernel_size=(1, 1),
            bias=False)
        self.p5_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=False)
        self.p4_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=False)
        self.p3_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=False)
        self.p2_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=False)


    def _init_weights(self):
        pass


    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.in5_conv(c5)
        in4 = self.in4_conv(c4)
        in3 = self.in3_conv(c3)
        in2 = self.in2_conv(c2)
        # defult nearest
        out4 = in4 + F.interpolate(in5, scale_factor=2)  # 1/16
        out3 = in3 + F.interpolate(out4, scale_factor=2)  # 1/8
        out2 = in2 + F.interpolate(out3, scale_factor=2)  # 1/4

        p5 = self.p5_conv(in5)
        p4 = self.p4_conv(out4)
        p3 = self.p3_conv(out3)
        p2 = self.p2_conv(out2)
        p5 = F.interpolate(p5, scale_factor=8)
        p4 = F.interpolate(p4, scale_factor=4)
        p3 = F.interpolate(p3, scale_factor=2)

        fuse = torch.cat([p5, p4, p3, p2], dim=1)
        return fuse



if __name__ == '__main__':
    from model.backbones.mobilenet_v3 import MobileNetV3
    net1 = MobileNetV3()
    net2 = DBFPN([16, 24, 56, 480], 96)
    inp = torch.randn((2, 3, 320, 320))
    out1 = net1(inp)
    print(net2(out1).shape)