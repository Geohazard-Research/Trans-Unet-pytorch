import torch
import torch.nn as nn

from nets.resnet import ResNet3, Bottleneck


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class Unet3(nn.Module):
    def __init__(self, block=Bottleneck, layers=[2]):
        super(Unet3, self).__init__()

        self.resnet = ResNet3(block, layers)

        in_filters = [768]

        out_filters = [320]


        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

    def forward(self, inputs):
        [feat1, feat2] = self.resnet.forward(inputs)

        up1 = self.up_concat1(feat1, feat2)

        return up1

