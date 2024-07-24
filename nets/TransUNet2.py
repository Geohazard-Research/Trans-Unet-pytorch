# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import  mit_b0
from .unet import Unet3, unetUp
from nets.resnet import  Bottleneck
from nets.UNet_3Plus  import UNet_3Plus1, UNet_3Plus2


class MLP(nn.Module):

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    
class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class TransUNet_Head(nn.Module):

    def __init__(self, num_classes=2, in_channels=[64, 128, 320, 512], dropout_ratio=0.1, block=Bottleneck,):
        super(TransUNet_Head, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=c4_in_channels)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=c3_in_channels)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=c2_in_channels)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=c1_in_channels)


        self.linear_pred    = nn.Conv2d(256, num_classes, kernel_size=1)
        self.dropout        = nn.Dropout2d(dropout_ratio)
        self.Unet1 = UNet_3Plus1(64)
        self.Unet2 = UNet_3Plus2(128)
        self.Unet3 = Unet3(block)

        in_filters = [640, 896, 832]

        out_filters = [256, 384, 512]

        self.up_concat3 = unetUp(in_filters[2], out_filters[2])

        self.up_concat2 = unetUp(in_filters[1], out_filters[1])

        self.up_concat1 = unetUp(in_filters[0], out_filters[0])


    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        n, _, h, w = c4.shape


        c3 = self.Unet3(c3)

        up3 = self.up_concat3(c3, c4)

        c2 = self.Unet2(c2)

        up2 = self.up_concat2(c2, up3)

        c1 = self.Unet1(c1)

        up1 = self.up_concat1(c1, up2)

        x = self.dropout(up1)

        x = self.linear_pred(x)

        return x

class TransUNet2(nn.Module):
    def __init__(self, num_classes = 2, phi = 'b0', pretrained = False):
        super(TransUNet2, self).__init__()
        self.in_channels = {
            'b0': [64, 128, 320, 512],
        }[phi]
        self.backbone   = {
            'b0': mit_b0,
        }[phi](pretrained)

        self.decode_head = TransUNet_Head(num_classes, self.in_channels)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        
        x = self.backbone.forward(inputs)
        x = self.decode_head.forward(x)
        
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x
