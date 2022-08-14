#THIS ONE
"""
    This script implement ghost module from 
    "GhostNet: More Features from Cheap Operations"
    https://arxiv.org/pdf/1911.11907.pdf
    Introduction in:
    https://owen-liuyuxuan.github.io/papers_reading_sharing.github.io/Building_Blocks/GhostNet/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


class GhostModule(nn.Module):
    """
        Ghost Module from https://github.com/iamhankai/ghostnet.pytorch.

    """
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
        f.write("GhostModule_init \n")
        f.close()
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio) #math.ceil returns the ceiling of whatever's inside as an integer
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.AvgPool2d(stride) if stride > 1 else nn.Sequential(), #perform average pooling if stride is bigger than 1 otherwise conv with specified parameters
            nn.Conv2d(inp, init_channels, kernel_size, 1, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential( # this is the additional part of GhostNet, introduces cheap operations to add more features without heavy computation
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
        f.write("GhostModule_forward \n")
        f.close()
        print("initial shape of input into ghost module x", x.shape)
        x1 = self.primary_conv(x)
        print("shape of x1 after primary conv", x1.shape)
        x2 = self.cheap_operation(x1)
        print("shape of x2 after Ghost Cheap Operation", x2.shape)
        out = torch.cat([x1,x2], dim=1)
        print("shape of concactenated Ghost output", out.shape)
        return out[:,:self.oup,:,:]

class ResGhostModule(GhostModule):
    """Some Information about ResGhostModule"""
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, relu=True, stride=1):
        f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
        f.write("ResGhostModule_init \n")
        f.close()
        assert(ratio > 2)
        super(ResGhostModule, self).__init__(inp, oup-inp, kernel_size, ratio-1, dw_size, relu=relu, stride=stride)
        self.oup = oup
        if stride > 1:
            self.downsampling = nn.AvgPool2d(kernel_size=stride, stride=stride) #same as before, if stride is more than 1 downsample
        else:
            self.downsampling = None

    def forward(self, x):
        f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
        f.write("ResGhostModule_forward \n")
        f.close()
        print("initial shape of input into ResGhost module x", x.shape)
        x1 = self.primary_conv(x)
        print("shape of x1 after primary conv in ResGhost", x1.shape)
        x2 = self.cheap_operation(x1)
        print("shape of x2 after ResGhost Cheap Operation", x2.shape)

        if not self.downsampling is None:
            x = self.downsampling(x)
            print("initial shape of input iafter downsampling x", x.shape)
        out = torch.cat([x, x1, x2], dim=1)
        print("shape of concactenated ResGhost output", out.shape)
        return out[:,:self.oup,:,:]
