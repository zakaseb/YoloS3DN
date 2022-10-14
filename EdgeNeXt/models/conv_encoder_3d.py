import torch
from torch import nn
from timm.models.layers import DropPath
from layers import LayerNorm

class ConvEncoder3D(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=4, kernel_size=7, spatial_dims=3, dropout=0):
        super().__init__()
        self.dwconv = get_conv_layer(spatial_dims, dim, dim , kernel_size=(kernel_size, kernel_size, kernel_size), stride=(1,1,1), dropout=dropout, conv_only=True, groups=dim)
        self.norm = get_norm_layer(name="instance", spatial_dims=spatial_dims, channels=dim)
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        print(" The passed dim is ", dim)
    def forward(self, x):
        input = x
        #print("ConvEncoder3D: The shape of the input is ", x.shape)
        x = self.dwconv(x)
        #print("ConvEncoder3D: The shape of x after dw.conv is  ", x.shape)
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, H, W, D) -> (N, H, W, D, C)
        #print("ConvEncoder3D: The shape of x after permute  is  ", x.shape)
        x = self.norm(x)
        #print("ConvEncoder3D: The shape of x after norm is ", x.shape)
        x = self.pwconv1(x)
        #print("ConvEncoder3D: The shape of x after pwconv1 is ", x.shape)
        x = self.act(x)
        x = self.pwconv2(x)
        #print("ConvEncoder3D: The shape of x after pwconv2 is ", x.shape)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)  # (N, H, W, D, C) -> (N, C, H, W, D)
        #print("ConvEncoder3D: The shape of x after permute is ", x.shape)
        x = input + self.drop_path(x)
        return x