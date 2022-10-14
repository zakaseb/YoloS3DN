from typing import Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import torch.utils.model_zoo as model_zoo
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', '/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/')))
from visualDet3D.networks.utils.registry import BACKBONE_DICT


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
    f.write("conv3x3 \n")
    f.close()
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, # in/out_planes: I/O, padding is dilation, kernel size of 3 therefore conv3x3
                     padding=dilation, bias=False, dilation=dilation) 


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
        f.write("BasicBlock_Init \n")
        f.close()
        super(BasicBlock, self).__init__() # reserved method to initialise attributes without calling base class
        self.conv1 = conv3x3(inplanes, planes, stride) #first 3x3 conv with I/O:Inplanes, planes and strides)
        self.bn1 = nn.BatchNorm2d(planes) #first batch norm of output of 1st conv
        self.relu = nn.ReLU(inplace=True) # 1st activation fct ReLU
        self.conv2 = conv3x3(planes, planes, dilation=dilation) # 2nd 3x3 conv
        self.bn2 = nn.BatchNorm2d(planes)# 2nd batch norm
        self.downsample = downsample 
        self.stride = stride

    def forward(self, x):
        f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
        f.write("BasicBlock_Forward \n")
        f.close()
        # print("shape of initial input basicblock", x.shape)
        residual = x
        # print("the shape of initial basicblock", x.shape)

        # print("shape of 1st input basicblock", x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print("shape of 1st output basicblock", out.shape) # first block of ResNet consisting of conv, batchnorm,relu. 

        # print("shape of 2nd input basicblock", out.shape)
        out = self.conv2(out) #2nd 3x3 conv
        out = self.bn2(out) #2nd batchnorm
        # print("shape of 2nd output basicblock", out.shape)

        if self.downsample is not None: # if downsampling is non-zero
            # print("shape before downsample basicblock", x.shape)
            residual = self.downsample(x) # downsampling of x is assigned to residual
            # print("shape after downsample basicblock", residual.shape)

        # print("shape of 3rd input basicblock", out.shape)
        out += residual #output of last downsampling output is assigned to out + out (i.e skip connection)
        out = self.relu(out) # activation function ReLU
        # print("shape of 3rd output basicblock", out.shape)

        return out


class Bottleneck(nn.Module):
    expansion = 4 # expansion parameter for number of output channels

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
        f.write("Bottleneck_init \n")
        f.close()
        super(Bottleneck, self).__init__() # used to give accesss to parent or sibling class, returns objects representing parent class
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride # same commments a previous class

    def forward(self, x):
        f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
        f.write("Bottleneck_Forward \n")
        f.close()
        residual = x
        # print("initial output of bottleneck", residual.shape)

        # print("1st input of bottleneck", out.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print("1st output of bottleneck", out.shape)


        # print("2nd input of bottleneck", out.shape)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # print("2nd output of bottleneck", out.shape)

        # print("3rd input of bottleneck", out.shape)
        out = self.conv3(out)
        out = self.bn3(out)
        # print("3rd output of bottleneck", out.shape)

        if self.downsample is not None:
            # print("shape before downsample bottleneck", x.shape)
            residual = self.downsample(x)
            # print("shape before downsample bottleneck", residual.shape)

        # print("4th input after downsample bottleneck", residual.shape)
        out += residual
        out = self.relu(out)
        # print("4th output shape after activation bottlneck", out.shape)

        return out


class ResNet(nn.Module):
    planes = [64, 128, 256, 512] # Dims to be modified?- output channels

    def __init__(self, block: Union[BasicBlock, Bottleneck],
                 layers: Tuple[int, ...],
                 num_stages: int = 4,
                 strides: Tuple[int, ...] = (1, 2, 2, 2),
                 dilations: Tuple[int, ...] = (1, 1, 1, 1),
                 out_indices: Tuple[int, ...] = (-1, 0, 1, 2, 3),
                 frozen_stages: int = -1,
                 norm_eval: bool = True, # batch normalization stats are not tracked
                 ):
        f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
        f.write("Resnet_init \n")
        f.close()
        self.inplanes = 64 #input channels
        super(ResNet, self).__init__() #initiate resNet attributes

        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4 # this is to make sure the number of stages is not smaller than 1 or bigger than 4. otherwise assertionerror
        self.strides = strides
        self.dilations = dilations
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        assert max(out_indices) < num_stages

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) # 3 is input channels, 64, is output channels
        self.bn1 = nn.BatchNorm2d(64) # 64 is number of features
        self.relu = nn.ReLU(inplace=True) # inplace operations can make changes on a tensor without making a copy of it, thus not occupying as much memory
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        for i in range(num_stages):
            setattr(self, f"layer{i + 1}", self._make_layer(block, self.planes[i], layers[i], stride=self.strides[i],
                                                            dilation=self.dilations[i])) # for each stage, assigns a bottleneck layer with aforementionned attributes ( not sure)

        for m in self.modules(): 
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d): # if is not a conv module (therefore a batchnorm module) then fill weights of said module and give 0 bias
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # prior = 0.01

        self.train()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
        f.write("Resnet_make_layer \n")
        f.close()
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion: # interger factor defining expansion of feature maps after a conv, planes is the number of feature maps(inplanes being the number of input feature maps), number of output feature maps are planes*expansion
            downsample = nn.Sequential( # if conditions above are met, stride unequel to 1 and .. then downsample
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample)) #appending components to make one layer 
        self.inplanes = planes * block.expansion #input feature maps are number of feature maps of a conv layer times the expansion rate
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def train(self, mode=True):
        f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
        f.write("Resnet_Train \n")
        f.close()
        super(ResNet, self).train(mode)

        if mode:
            self.freeze_stages()
            if self.norm_eval:
                self.freeze_bn()

    def freeze_stages(self):
        f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
        f.write("Resnet_freeze_stages \n")
        f.close()
        if self.frozen_stages >= 0:
            self.conv1.eval() # entering conv and bn layers in evaluation mode (turned off)
            self.bn1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False # stoping gradient descent of params of said conv 

            for param in self.bn1.parameters():
                param.requires_grad = False # stoping gradient descent of params of said bn 

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False # parameters in frozen stages are not updated 

    def freeze_bn(self):
        f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
        f.write("Resnet_Freese_bn \n")
        f.close()
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):  # Will freeze both batchnorm and sync batchnorm
                layer.eval()

    def forward(self, img_batch):
        f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
        f.write("Resnet_Forward \n")
        f.close()
        outs = []
        # print("shape of img_batch before conv1", len(outs))
        x = self.conv1(img_batch)
        # print("shape of img_batch after conv1", x.shape)
        x = self.bn1(x)
        # print("shape of img_batch after bn1", x.shape)
        x = self.relu(x)
        # print("shape of img_batch after relu", x.shape)
        if -1 in self.out_indices:
            outs.append(x) #output of one block of conv bn and relu into 1 list
        x = self.maxpool(x)
        # print("shape of img_batch after maxpool", x.shape)
        for i in range(self.num_stages): #unclear
            layer = getattr(self, f"layer{i + 1}") # high level, this for loop gathers the attributes of one layer (mentionned above) and assigns it to layer and projects it onto x
            x = layer(x)
            # print("appended output for the entire layer:", x.shape)
            if i in self.out_indices:
                outs.append(x) # saving different feature maps from different stages 

        #print(x.shape)
        # print("final output outs", len(outs))
        return outs


def resnet18(pretrained=True, **kwargs):
    f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
    f.write("Resnet18 \n")
    f.close()
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnet34(pretrained=True, **kwargs):
    f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
    f.write("Resnet34 \n")
    f.close()
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnet50(pretrained=True, **kwargs):
    f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
    f.write("Resnet50 \n")
    f.close()
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def resnet101(pretrained=True, **kwargs):
    f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
    f.write("Resnet101 \n")
    f.close()
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnet152(pretrained=True, **kwargs):
    f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
    f.write("Resnet152 \n")
    f.close()
    """Constructs a ResNet-152 moPIPELINE_DICTdel.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model


@BACKBONE_DICT.register_module
def resnet(depth, **kwargs):
    if depth == 18:
        model = resnet18(**kwargs)
    elif depth == 34:
        model = resnet34(**kwargs)
    elif depth == 50:
        model = resnet50(**kwargs)
    elif depth == 101:
        model = resnet101(**kwargs)
    elif depth == 152:
        model = resnet152(**kwargs)
    else:
        raise ValueError(
            'Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    return model


if __name__ == '__main__':
    model = resnet34(False).cuda()
    model.eval()
    image = torch.rand(2, 3, 224, 224).cuda()
    print("image shape", image.shape)

    output = model(image)
    print("shape of output of model in main", len(output))
    print(model)

    #print(len(output))
