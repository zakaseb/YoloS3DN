#THIS ONE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', '/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/')))
from visualDet3D.networks.lib.blocks import AnchorFlatten, ConvBnReLU
from visualDet3D.networks.lib.ghost_module import ResGhostModule, GhostModule
from visualDet3D.networks.lib.PSM_cost_volume import PSMCosineModule, CostVolume
import EdgeNeXt
from EdgeNeXt.models.edgenext import EdgeNeXt
from EdgeNeXt.models.model import edgenext_xx_small, edgenext_small, edgenext_small_bn_hs, edgenext
from visualDet3D.networks.backbones import resnet
from visualDet3D.networks.backbones.resnet import BasicBlock
from visualDet3D.networks.lib.look_ground import LookGround

class CostVolumePyramid(nn.Module):
    """Some Information about CostVolumePyramid"""
    def __init__(self, depth_channel_4, depth_channel_8, depth_channel_16):
        f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
        f.write("CostVolumePyramid_Init \n")
        f.close()
        super(CostVolumePyramid, self).__init__()
        self.depth_channel_4  = depth_channel_4 # 4, 8, 16 are number of channels # 24 <= unclear what this number means 
        self.depth_channel_8  = depth_channel_8 # 24
        self.depth_channel_16 = depth_channel_16 # 96

        input_features = depth_channel_4 # 24
        
        self.four_to_eight = nn.Sequential(
            ResGhostModule(input_features, 3 * input_features, 3, ratio=3), # input is input_features and output is 3*input_features, kernel size of 3
            nn.AvgPool2d(2), # kernel size of 2?
            #nn.Conv2d(3 * input_features, 3 * input_features, 3, padding=1, bias=False),
            #nn.BatchNorm2d(3 * input_features),
            #nn.ReLU(),
            BasicBlock(3 * input_features, 3 * input_features), #3* input_features is input and 3*input_channels is planes (number of feature maps, revert to resnet)
        )
        # import pdb
        # pdb.set_trace()
        input_features = 3 * input_features + depth_channel_8 # 3 * 24 + 24 = 96
        self.eight_to_sixteen = nn.Sequential(
            ResGhostModule(input_features, 3 * input_features, 3, ratio=3),
            nn.AvgPool2d(2),
            BasicBlock(3 * input_features, 3 * input_features),
            #nn.Conv2d(3 * input_features, 3 * input_features, 3, padding=1, bias=False),
            #nn.BatchNorm2d(3 * input_features),
            #nn.ReLU(),
        )
        # import pdb
        # pdb.set_trace()        
        input_features = 3 * input_features + depth_channel_16 # 3 * 96 + 96 = 384 #number of channels
        # import pdb
        # pdb.set_trace()          
        self.depth_reason = nn.Sequential(
            ResGhostModule(input_features, 3 * input_features, kernel_size=3, ratio=3),
            BasicBlock(3 * input_features, 3 * input_features),
            #nn.Conv2d(3 * input_features, 3 * input_features, 3, padding=1, bias=False),
            #nn.BatchNorm2d(3 * input_features),
            #nn.ReLU(),
        )
        # import pdb
        # pdb.set_trace()          
        self.output_channel_num = 3 * input_features #1152 #number of channels of output
        # import pdb
        # pdb.set_trace()  

        self.depth_output = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.output_channel_num, int(self.output_channel_num/2), 3, padding=1), # output_channel_num is number of input channels, utput_channel_num/2 is for output, kernel size 3
            nn.BatchNorm2d(int(self.output_channel_num/2)), # int(self.output_channel_num/2) is number of features to be downsampled
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(int(self.output_channel_num/2), int(self.output_channel_num/4), 3, padding=1), # int(self.output_channel_num/2) is input and int(self.output_channel_num/4) is output
            nn.BatchNorm2d(int(self.output_channel_num/4)), # int(self.output_channel_num/4) is number of features to be downsampled
            nn.ReLU(),
            nn.Conv2d(int(self.output_channel_num/4), 96, 1), # int(self.output_channel_num/2) is number of channels for input, 96 is number of output channels, kernel size 1
        )
        # import pdb
        # pdb.set_trace()  

    def forward(self, psv_volume_4, psv_volume_8, psv_volume_16):
        f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
        f.write("CostVolumePyramid_Forward \n")
        f.close()
        # import pdb
        # pdb.set_trace()        
        psv_4_8 = self.four_to_eight(psv_volume_4) # building the pyramid cost volume for each stage 4, 8 and 16 at a time onwards, starting with applying REsGhost, then Avg Pool the Basic block
        # import pdb
        # pdb.set_trace()         
        # print(psv_4_8.shape)
        psv_volume_8 = torch.cat([psv_4_8, psv_volume_8], dim=1) # same goes for the rest
        # import pdb
        # pdb.set_trace() 
        # print(psv_volume_8.shape)
        psv_8_16 = self.eight_to_sixteen(psv_volume_8)
        # import pdb
        # pdb.set_trace() 
        # print(psv_8_16.shape)
        psv_volume_16 = torch.cat([psv_8_16, psv_volume_16], dim=1)
        # import pdb
        # pdb.set_trace() 
        # print(psv_volume_16.shape)
        psv_16 = self.depth_reason(psv_volume_16) # this is done without downsampling
        # import pdb
        # pdb.set_trace() 
        # print(psv_16.shape)
        if self.training:
            return psv_16, self.depth_output(psv_16) #upsample then conv
            # import pdb
            # pdb.set_trace()

        # import pdb
        # pdb.set_trace()  

        return psv_16, torch.zeros([psv_volume_4.shape[0], 1, psv_volume_4.shape[2], psv_volume_4.shape[3]])

class StereoMerging(nn.Module):
    def __init__(self, base_features):
        f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
        f.write("Stereomerging_init \n")
        f.close()
        super(StereoMerging, self).__init__()
        # import pdb
        # pdb.set_trace()
        self.cost_volume_0 = PSMCosineModule(downsample_scale=4, max_disp=96, input_features=base_features) #caculating the cost volume for each stage using PSM as per the .py doc, this is for the purpose of stereo matching
        PSV_depth_0 = self.cost_volume_0.depth_channel #for said specific channel? note that downsample_scale 4, 8, 16 is the same as depth_channel for the previous stages aforementionned 

        self.cost_volume_1 = PSMCosineModule(downsample_scale=8, max_disp=192, input_features=base_features * 2)
        PSV_depth_1 = self.cost_volume_1.depth_channel

        self.cost_volume_2 = CostVolume(downsample_scale=16, max_disp=192, input_features=base_features * 4, PSM_features=8)
        PSV_depth_2 = self.cost_volume_2.output_channel

        self.depth_reasoning = CostVolumePyramid(PSV_depth_0, PSV_depth_1, PSV_depth_2) # the combination of the previous 3 stages but manner how it is combined is unknwon
        self.final_channel = self.depth_reasoning.output_channel_num + base_features * 4

    def forward(self, left_x, right_x):
        f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
        f.write("Stereomerging_forward \n")
        f.close()
        # print("shape of x_left in forward in core", np.shape(left_x))
        # print("shape of x_right in forward in core", np.shape(right_x))
        # import pdb
        # pdb.set_trace()
        PSVolume_0 = self.cost_volume_0(left_x[0], right_x[0]) # calculates cost volume of respective right and left feature maps for each given stage with the the previously designed parameters ( max_disp, downsample scale, input_features= base_features*.., PSM_features)
        PSVolume_1 = self.cost_volume_1(left_x[1], right_x[1])
        # import pdb
        # pdb.set_trace()        
        PSVolume_2 = self.cost_volume_2(left_x[2], right_x[2])
        # import pdb
        # pdb.set_trace()            
        PSV_features, depth_output = self.depth_reasoning(PSVolume_0, PSVolume_1, PSVolume_2) # c = 1152
        # import pdb
        # pdb.set_trace()
        features = torch.cat([left_x[2], PSV_features], dim=1) # c = 1152 + 256 = 1408
        # import pdb
        # pdb.set_trace()
        # print("the shape of depth_output in core", depth_output.shape)
        # print("the shape of features in core", features.shape)
        return features, depth_output

class YoloStereo3DCore(nn.Module):
    """
        Inference Structure of YoloStereo3D
        Similar to YoloMono3D,
        Left and Right image are fed into the backbone in batch. So they will affect each other with BatchNorm2d.
    """
    def __init__(self, backbone_arguments):
        f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
        f.write("yolosterero3dCore_init \n")
        f.close()
        super(YoloStereo3DCore, self).__init__()
        # EdgeN = EdgeNeXt(depths=[3, 3, 9, 3], dims=[48, 96, 160, 304], expan_ratio=4,
        #              global_block=[0, 1, 1, 1],
        #              global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
        #              use_pos_embd_xca=[False, True, False, False],
        #              kernel_sizes=[3, 5, 7, 9],
        #              d2_scales=[2, 2, 3, 4],
        #              classifier_dropout=0.0) #Edgnext Small
        self.backbone = edgenext(**backbone_arguments) # Resnet, change backbone from here

        base_features = 256 #if backbone_arguments['depth'] > 34 else 64 # meaning which depth of resnet
        self.neck = StereoMerging(base_features) #stereomerging outputs features and depth output.


    def forward(self, images):
        f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
        f.write("yolosterero3dCore_forward \n")
        f.close()
        batch_size = images.shape[0]
        left_images = images[:, 0:3, :, :]
        right_images = images[:, 3:, :, :]

        # import pdb
        # pdb.set_trace()

        images = torch.cat([left_images, right_images], dim=0) #concatenate left and right images

        # import pdb
        # pdb.set_trace()        
        f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
        f.write("yolosterero3dCore_forwrd_BeforeCalling_Backbone \n")
        f.close()
        features = self.backbone(images) #applying backbone on images, all of resnet applied here
        # import pdb
        # pdb.set_trace()           
        # n = len(features)
        # for i in range (0, n):
        #     print("features", features[i].shape)
        # import pdb
        # pdb.set_trace()   
        # print(features.size)
        left_features  = [feature[0:batch_size] for feature in features]# for left it fills "feature" from 0 to batch_size and and then from batch size till then end for right images
        right_features = [feature[batch_size:]  for feature in features]
        # for i in range(0,n):
        #     print("left features" , left_features[i].shape)
        #     print("right features" , right_features[i].shape)        
        # import pdb
        # pdb.set_trace()
        features, depth_output = self.neck(left_features, right_features) #applying stereomerging between left features and right features
        # import pdb
        # pdb.set_trace()        
        # print("shape of depth output on forward pass in yolostereo3d", depth_output.shape)
        # print("shape of depth output on forward pass in yolostereo3d", features.shape)

        output_dict = dict(features=features, depth_output=depth_output) #storing the features and depth output to a new dictionary
        # import pdb
        # pdb.set_trace()        
        return output_dict


