#THIS ONE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', '/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/')))
from visualDet3D.networks.utils.registry import DETECTOR_DICT
from visualDet3D.utils.timer import profile
from visualDet3D.networks.heads import losses
from visualDet3D.networks.detectors.yolostereo3d_core import YoloStereo3DCore
from visualDet3D.networks.heads.detection_3d_head import StereoHead
from visualDet3D.networks.lib.blocks import AnchorFlatten, ConvBnReLU
from EdgeNeXt.models import edgenext # new import for EdgeNext
from visualDet3D.networks.backbones.resnet import BasicBlock



@DETECTOR_DICT.register_module
class Stereo3D(nn.Module):
    """
        Stereo3D
    """
    def __init__(self, network_cfg):
        f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
        f.write("Stereo3D_init \n")
        f.close()
        super(Stereo3D, self).__init__()

        self.obj_types = network_cfg.obj_types

        self.build_head(network_cfg) 

        self.build_core(network_cfg) # look up difference between head and core in yolo

        self.network_cfg = network_cfg

    def build_core(self, network_cfg):
        f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
        f.write("Stereo3D_buildcore \n")
        f.close()
        self.core = YoloStereo3DCore(network_cfg.backbone) # calls for said class in yolostereo3d_core 

    def build_head(self, network_cfg):
        f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
        f.write("Stereo3D_build_head \n")
        f.close()
        self.bbox_head = StereoHead(
            **(network_cfg.head) # look up stereoHead in detection_3d_head.py 
        )

        self.disparity_loss = losses.DisparityLoss(maxdisp=96) #Disparity.Loss is from losses.py

    def train_forward(self, left_images, right_images, annotations, P2, P3, disparity=None):
        f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
        f.write("Stereo3D_train_forward \n")
        f.close()
        """
        Args:
            img_batch: [B, C, H, W] tensor # B: batchsize, C: number of channels/channel depth?H: Height, W: Width
            annotations: check visualDet3D.utils.utils compound_annotation # found in utils.py
            calib: visualDet3D.kitti.data.kitti.KittiCalib or anything with obj.P2
        Returns:
            cls_loss, reg_loss: tensor of losses
            loss_dict: [key, value] pair for logging
        """
        # import pdb
        # pdb.set_trace()
        output_dict = self.core(torch.cat([left_images, right_images], dim=1))
        # import pdb
        # pdb.set_trace()
        depth_output   = output_dict['depth_output']
        # import pdb
        # pdb.set_trace()
        # print("depth_output in yolostereo3d_core", len(depth_output))

        cls_preds, reg_preds = self.bbox_head( #class prediction, regression predictions
                dict(
                    features=output_dict['features'],
                    P2=P2, #dataset parameter of calibration
                    image=left_images
                )
            )
        # import pdb
        # pdb.set_trace()

        anchors = self.bbox_head.get_anchor(left_images, P2)

        # import pdb
        # pdb.set_trace()

        cls_loss, reg_loss, loss_dict = self.bbox_head.loss(cls_preds, reg_preds, anchors, annotations, P2) # loss function from 3d_head and stored in loss dict

        # import pdb
        # pdb.set_trace()

        if reg_loss.mean() > 0 and not disparity is None and not depth_output is None: #if mean regression loss is >0 and disparity loss is non null and there depth_output is non null 
            disp_loss = 1.0 * self.disparity_loss(depth_output, disparity)
            loss_dict['disparity_loss'] = disp_loss
            reg_loss += disp_loss #disaprity loss is being added to regression loss 
            # import pdb
            # pdb.set_trace()

            self.depth_output = depth_output.detach()

        else:
            loss_dict['disparity_loss'] = torch.zeros_like(reg_loss)

        return cls_loss, reg_loss, loss_dict

    def test_forward(self, left_images, right_images, P2, P3):
        f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
        f.write("Stereo3D_test_forward \n")
        f.close()
        assert left_images.shape[0] == 1 # we recommmend image batch size = 1 for testing
        # print("left image shape in test_forward of detector", left_images.shape)

        output_dict = self.core(torch.cat([left_images, right_images], dim=1))
        depth_output   = output_dict['depth_output']

        cls_preds, reg_preds = self.bbox_head(
                dict(
                    features=output_dict['features'],
                    P2=P2,
                    image=left_images
                )
            )

        anchors = self.bbox_head.get_anchor(left_images, P2)

        scores, bboxes, cls_indexes = self.bbox_head.get_bboxes(cls_preds, reg_preds, anchors, P2, left_images)
        
        return scores, bboxes, cls_indexes


    def forward(self, inputs):
        f = open("/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Sequence.txt", "a")
        f.write("Stereo3D_forward \n")
        f.close()
        if isinstance(inputs, list) and len(inputs) >= 5:
            # import pdb
            # pdb.set_trace()
            return self.train_forward(*inputs)
        else:
            return self.test_forward(*inputs)
