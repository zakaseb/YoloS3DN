# from timm.models.registry import register_model
from visualDet3D.networks.utils.registry import BACKBONE_DICT

from .edgenext import EdgeNeXt
from .edgenext_bn_hs import EdgeNeXtBNHS
import torch

"""
-- Main Models
    XX-Small -> 1.3M
    X-Small -> 2.3M
    Small -> 5.6M
"""


@BACKBONE_DICT.register_module
def edgenext_xx_small(self, pretrained=False, **kwargs):
    # 1.33M & 260.58M @ 256 resolution
    # 71.23% Top-1 accuracy
    # No AA, Color Jitter=0.4, No Mixup & Cutmix, DropPath=0.0, BS=4096, lr=0.006, multi-scale-sampler
    # Jetson FPS=51.66 versus 47.67 for MobileViT_XXS
    # For A100: FPS @ BS=1: 212.13 & @ BS=256: 7042.06 versus FPS @ BS=1: 96.68 & @ BS=256: 4624.71 for MobileViT_XXS
    model = EdgeNeXt(depths=[2, 2, 6, 2], dims=[24, 48, 88, 168], expan_ratio=4,
                     global_block=[0, 1, 1, 1],
                     global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
                     use_pos_embd_xca=[False, True, False, False],
                     kernel_sizes=[3, 5, 7, 9],
                     heads=[4, 4, 4, 4],
                     d2_scales=[2, 2, 3, 4],
                     classifier_dropout=0.0)

    return model


@BACKBONE_DICT.register_module
def edgenext_x_small(pretrained=False, **kwargs):
    # 2.34M & 538.0M @ 256 resolution
    # 75.00% Top-1 accuracy
    # No AA, No Mixup & Cutmix, DropPath=0.0, BS=4096, lr=0.006, multi-scale-sampler
    # Jetson FPS=31.61 versus 28.49 for MobileViT_XS
    # For A100: FPS @ BS=1: 179.55 & @ BS=256: 4404.95 versus FPS @ BS=1: 94.55 & @ BS=256: 2361.53 for MobileViT_XS
    model = EdgeNeXt(depths=[3, 3, 9, 3], dims=[32, 64, 100, 192], expan_ratio=4,
                     global_block=[0, 1, 1, 1],
                     global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
                     use_pos_embd_xca=[False, True, False, False],
                     kernel_sizes=[3, 5, 7, 9],
                     heads=[4, 4, 4, 4],
                     d2_scales=[2, 2, 3, 4],
                     classifier_dropout=0.0)

    return model


@BACKBONE_DICT.register_module
def edgenext_small(pretrained=False, **kwargs):
    # 5.59M & 1260.59M @ 256 resolution
    # 79.43% Top-1 accuracy
    # AA=True, No Mixup & Cutmix, DropPath=0.1, BS=4096, lr=0.006, multi-scale-sampler
    # Jetson FPS=20.47 versus 18.86 for MobileViT_S
    # For A100: FPS @ BS=1: 172.33 & @ BS=256: 3010.25 versus FPS @ BS=1: 93.84 & @ BS=256: 1785.92 for MobileViT_S
    # model = EdgeNeXt(depths=[3, 3, 9, 3], dims=[32, 64, 128, 256], expan_ratio=4,
    #                  global_block=[0, 1, 1, 1],
    #                  global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
    #                  use_pos_embd_xca=[False, True, False, False],
    #                  kernel_sizes=[3, 5, 7, 9],
    #                  d2_scales=[2, 2, 3, 4],
    #                  classifier_dropout=0.0)
    model = EdgeNeXt(depths=[3, 3, 9], dims=[64, 128, 256], expan_ratio=4,
                     global_block=[0, 1, 1],
                     global_block_type=['None', 'SDTA', 'SDTA'],
                     use_pos_embd_xca=[False, True, False],
                     kernel_sizes=[3, 5, 7],
                     d2_scales=[2, 2, 3],
                     classifier_dropout=0.0)

    return model


"""
    Using BN & HSwish instead of LN & GeLU
"""


@BACKBONE_DICT.register_module
def edgenext_xx_small_bn_hs(pretrained=False, **kwargs):
    # 1.33M & 259.53M @ 256 resolution
    # 70.33% Top-1 accuracy
    # For A100: FPS @ BS=1: 219.66 & @ BS=256: 10359.98
    model = EdgeNeXtBNHS(depths=[2, 2, 6, 2], dims=[24, 48, 88, 168], expan_ratio=4,
                         global_block=[0, 1, 1, 1],
                         global_block_type=['None', 'SDTA_BN_HS', 'SDTA_BN_HS', 'SDTA_BN_HS'],
                         use_pos_embd_xca=[False, True, False, False],
                         kernel_sizes=[3, 5, 7, 9],
                         heads=[4, 4, 4, 4],
                         d2_scales=[2, 2, 3, 4],
                         **kwargs)

    return model


@BACKBONE_DICT.register_module
def edgenext_x_small_bn_hs(pretrained=False, **kwargs):
    # 2.34M & 535.84M @ 256 resolution
    # 74.87% Top-1 accuracy
    # For A100: FPS @ BS=1: 179.25 & @ BS=256: 6059.59
    model = EdgeNeXtBNHS(depths=[3, 3, 9, 3], dims=[32, 64, 100, 192], expan_ratio=4,
                         global_block=[0, 1, 1, 1],
                         global_block_type=['None', 'SDTA_BN_HS', 'SDTA_BN_HS', 'SDTA_BN_HS'],
                         use_pos_embd_xca=[False, True, False, False],
                         kernel_sizes=[3, 5, 7, 9],
                         heads=[4, 4, 4, 4],
                         d2_scales=[2, 2, 3, 4],
                         **kwargs)

    return model


@BACKBONE_DICT.register_module
def edgenext_small_bn_hs(pretrained=False, **kwargs):
    # 5.58M & 1257.28M @ 256 resolution
    # 78.39% Top-1 accuracy
    # For A100: FPS @ BS=1: 174.68 & @ BS=256: 3808.19
    model = EdgeNeXtBNHS(depths=[3, 3, 9], dims=[64, 128, 256], expan_ratio=4,
                         global_block=[0, 1, 1],
                         global_block_type=['None', 'SDTA_BN_HS', 'SDTA_BN_HS'],
                         use_pos_embd_xca=[False, True, False],
                         kernel_sizes=[3, 5, 7],
                         d2_scales=[2, 2, 3],
                         classifier_dropout=0.0)

    return model

@BACKBONE_DICT.register_module
def edgenext(pretrained=False, **kwargs):
    if kwargs['edgenext_variant'] == 'small':
        model = EdgeNeXt(depths=[3, 3, 9], dims=[64, 128, 256], expan_ratio=4,
                     global_block=[0, 1, 1],
                     global_block_type=['None', 'SDTA', 'SDTA'],
                     use_pos_embd_xca=[False, True, False],
                     kernel_sizes=[3, 5, 7],
                     d2_scales=[2, 2, 3],
                     classifier_dropout=0.0)
    elif kwargs['edgenext_variant'] == 'small_bn_hs':
        model = EdgeNeXtBNHS(depths=[3, 3, 9], dims=[64, 128, 256], expan_ratio=4,
                         global_block=[0, 1, 1],
                         global_block_type=['None', 'SDTA_BN_HS', 'SDTA_BN_HS'],
                         use_pos_embd_xca=[False, True, False],
                         kernel_sizes=[3, 5, 7],
                         d2_scales=[2, 2, 3],
                         classifier_dropout=0.0)
    if pretrained:
        state_dict = torch.load(kwargs['pretrained_edgenext_path'], map_location='cpu')
        del state_dict['model']['head.bias']
        del state_dict['model']['head.weight']
        model.load_state_dict(state_dict['model'], strict=False)
    return model


