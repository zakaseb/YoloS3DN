# from .visualDet3D.utils import*]
import torch
from models.edgenext import EdgeNeXt
# def testFunc( **kwargs ):
#     options = {
#             'option1' : 'default_value1',
#             'option2' : 'default_value2',
#             'option3' : 'default_value3', }

#     options.update(kwargs)
#     print(options)
# def __init__(self, kwargs)
#     self.attribute = kwargs.pop('name', default_value)



# # img = torch.randn(5, 3, 256, 256)
test_net = EdgeNeXt(depths=[2, 2, 6, 2], dims=[24, 48, 88, 168], expan_ratio=4,
                     global_block=[0, 1, 1, 1],
                     global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
                     use_pos_embd_xca=[False, True, False, False],
                     kernel_sizes=[3, 5, 7, 9],
                     heads=[4, 4, 4, 4],
                     d2_scales=[2, 2, 3, 4], classifier_dropout=0.0)

print(test_net)