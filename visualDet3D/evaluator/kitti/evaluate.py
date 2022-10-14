import time
from .kitti_common import get_label_annos
from .eval import get_official_eval_result, get_coco_eval_result
from numba import cuda

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def evaluate(label_path="/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Training/label_2", #changed label path to local instead of author's
             result_path="/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Project/Stereo3D/output/validation/data", #changed path into local data path instead of author's
             label_split_file="/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/visualDet3D/data/kitti/chen_split/val.txt", # added path to know which val.txt to use
             current_classes=[0],
             gpu=0):
    cuda.select_device(gpu)
    dt_annos = get_label_annos(result_path)
    val_image_ids = _read_imageset_file(label_split_file)
    gt_annos = get_label_annos(label_path, val_image_ids)
    result_texts = []
    for current_class in current_classes:
        result_texts.append(get_official_eval_result(gt_annos, dt_annos, current_class))
    return result_texts
