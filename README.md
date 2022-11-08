# YoloS3DN: Towards Low-Latency ViT-Powered Stereo 3D Object Detection.

## Abstract:

Stereo 3D Object Detection has been an ever-growing challenge in the field of Computer
Vision, specifically because of its role in deploying Autonomous Driving solutions
that are computationally lightweight, fast, and accurate. This task is particularly
challenging for models that utilize point cloud for their disparity estimation for depth 3D
reconstruction. Moreover, Vision Transformers have recently started outperforming CNNs
in image classification tasks, all whilst consuming fewer resources with fewer parameters
and FLOPs. To this end, YoloS3DN is proposed, a lightweight, real-time capable iteration
of YoloStereo3D with a Vision Transformer backbone inspired from EdgeNeXt. This network scales back into the
2D object detection while reinforcing them with stereo features. This model was trained
on a single NVIDIA RTX 3090 GPU on the KITTI Stereo Dataset and validated on the 3D
Object Detection Benchmark of 2017. Ablation and comparative experiments display
experimental, comparative, and qualitative results with superior performance in inference
speed and state-of-the-art comparable performance in accuracy. The advancements in
this field push innovation towards Stereo-based Object detection in real-time and real-world
autonomous driving and Robotics solutions.

## Method Architecture:

This method is inspired from YoloStereo3D by Liu et al, and EdgeNeXt by Maaz et al. The paper will be made available upon acceptance of publication. The architecture is found below:


![Architecture of YoloS3DN (top) with the architecture of the modified EdgeNext
backbone(bottom)](https://mbzuaiac-my.sharepoint.com/:i:/g/personal/zakaria_sebaitre_mbzuai_ac_ae/ETuHihVR9x5ImOG_m3TJz1oBDVBJPMcq74nfh3udBsK3UQ)
![image](https://user-images.githubusercontent.com/45034431/200579644-e1c8f8fe-5530-4f81-96c0-a885531abd5c.png)


This repo aims to provide flexible and reproducible visual 3D detection on KITTI dataset. We expect scripts starting from the current directory, and treat ./visualDet3D as a package that we could modify and test directly instead of a library. Several useful scripts are provided in the main directory for easy usage.

We believe that visual tasks are interconnected, so we make this library extensible to more experiments. 
The package uses registry to register datasets, models, processing functions and more, allowing easy inserting of new tasks/mlOW odels while not interfere with the existing ones.

## Related Paper:

This repo contains the official implementation of 2021 *RAL* \& *ICRA* paper [**Ground-aware Monocular 3D Object Detection for Autonomous Driving**](https://ieeexplore.ieee.org/document/9327478). [Arxiv Page](https://arxiv.org/abs/2102.00690). Pretrained model can be found at [release pages](https://github.com/Owen-Liuyuxuan/visualDet3D/releases/tag/1.0).
```
@ARTICLE{9327478,
  author={Y. {Liu} and Y. {Yuan} and M. {Liu}},
  journal={IEEE Robotics and Automation Letters}, 
  title={Ground-aware Monocular 3D Object Detection for Autonomous Driving}, 
  year={2021},
  doi={10.1109/LRA.2021.3052442}}
```

Also the official implementation of 2021 *ICRA* paper [**YOLOStereo3D: A Step Back to 2D for Efficient Stereo 3D Detection**](https://arxiv.org/abs/2103.09422). Pretrained model can be found at [release pages](https://github.com/Owen-Liuyuxuan/visualDet3D/releases/tag/1.1).
```
@inproceedings{liu2021yolostereo3d,
  title={YOLOStereo3D: A Step Back to 2D for Efficient Stereo 3D Detection},
  author={Yuxuan Liu and Lujia Wang and Ming, Liu},
  booktitle={2021 International Conference on Robotics and Automation (ICRA)},
  year={2021},
  organization={IEEE}
}
```

We further incorperate an *Unofficial* re-implementation of **Monocular 3D Detection with Geometric Constraints Embedding and Semi-supervised Training** (KM3D) as a reference on how to integrate with other frameworks. (Notice that the codes are from the [originally official repo](https://github.com/Banconxuan/RTM3D), and we **DO NOT** guarantee a complete re-implementation).

Update (2021.07.02): We provide an *Unofficial* re-implementation of **Objects are Different: Flexible Monocular 3D Object Detection** (MonoFlex) with few additional codes, based on the KM3D structure. Many of the core codes are from [original official repo](https://github.com/zhangyp15/MonoFlex). We did not implement the edge merge operation and the corner loss, but we manage to maintain most of the performance based on the proposed depth fusion methods(validation AP reaches 15%).

Update (2021.12.11): We provide an *Unofficial* re-implmentation of **Digging Into Output Representation For Monocular 3D Object Detection** (Digging_M3D) to introduce an simple but important numerical trick to significantly improve the KITTI mAP scores and make a significant change to the KITTI leaderboard. Details can be found in the [paper](https://openreview.net/forum?id=mPlm356yMIP). At the time of the open-source, the paper has not been officially published, and we will keep up with the update of the paper.

## Key Features

- **SOTA Performance** State of the art result on visual 3D detection.
- **Modular Design** Modular design for dataset, network and running pipelines.
- **Support Various Task** Compatible with the training and testing of mono/stereo 3D detection and depth prediction.
- **Distributed & Single GPU** Support training with multiple GPUs.
- **Installation-Free Setup** The setup process only build operations and does not require installation to keep the environment clean.
- **Global Path-based IMDB** Do not need data placed inside the folder, convienient for managing data and code separately.


We provide start-up solutions for [Mono3D](docs/mono3d.md), [Stereo3D](docs/stereo3d.md), [Depth Predictions](docs/monoDepth.md) and more (until further publication).

Reference: this repo borrows codes and ideas from [retinanet](https://github.com/yhenon/pytorch-retinanet),
[mmdetection](https://github.com/open-mmlab/mmdetection),
[M3D-RPN](https://github.com/garrickbrazil/M3D-RPN),
[DORN](https://github.com/dontLoveBugs/SupervisedDepthPrediction),
[EdgeNets](https://github.com/sacmehta/EdgeNets),
[det3](https://github.com/pyun-ram/FL3D)

## Setup
### Environment setup. 

```bash
pip3 install -r requirement.txt
```
or manually check dependencies.

```bash
# build ops (deform convs and iou3d), We will not install operations into the system environment
./make.sh
```

## Start Training

Please check the corresponding task: [Mono3D](docs/mono3d.md), [Stereo3D](docs/stereo3d.md), [Depth Predictions](docs/monoDepth.md). More demo will be available through contributions and further paper submission.

### Config and Path setup. 

Please modify the path and other parameters in **config/\*.py**. **config/\*_example** files are templates.

**Notice**:
*_examples are **NOT** utilized by the code and \*.py under /config is **ignored** by .gitignore.

The content of the selected config file will be recorded in tensorboard at the beginning of training.

**important paths to modify in config** :
1. cfg.path.data_path: Path to KITTI training data. We expect calib, image_2, image_3, label_2 being the subfolder (directly unzipping the downloaded zips will be fine)
2. cfg.path.test_path: Path to KITTI testing data.  We expect calib, image_2 being the subfolder.
3. cfg.path.visualDet3D_path: Path to the "visualDet3D" directorty of the current repo
4. cfg.path.project_path: Path to the workdirs of the projects (will have temp_outputs, log, checkpoints)

Please check the template's comments and other comments in codes to fully exploit the repo.

## Further Info and Bug Issues

1. Open issues on the repo if you meet troubles or find a bug or have some suggestions.
2. Email to yliuhb@connect.ust.hk


## Other Resources

- [RAM-LAB](https://www.ram-lab.com)
- [Collections of Papers and Readings](https://owen-liuyuxuan.github.io/papers_reading_sharing.github.io/);
-  [Collection for Mono3D](https://owen-liuyuxuan.github.io/papers_reading_sharing.github.io/3dDetection/RecentCollectionForMono3D/); [Ground-Aware 3D](https://owen-liuyuxuan.github.io/papers_reading_sharing.github.io/3dDetection/GroundAwareConvultion/)
- [Collection for Stereo3D](https://owen-liuyuxuan.github.io/papers_reading_sharing.github.io/3dDetection/RecentCollectionForStereo3D/); [YOLOStereo3D](https://owen-liuyuxuan.github.io/papers_reading_sharing.github.io/3dDetection/YOLOStereo3D/)

## Related Codes

- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [M3D-RPN](https://github.com/garrickbrazil/M3D-RPN)
- [Retinanet](https://github.com/yhenon/pytorch-retinanet)
- [DORN](https://github.com/dontLoveBugs/SupervisedDepthPrediction)
- [det3](https://github.com/pyun-ram/FL3D)
- [RTM3D](https://github.com/Banconxuan/RTM3D)# YoloS3DN
