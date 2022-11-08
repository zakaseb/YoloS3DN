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
autonomous driving and Robotics solutions. The paper will be shared upon acceptance. 

## Method Architecture:

This method is inspired from YoloStereo3D by Liu et al, and EdgeNeXt by Maaz et al. The paper will be made available upon acceptance of publication. The architecture is found below:


![Architecture of YoloS3DN (top) with the architecture of the modified EdgeNext
backbone(bottom)](https://user-images.githubusercontent.com/45034431/200579644-e1c8f8fe-5530-4f81-96c0-a885531abd5c.png)

This repo aims at improving at the current SOTA of 3D Stereo Object in the KITTI 3D Object Detection benchmark in terms of number of parameters and average inference speeds. 

## Related Papers:
This repo is inspired from various methods, but more specifically YoloStereo3D, official implementation of 2021 *ICRA* paper [**YOLOStereo3D: A Step Back to 2D for Efficient Stereo 3D Detection**](https://arxiv.org/abs/2103.09422), the repo can be found at https://github.com/Owen-Liuyuxuan/visualDet3D. Additionally, it is based on the official implementation of [**EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications**]((https://arxiv.org/abs/2206.10589)) which can be found in https://github.com/mmaaz60/EdgeNeXt. Please refer to those papers for further inspriations and clarifications. 
```
@inproceedings{liu2021yolostereo3d,
  title={YOLOStereo3D: A Step Back to 2D for Efficient Stereo 3D Detection},
  author={Yuxuan Liu and Lujia Wang and Ming, Liu},
  booktitle={2021 International Conference on Robotics and Automation (ICRA)},
  year={2021},
  organization={IEEE}
}
```

```
@misc{https://doi.org/10.48550/arxiv.2206.10589,
  doi = {10.48550/ARXIV.2206.10589},
  
  url = {https://arxiv.org/abs/2206.10589},
  
  author = {Maaz, Muhammad and Shaker, Abdelrahman and Cholakkal, Hisham and Khan, Salman and Zamir, Syed Waqas and Anwer, Rao Muhammad and Khan, Fahad Shahbaz},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}

}
```



## Key Features

- **SOTA Performance** State of the art results on visual 3D detection in terms of average inference speeds and number of params.
- **Modular Design** Modular design for dataset, network and running pipelines.
- **Support Various Task** Compatible with the training and testing of mono/stereo 3D detection and depth prediction.
- **Distributed & Single GPU** Support training with multiple GPUs.
- **Installation-Free Setup** The setup process only build operations and does not require installation to keep the environment clean.
- **Global Path-based IMDB** Do not need data placed inside the folder, convienient for managing data and code separately.




## Setup
### Environment setup. 

```bash
pip3 install -r requirement.txt
```
or 

```bash
conda env create -f ./stereo3D.yaml
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


**important paths to modify in config** :
1. cfg.path.data_path: Path to KITTI training data. We expect calib, image_2, image_3, label_2 being the subfolder (directly unzipping the downloaded zips will be fine)
2. cfg.path.test_path: Path to KITTI testing data.  We expect calib, image_2 being the subfolder.
3. cfg.path.visualDet3D_path: Path to the "visualDet3D" directorty of the current repo
4. cfg.path.project_path: Path to the workdirs of the projects (will have temp_outputs, log, checkpoints)

Please check the template's comments and other comments in codes to fully exploit the repo.


```
## Compute image database and anchors mean/std
# You can run ./launcher/det_precompute.sh without arguments to see helper documents
./launcher/det_precompute.sh config/config_stereo_3d.py train
./launcher/det_precompute.sh config/config_stereo_3d.py test



## run this if disparity map is needed, can be computed with point cloud or openCV BlockMatching
# You can run ./launcher/disparity_precompute.sh without arguments to see helper documents
./disparity_precompute.sh config/config_stereo_3d.py $IsUsingPointCloud

## train the model with one GPU
# You can run ./launcher/train.sh without arguments to see helper documents
./launcher/train.sh  --config/config_stereo_3d.py 0 $experiment_name # validation goes along

## produce validation/test result # we only support single GPU testing
# You can run ./launcher/eval.sh without arguments to see helper documents
./launcher/eval.sh --config/config_stereo_3d.py 0 $CHECKPOINT_PATH validation/test

```
