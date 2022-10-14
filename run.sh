## Compute image database and anchors mean/std
# You can run ./launcher/det_precompute.sh without arguments to see helper documents
./launchers/det_precompute.sh config/config_stereo_3d.py train
./launchers/det_precompute.sh config/config_stereo_3d.py test

# ## run this if disparity map is needed, can be computed with point cloud or openCV BlockMatching
# # You can run ./launcher/disparity_precompute.sh without arguments to see helper documents
/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/launchers/disparity_precompute.sh config/config_stereo_3d.py True

## train the model with one GPU
# You can run ./launcher/train.sh without arguments to see helper documents
# ./launchers/train.sh  /home/zakaseb/Thesis/YoloStereo3D/Stereo3D/config/config_stereo_3d.py 0 trial # validation goes along

## produce validation/test result # we only support single GPU testing
# You can run ./launcher/eval.sh without arguments to see helper documents
# ./launchers/eval.sh /home/zakaseb/Thesis/YoloStereo3D/Stereo3D/config/config_stereo_3d.py 0 /home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Project/Stereo3D/checkpoint/Stereo3D_latest.pth test
# ./launchers/eval.sh /home/zakaseb/Thesis/YoloStereo3D/Stereo3D/config/config_stereo_3d.py 0 /home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Project/Stereo3D/checkpoint/Stereo3D_latest.pth validation