# tracking_ros  [![python_check](https://github.com/ojh6404/tracking_ros/actions/workflows/python_check.yml/badge.svg)](https://github.com/ojh6404/tracking_ros/actions/workflows/python_check.yml)

ROS1 package for detecting and tracking objects using [SAM](https://github.com/facebookresearch/segment-anything.git), [Cutie](https://github.com/hkchengrex/Cutie.git), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO.git) and [DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA.git), inspired by [detic_ros](https://github.com/HiroIshida/detic_ros.git).

## Usage
Tested : image of 480X640 30hz, 3090ti
### Interactive prompt for generating mask and tracking object using SAM and Cutie.
https://github.com/ojh6404/tracking_ros/assets/54985442/f8a49814-2645-4b71-887e-1c8f02da5c38

sam_node publishes segmentation prompt which is used by cutie_node to track objects. It runs almost real-time (~30hz).
### Detecting and tracking object using SAM, GroundingDINO and DEVA.
https://github.com/ojh6404/tracking_ros/assets/54985442/f55e8850-a7bc-41fc-b398-1c7dda47c66d

deva_ndoe queries objects GroundingDINO and SAM at some intervals, so it can track new object after tracking is started. It runs ~15hz and you can adjust `cfg['detection_every']` for performance.
See [`node_scripts/model_config.py`](node_scripts/model_config.py)

## Setup

### Prerequisite
This package is build upon
- ROS1 (Noetic)
- catkin virtualenv (python>=3.9 used for DEVA)
- (Optional) docker and nvidia-container-toolkit (for environment safety)

### Build package

#### on your workspace
If you want build this package directly on your workspace, please be aware of python environment dependencies (python3.9 and pytorch is needed to build package).
```bash
mkdir -p ~/ros/catkin_ws/src && cd ~/ros/catkin_ws/src
git clone https://github.com/ojh6404/tracking_ros.git
wstool init
wstool merge -t . tracking_ros/rosinstall.noetic
wstool update -t . # jsk-ros-pkg/jsk_visualization for GUI
cd tracking_ros && ./prepare.sh
cd ~/ros/catkin_ws && catkin b
```

#### using docker (Recommended)
Otherwise, you can build this package on docker environment.
```bash
git clone https://github.com/ojh6404/tracking_ros.git
cd tracking_ros
docker build -t tracking_ros .
```

## How to use
Please refer sample_track.launch and deva.launch
### Tracking using SAM and Cutie with interactive gui prompt.
#### 1. run directly
```bash
roslaunch tracking_ros sample_track.launch \
    input_image:=/kinect_head/rgb/image_rect_color \
    mode:=prompt \
    model_type:=vit_t \
    device:=cuda:0
```
#### 2. using docker
You need to launch tracker and gui seperately cause docker doesn't have gui, so launch tracker by
```bash
./run_docker -host pr1040 -mount ./launch -name track.launch \
    input_image:=/kinect_head/rgb/image_rect_color \
    mode:=prompt \
    model_type:=vit_t \
    device:=cuda:0
```
where
- `-host` : hostname like `pr1040` or `localhost`
- `-mount` : mount launch file directory for launch inside docker.
- `-name` : launch file name to run

and launch rqt gui on your gui machine by
```bash
roslaunch tracking_ros sam_gui.launch
```

### Detecting and tracking object.
```bash
roslaunch tracking_ros deva.launch \
    input_image:=/kinect_head/rgb/image_rect_color \
    classes:="monitor; keyboard; cup" \
    model_type:=vit_t \
    device:=cuda:0
```
or
```bash
./run_docker -host pr1040 -mount ./launch -name deva.launch \
    input_image:=/kinect_head/rgb/image_rect_color \
    classes:="monitor; keyboard; cup" \
    model_type:=vit_t \
    device:=cuda:0
```

### TODO
- add rostest and docker build test
- add [CoTracker](https://github.com/facebookresearch/co-tracker.git) and [Track Any Point](https://github.com/google-deepmind/tapnet.git).
