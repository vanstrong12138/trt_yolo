# TRT YOLO

[![Ubuntu 22.04](https://img.shields.io/badge/Ubuntu-22.04-blue.svg?logo=ubuntu)](https://ubuntu.com/)
[![ROS2 Humble](https://img.shields.io/badge/ros2-humble-brightgreen.svg?logo=ros)](https://wiki.ros.org/humble)

Support List:

|Model Name|size|task|Real|
|-|-|-|-|
|Yolov8|n|detection|✅|
|Yolov8|n|segmentation|✅|
|Yolov8|n|pose|✅|
|Yolo11|n|detection|✅|
|Yolo11|n|segmentation|✅|
|Yolo11|n|pose|✅|
|Yolo11|n|obb-detection|✅|

> [!IMPORTANT]
> Before running each model, you need to execute the serialization program to export the .engine file.

## Preparation

### FOR Jetson Orin Series

1. install source

```bash
sudo bash -c 'echo "deb https://repo.download.nvidia.com/jetson/common r36.4 main" >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list'
sudo bash -c 'echo "deb https://repo.download.nvidia.com/jetson/t234 r36.4 main" >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list'
```

2. install dependencies
```bash
sudo apt update
sudo apt install tensorrt cudnn 
```

3. clone package

```bash
cd your_workspace/src/
git clone https://github.com/vanstrong12138/trt_yolo.git
cd ..
colcon build
```

### FOR Desktop PC/Laptop

TODO: setup method

## Run

1. source workspace
```bash
source your_workspace/install/setup.bash
```

2. run yolov8n detection example

- first, for first run, you need to run the model serialization program to export the .engine file 

```bash
ros2 run trt_yolo yolov8n_det.launch.py mode:=serialize wts_path:=/path/to/yolov8n_det.wts
# this step will generate the yolov8n_det.engine file in the install directory
# it will take about 10 minutes
```

- second, when engine file is generated, run the detection node

```bash
ros2 run trt_yolo yolov8n_det.launch.py labels_file:=/path/to/coco.txt image_topic:=/your/color/image/topic
```

3. run yolo11n segmentation example

- first, for first run, you need to run the model serialization program to export the .engine file 

```bash
ros2 run trt_yolo yolo11n_seg.launch.py mode:=serialize wts_path:=/path/to/yolo11n_seg.wts
# this step will generate the yolo11n_seg.engine file in the install directory
# it will take about 10 minutes
```

- second, when engine file is generated, run the detection node

```bash
ros2 run trt_yolo yolo11n_seg.launch.py labels_file:=/path/to/coco.txt image_topic:=/your/color/image/topic
```

4. for advance usage, you can run the following command to see more options
```bash
# --show-args for detail options
ros2 run trt_yolo yolo11n_seg.launch.py --show-args
```

## Acknowledgements
- [Yolov8](https://github.com/ultralytics/ultralytics)
- [Yolo11](https://github.com/ultralytics/ultralytics)
- [tensorrtx](https://github.com/wang-xinyu/tensorrtx)





