# Realtime Detection  

These are packages for using Intel RealSense D457 with ROS and YOLOx with AidLite  
on AIBox 6490 to do objects' realtime detection.

## Devices  
Computer Unit: [AIBox QCS6490](https://aidlux.com/product/edge-computing/aibox-6490)  
YOLO Version:  [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX.git)  
Camera:        [Realsense D457(rgb)](https://www.intelrealsense.com/depth-camera-d457/)  

## Dependency  
- ROS noetic
- [AidLite SDK for C++](https://v2.docs.aidlux.com/sdk-api/aidlite-sdk/aidlite-c++)
- glog && gflags
- OpenCV(4.2.0 recommended)

## Installation Instructions

### Realsense SDK  
You can install librealsense and realsense according to the steps in [realsense-ros](https://github.com/IntelRealSense/realsense-ros.git).

### Realtime Detection
''' bash  
git clone https://github.com/zzzzyp-sgg/realtime_detection.git  
mkdir build && cd build  
cmake ..  
make -j4  
'''

## Usage Instructions  
After launch realsense camera, you can start the detection node in ROS:  
''' bash  
# rember to source devel/setup.bash
rosrun realtime_detection d457test_node [config_file]
'''
We provide a simple example and the YOLOx model file is from: [yolox-ti-lite_tflite](https://github.com/motokimura/yolox-ti-lite_tflite.git).  
This is currently a sample version and more detailed features are still under development.
