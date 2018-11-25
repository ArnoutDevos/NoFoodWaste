# LauzHack18 - Privacy preserving CV

This package is designed to achieve two different objectives :         
1. Provide real time object recognition and location for environmental awareness         
2. Provide data minimization to protect privacy     

For your confort, this file is available in .md or .html.

## Introduction
Developed by Sylvain C for LauzHack18 project. This code was inspired and modified from several online resources such as OpenCV, PYImageSearch and tensorflow tutorial. The ```Darknet/``` directory is a clone of Joseph Redmon git : ```https://github.com/pjreddie/darknet.git```. This package provides a code to launch his CNNs called darknet and YOLO. Finally ```Analysis/``` provides a analysis of the different CNN performances.

Note : The most interesting part of this work lies in the Analysis and not in the CNNs implementation.

## Note 
Every commands were launched from the terminal. Please make sure you are in the right directory before launching.

## Dependencies and requirements
This code was tested on OSX 10.11.6 using python 2.7. The machine had 8GB 1867 MHz memory and a 2.7 GHz Intel Core i5 processor. We decided not to run it on the GPU in order to keep in mind that in our project mindset, this software should be run on edge devices with no GPU resources. 

The following packages need to be installed on the machine :      
- OpenCV    
- imutils   
- time   
- python 2.7      
- matplotlib     
- tabulate     
- numpy     
- pickle     
- os     
- argparse     
- Matlab 2017a     



## 1. Real-time recognition
In order to run a real-time detection using a webcam and MobileNet SSD CNN, the user needs to go to ```mobileNet/RT_object_detect/``` and run ```real_time_object_detection2.py```. If a common object such as a chair, table or sofa is in front of the camera, a warning message will be displayed on the console. 

```
python real_time_object_detection2.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
```

To run the network over a single image, go to ```mobileNet/object_detect/``` and run ```object_detection2.py```.

```
python object_detection2.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel --image images/example_1.JPEG
```
Finally, to run the CNN over a batch of images, specify the directory with ```--directory``` and run ```main_obj_det.py```.

```
python main_obj_det.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel --directory images
```
This will generate a .txt file similar to mobile_dist.txt.
