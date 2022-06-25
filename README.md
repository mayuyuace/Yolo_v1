# Yolo_v1

## Config

python:3.8.10

cuda:11.1

pytorch:1.8.0+cu111

modules: torch torchvision numpy cv2

DATA: https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar. Download and name it voc2012.


## How to rum

python train.py

## File struct

-main

--labels

--model_pkl

--voc2012

---Annotations

---ImageSets

---JPEGImages

---SegmentationClass

---SegmentationObject
