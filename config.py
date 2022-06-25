import os

CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']

ROOT_PATH = os.getcwd()
print(ROOT_PATH)
DATASET_PATH = f'{ROOT_PATH}/voc2012'

BOX_NUM = 2

EPOCHS = 1
BATCH_SIZE = 1
LR = 0.001