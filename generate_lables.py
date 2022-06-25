import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np

from config import ROOT_PATH, CLASSES, DATASET_PATH

def convert_box(size, box):
    nw = 1. / size[0]
    nh = 1. / size[1]

    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]

    x = x * nw
    y = y * nh
    w = w * nw
    h = h * nh

    return [x, y, w, h]

def convert_annotation(file_path):
    image = open(file_path)
    file_name = file_path.split("/")[-1].split(".")[0]
    txt_file = open(f'{ROOT_PATH}/labels/{file_name}.txt', 'w')

    tree = ET.parse(image)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    for obj in root.iter("object"):
        difficult = obj.find("difficult").text
        cls = obj.find("name").text
        if cls not in CLASSES or difficult == 1:
            continue
        cls_id = CLASSES.index(cls)
        box = obj.find("bndbox")
        points = [float(box.find('xmin').text), 
                  float(box.find('xmax').text), 
                  float(box.find('ymin').text),
                  float(box.find('ymax').text)]
        my_box = convert_box((w, h), points)

        txt_file.write(str(cls_id) + " " + " ".join([str(x) for x in my_box]) + '\n')

def show_img(data_set, img_name):
    img = cv2.imread(f'{data_set}/JPEGImages/{img_name}.jpg')

    h, w = img.shape[:2]
    #print(w, h)

    with open(f"{ROOT_PATH}/labels/{img_name}.txt", "r") as labels:
        for label in labels:
            label = label.split(" ")
            label = [float(x.strip()) for x in label]
            #print(CLASSES[int(label[0])])

            pt1 = (int(label[1] * w - label[3] * w / 2), int(label[2] * h - label[4] * h / 2))
            pt2 = (int(label[1] * w + label[3] * w / 2), int(label[2] * h + label[4] * h / 2))
            cv2.putText(img,CLASSES[int(label[0])],pt1,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
            cv2.rectangle(img,pt1,pt2,(0,0,255,2))
        
    cv2.imshow("img",img)
    cv2.waitKey(0)

def generate_label_txt(data_set):
    files_path = data_set + '/Annotations'
    files = os.listdir(files_path)
    #files = ['2008_000397.xml']
    for file in files:
        convert_annotation(f'{files_path}/{file}')
        #show_img(data_set, file.split('.')[0])



if __name__ == "__main__":
    generate_label_txt(DATASET_PATH)