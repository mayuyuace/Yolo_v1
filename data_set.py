import cv2
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader

from config import DATASET_PATH, CLASSES, BOX_NUM, BATCH_SIZE, ROOT_PATH

def convert_box(box):
    gridsize = 1.0 / 7
    labels = np.zeros((7, 7, 5 * BOX_NUM + len(CLASSES)))

    for i in range(len(box) // 5):
        gridx = int(box[i * 5 + 1] // gridsize) 
        gridy = int(box[i * 5 + 2] // gridsize)
        
        gridpx = box[i * 5 + 1] / gridsize - gridx
        gridpy = box[i * 5 + 2] / gridsize - gridy

        
        for b in range(BOX_NUM):
            labels[gridy, gridx, b * 5: b * 5 + 5] = np.array([gridpx, gridpy, box[i * 5 + 3], box[i * 5 + 4], 1])

        labels[gridy, gridx, BOX_NUM * 5 + int(box[i * 5])] = 1
    
    return labels

class VOC2012(Dataset):
    def __init__(self, is_train=True, is_aug=True):
        if is_train:
            with open(f"{DATASET_PATH}/ImageSets/Main/train.txt", 'r') as f:
                self.filenames = [x.strip() for x in f]
        else:
            with open(f"{DATASET_PATH}/ImageSets/Main/val.txt", 'r') as f:
                self.filenames = [x.strip() for x in f]
        self.image_path = f'{DATASET_PATH}/JPEGImages'
        self.label_path = f'{ROOT_PATH}/labels'
        self.is_aug = is_aug
    
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        img = cv2.imread(f'{self.image_path}/{self.filenames[item]}.jpg')
        h, w = img.shape[0:2]
        input_size = (448, 448)

        w_pad, h_pad = 0, 0
        if h > w:
            w_pad = (h - w) // 2
            img = np.pad(img, ((0, 0), (w_pad, w_pad), (0, 0)), 'constant', constant_values=0)
        elif w > h:
            h_pad = (w - h) // 2
            img = np.pad(img, ((h_pad, h_pad), (0, 0), (0, 0)), 'constant', constant_values=0)
        
        img = cv2.resize(img, input_size)

        if self.is_aug:
            aug = transforms.Compose([transforms.ToTensor()])
            img = aug(img)

        with open(f'{self.label_path}/{self.filenames[item]}.txt') as f:
            box = f.read().split('\n')
        
        box = [x.split() for x in box]
        box = [float(x) for y in box for x in y]

        if len(box) % 5 != 0:
            raise ValueError("File:"+self.label_path+self.filenames[item]+".txt"+"——bbox Extraction Error!")
        
        for i in range(len(box) // 5):
            if w_pad != 0:
                box[i * 5 + 1] = (box[i * 5 + 1] * w + w_pad) / h
                box[i * 5 + 3] = (box[i * 5 + 3] * w) / h
            elif h_pad != 0:
                box[i * 5 + 2] = (box[i * 5 + 2] * h + h_pad) / w
                box[i * 5 + 4] = (box[i * 5 + 4] * h) / w
        
        labels = convert_box(box)
        labels = transforms.ToTensor()(labels)
        return img,labels

if __name__ == "__main__":
    train_data = VOC2012(is_train=True)
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=False)

    for i, (inputs, labels) in enumerate(train_dataloader):
        labels = labels.numpy()
        if np.any(labels < 0):
            print(i)
            print(labels)