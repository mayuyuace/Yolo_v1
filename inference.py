import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader

from data_set import VOC2012
from iou import calculate_iou
from config import CLASSES, BOX_NUM, ROOT_PATH

def NMS(bbox, conf_thresh=0.1, iou_thresh=0.3):
    bbox_prob = bbox[:,5:].clone()
    bbox_confi = bbox[:, 4].clone().unsqueeze(1).expand_as(bbox_prob)
    bbox_cls_spec_conf = bbox_confi*bbox_prob 
    bbox_cls_spec_conf[bbox_cls_spec_conf<=conf_thresh] = 0  
    for c in range(20):
        rank = torch.sort(bbox_cls_spec_conf[:,c],descending=True).indices
        for i in range(49 * BOX_NUM):
            if bbox_cls_spec_conf[rank[i],c]!=0:
                for j in range(i+1,49 * BOX_NUM):
                    if bbox_cls_spec_conf[rank[j],c]!=0:
                        iou = calculate_iou(bbox[rank[i],0:4],bbox[rank[j],0:4])
                        if iou > iou_thresh:  
                            bbox_cls_spec_conf[rank[j],c] = 0
    bbox = bbox[torch.max(bbox_cls_spec_conf,dim=1).values>0] 
    bbox_cls_spec_conf = bbox_cls_spec_conf[torch.max(bbox_cls_spec_conf,dim=1).values>0]
    res = torch.ones((bbox.size()[0],6))
    res[:,1:5] = bbox[:,0:4]  
    res[:,0] = torch.argmax(bbox[:,5:],dim=1).int()
    res[:,5] = torch.max(bbox_cls_spec_conf,dim=1).values  
    return res

def labels2bbox(matrix):
    if matrix.size()[0:2]!=(7,7):
        raise ValueError("Error: Wrong labels size:",matrix.size())
    bbox = torch.zeros((49 * BOX_NUM, 25)).cuda()
    matrix = torch.reshape(matrix, (49, 5 * BOX_NUM + len(CLASSES)))

    m = torch.Tensor(2, 7, 7).cuda()

    for i in range(7):
        m[0, i ,:] = i
        m[1, :, i] = i
    
    m = m.reshape(2, 49)

    for i in range(BOX_NUM):
        bbox[49 * i: 49 * (i + 1), 0] = (matrix[ :, 5 * i + 0] + m[1, ]) / 7 + matrix[ :, 5 * i + 2] / 2
        bbox[49 * i: 49 * (i + 1), 1] = (matrix[ :, 5 * i + 1] + m[0, ]) / 7 + matrix[ :, 5 * i + 3] / 2
        bbox[49 * i: 49 * (i + 1), 2] = (matrix[ :, 5 * i + 0] + m[1, ]) / 7 - matrix[ :, 5 * i + 2] / 2
        bbox[49 * i: 49 * (i + 1), 3] = (matrix[ :, 5 * i + 1] + m[0, ]) / 7 - matrix[ :, 5 * i + 3] / 2
        bbox[49 * i: 49 * (i + 1), 4] = matrix[ :, 5 * i + 4]
        bbox[49 * i: 49 * (i + 1):, 5:] = matrix[ :, 5 * BOX_NUM:]
    return NMS(bbox)  

# mark box
COLOR = [(255,0,0),(255,125,0),(255,255,0),(255,0,125),(255,0,250),
         (255,125,125),(255,125,250),(125,125,0),(0,255,125),(255,0,0),
         (0,0,255),(125,0,255),(0,125,255),(0,255,255),(125,125,255),
         (0,255,0),(125,255,125),(255,255,255),(100,100,100),(0,0,0),]

def draw_bbox(img,bbox):
    h,w = img.shape[0:2]
    n = bbox.size()[0]

    for i in range(n):
        p1 = (int(w*bbox[i,1]), int(h*bbox[i,2]))
        p2 = (int(w*bbox[i,3]), int(h*bbox[i,4]))
        cls_name = CLASSES[int(bbox[i,0])]
        confidence = bbox[i,5]

        cv2.rectangle(img,p1,p2,color=COLOR[int(bbox[i,0])])
        cv2.putText(img,cls_name,p1,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
    cv2.imshow("bbox",img)
    cv2.waitKey(0)

def inference(model):
    if isinstance(model, str):
        model = torch.load(model)
    
    val_dataloader = DataLoader(VOC2012(is_train=False), batch_size=1, shuffle=False)

    for i, (inputs, labels) in enumerate(val_dataloader):
        inputs = inputs.cuda()
        #test labels2bbox
        #labels = labels.squeeze(dim=0)
        #labels = labels.permute((1,2,0))
        #bbox = labels2bbox(labels) 

        pred = model(inputs)
        pred = pred.squeeze(dim=0)
        pred = pred.permute((1,2,0))
        bbox = labels2bbox(pred) 
        
        inputs = inputs.squeeze(dim=0)
        inputs = inputs.permute((1,2,0))
        img = inputs.cpu().numpy().copy()
        img = 255*img 
        img = img.astype(np.uint8)
        draw_bbox(img,bbox.cpu()) 
        input()

if __name__ == "__main__":
    model = f"{ROOT_PATH}/model_pkl/YOLO_v1_100epoch.pkl"
    inference(model=model)