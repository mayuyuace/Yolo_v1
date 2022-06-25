import torch.nn as nn
import torch

from iou import calculate_iou
from config import BOX_NUM, CLASSES

class LOSS(nn.Module):
    def __init__(self):
        super(LOSS, self).__init__()

    def forward(self, pred, labels):
        """
        pred: (batchsize, 30, 7, 7)
        labels: (batchsize, 30, 7, 7)
        """
        #网格数量
        gridx_num, gridy_num = labels.size()[-2:]
        #每个网格2个框
        b_num = BOX_NUM
        #20种分类
        class_num = len(CLASSES)
        #无目标置信度损失
        noobj_confi_loss = 0.
        #有目标置信度损失
        obj_confi_loss = 0.
        #有目标坐标损失
        obj_loss = 0.
        #有目标类别损失
        class_loss = 0.
        batchsize = labels.size()[0]

        pred_box = []
        for i in range(b_num):
            pred_box.append(pred[ : , 5 * i: 5 * i + 4, :, : ])

        label_box = []
        for i in range(b_num):
            label_box.append(labels[ : , 5 * i: 5 * i + 4, :, : ])

        m = pred_box[0][:, [0], :, :] 
        for i in range(7):
            m[:, :, :, i] = i

        n = pred_box[0][:, [1], :, :]
        for i in range(7):
            n[:, :, i, :] = i

        for i in range(b_num):
            pred_box[i] = torch.cat(((pred_box[i][:, [0], :, :] + m) / gridx_num + pred_box[i][:, [2], :, :,] / 2, 
                                    (pred_box[i][:, [1], :, :] + n) / gridy_num + pred_box[i][:, [3], :, :,] / 2,
                                    (pred_box[i][:, [0], :, :] + m) / gridx_num - pred_box[i][:, [2], :, :,] / 2,
                                    (pred_box[i][:, [1], :, :] + n) / gridy_num - pred_box[i][:, [3], :, :,] / 2),
                                    dim=1)
        
        for i in range(b_num):
            label_box[i] = torch.cat(((label_box[i][:, [0], :, :] + m) / gridx_num + label_box[i][:, [2], :, :,] / 2, 
                                  (label_box[i][:, [1], :, :] + n) / gridy_num + label_box[i][:, [3], :, :,] / 2,
                                  (label_box[i][:, [0], :, :] + m) / gridx_num - label_box[i][:, [2], :, :,] / 2,
                                  (label_box[i][:, [1], :, :] + n) / gridy_num - label_box[i][:, [3], :, :,] / 2),
                                  dim=1)

        ious = []
        for i in range(b_num):
            ious.append(calculate_iou(pred_box[i], label_box[i]))
        
        iou = ious[0]
        for i in range(b_num - 1):
            iou = torch.max(iou, ious[i + 1])

        masks = []
        for i in range(b_num):
            masks.append(ious[i])
            masks[i][iou != masks[i]] == 0
            masks[i][iou == masks[i]] == 1

        for i in range(b_num):
            obj_loss += 5.0 * torch.sum(labels[:, [i * 5 + 4], :, :] * masks[i] * 
                                        ((pred[:, [i * 5], :, :] - labels[:, [i * 5], :, :]) ** 2 + 
                                        (pred[:, [i * 5 + 1], :, :] - labels[:, [i * 5 + 1], :, :]) ** 2 + 
                                        (pred[:, [i * 5 + 2], :, :].sqrt() - labels[:, [i * 5 + 2], :, :].sqrt()) ** 2 + 
                                        (pred[:, [i * 5 + 3], :, :].sqrt() - labels[:, [i * 5 + 3], :, :].sqrt()) ** 2))
            
            obj_confi_loss += torch.sum((masks[i] * (pred[:, [i * 5 + 4], :, :] - ious[i]) ** 2) * labels[:, [i * 5 + 4], :, :])

            noobj_confi_loss += 0.5 * torch.sum(((1 - labels[: , [i * 5 + 4], :, :]) * pred[:, [i * 5 + 4], : , : ]) ** 2 + 
                                                 labels[: , [i * 5 + 4], :, :] * (1 - masks[i]) * (pred[:, [i * 5 + 4], :, :] - ious[i]) ** 2)


        class_loss = torch.sum(((pred[:, b_num * 5:, :, :] - labels[:, b_num * 5:, :, :]) ** 2) * labels[:, [4], :, :].repeat(1, class_num, 1, 1))


        loss = obj_loss + class_loss + obj_confi_loss + noobj_confi_loss

        return loss / batchsize

if __name__ == "__main__":
    l = LOSS()
    pred = torch.Tensor(30, (5 * BOX_NUM + len(CLASSES)), 7, 7)
    labels = torch.Tensor(30, (5 * BOX_NUM + len(CLASSES)), 7, 7)
    loss = l.forward(pred, labels)
    print(loss.shape)
    print(loss)