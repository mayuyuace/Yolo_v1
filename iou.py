import torch

def calculate_iou(bbox1, bbox2):
    '''
    bbox1(x1, y1, x2, y2) (batchsize, 4, 7, 7)
    bbox2(x3, y3, x4, y4) (batchsize, 4, 7, 7)
    '''

    #for inference
    #print(bbox1.shape)
    if len(bbox1.shape) < 4:
        bbox1 = bbox1.reshape(1, 4, 1, 1)
        bbox2 = bbox2.reshape(1, 4, 1, 1)

    #print(bbox1.shape)
    left1 = bbox1[:, [0], :, :]
    right1 = bbox1[:, [2], :, :]
    up1 = bbox1[:, [1], :, :]
    down1 = bbox1[:, [3], :, :]

    left2 = bbox2[:, [0], :, :]
    right2 = bbox2[:, [2], :, :]
    up2 = bbox2[:, [1], :, :]
    down2 = bbox2[:, [3], :, :]

    left = torch.max(left1, left2)
    right = torch.min(right1, right2)
    up = torch.max(up1, up2)
    down = torch.min(down1, down2)

    w = right - left
    w[w < 0] = 0
    h = down - up
    h[h < 0] = 0

    area = h * w
    all = (right1 - left1) * (down1 - up1) + (right2 - left2) * (down2 - up2) - area

    iou = area / all

    return iou

if __name__ == "__main__":
    box1 = torch.Tensor([[[[0]], [[0]], [[2]], [[2]]]])
    box2 = torch.Tensor([[[[0.5]], [[0.5]], [[1.5]], [[1.5]]]])

    iou = calculate_iou(box1, box2)

    print(iou)
    print(iou.shape)

    box1 = torch.Tensor([[[[0]], [[0]], [[2]], [[2]]]])
    box2 = torch.Tensor([[[[1]], [[1]], [[4]], [[4]]]])

    iou = calculate_iou(box1, box2)

    print(iou)
    print(iou.shape)

