import torchvision.models as tvmodel
import torch.nn as nn
import torch

from config import BOX_NUM, CLASSES

class YOLO_V1_resnet(nn.Module):
    def __init__(self):
        super(YOLO_V1_resnet, self).__init__()

        #使用预训练的resnet提取特征
        resnet = tvmodel.resnet34(pretrained=True)  
        #全连接层之前的输出通道数
        resnet_out_channel = resnet.fc.in_features
        #去除池化层和全连接层
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        
        for p in self.parameters():
            p.requires_grad = False

        #原yolov1的最后四个卷积层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(resnet_out_channel, 1024, 3, padding=1),
            #与原yolov1相比，多了BN层
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU()
        )
        #原yolov1的最后两个全连接层
        self.linear_layers = nn.Sequential(
            nn.Linear(7 * 7 * 1024,4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 7 * 7 * (5 * BOX_NUM + len(CLASSES))),
            nn.Sigmoid()
        )

    def forward(self, input):
        input = self.resnet(input)
        input = self.conv_layers(input)
        input = input.view(input.size()[0], -1)
        input = self.linear_layers(input)
        return input.reshape(-1, (5 * BOX_NUM + len(CLASSES)), 7, 7)

if __name__ == "__main__":
    net = YOLO_V1_resnet()
    input = torch.randn((20, 3, 448, 448))
    print(net)
    output = net(input)
    print(output.shape)
