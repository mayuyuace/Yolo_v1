import torch
from torch.utils.data import DataLoader

from data_set import VOC2012
from net import YOLO_V1_resnet
from loss import LOSS
from generate_lables import generate_label_txt
from config import ROOT_PATH, EPOCHS, BATCH_SIZE, LR, DATASET_PATH

def train():
    epoch = EPOCHS
    batch_size = BATCH_SIZE
    lr = LR

    train_data = VOC2012(is_train=True)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    val_data = VOC2012(is_train=False)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    model = YOLO_V1_resnet().cuda()
    #model = YOLO_V1_resnet()
    criterion = LOSS()
    optimizer = torch.optim.SGD(filter(lambda x : x.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=0.0005)

    for e in range(epoch):
        print(f"train:{e+1} epoch")
        model.train()
        train_loss = torch.Tensor([0]).cuda()
        #total_loss = torch.Tensor([0])
        
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.cuda()
            labels = labels.float().cuda()
            #labels = labels.float()
            pred = model(inputs)
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss

        val_loss = torch.Tensor([0]).cuda()
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_dataloader):
                inputs = inputs.cuda()
                labels = labels.float().cuda()
                #labels = labels.float()
                pred = model(inputs)
                loss = criterion(pred, labels)
                val_loss = val_loss + loss
        
        print(f"loss:{train_loss / len(train_dataloader)}")

        print(f"val_loss:{val_loss / len(val_dataloader)}")
        
        if((e + 1) % 10 == 0):
            torch.save(model, f"{ROOT_PATH}/model_pkl/YOLO_v1_{e + 1}epoch.pkl")
            print("model saved!")

if __name__ == "__main__":
    generate_label_txt(DATASET_PATH)
    train()
