from dataset import VOCdataset
from resnet_yolo import YOLOv1_resnet
from yolo_loss import Yolov1Loss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import numpy as np
from torchvision import transforms as transforms
import torch
DATASET_PATH='./VOC2007/'
CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']
if __name__ == '__main__':
    epoch = 50
    batchsize = 5
    lr = 0.001
    
    train_data = VOCdataset(is_train=True)
    train_dataloader = DataLoader(train_data,batch_size=batchsize,shuffle=True)

    model = YOLOv1_resnet().cuda()
    # model.children()里是按模块(Sequential)提取的子模块，而不是具体到每个层，具体可以参见pytorch帮助文档
    # 冻结resnet34特征提取层，特征提取层不参与参数更新
    for layer in model.children():
        layer.requires_grad = False
        break
    criterion = Yolov1Loss()
    optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)

    is_tensorboard=True
    if is_tensorboard:
        writer=SummaryWriter(log_dir='yolov1_loss_curve')
    global_step=0
    for e in range(epoch):
        model.train()
        yl = torch.Tensor([0]).cuda()
        total_loss=0
        for i,(inputs,labels) in enumerate(train_dataloader):
            global_step+=1
            inputs = inputs.cuda()
            labels = labels.float().cuda()
            pred = model(inputs)
            loss = criterion(pred, labels)
            total_loss+=loss.data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Epoch %d/%d| Step %d/%d| Loss: %.2f"%(e,epoch,i,len(train_data)//batchsize,loss))
            
            writer.add_scalar('train loss',total_loss / (i + 1),global_step)
        if (e+1)%10==0:
            torch.save(model,"./yolo_models/YOLOv1_epoch"+str(e+1)+".pth")
            # compute_val_map(model)