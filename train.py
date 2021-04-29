import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from model import *
from RegNet import *
import torchvision.transforms as transforms
from DataLoader import ChineseHandWriteDataset
import time
import matplotlib.pyplot as plt
import argparse
import torch.utils.data as data
import json

## Efficient Net V1
# from efficientnet_pytorch import EfficientNet

try:
    from tqdm import tqdm
except ImportError:
    print('tqdm could not be imported. If you want to use progress bar during training,'
          'install tqdm from https://github.com/tqdm/tqdm.')

parser = argparse.ArgumentParser(
    description="ESun Competition HandWrite Recognition")
parser.add_argument("-e", "--epochs", type=int, default=100)
parser.add_argument("-se", "--start_epoch", type=int, default=0)
parser.add_argument("-b", "--batchsize", type=int, default=128)
parser.add_argument("-l", "--learning_rate", type=float, default=0.01)
parser.add_argument("-s", "--split_rate", type=float, default=0.8)
parser.add_argument("-r", "--resize", type=int, default=True)
parser.add_argument("-rs", "--resize_size", type=int, default=128)
parser.add_argument("-vb", "--validbatchsize", type=int, default=64)
parser.add_argument("-g", "--gpu", type=int, default=0)
args = parser.parse_args()

# file path
image_path = './train_image'
path = './data'
label_path = 'training data dic.txt'

# Environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# Hyper Parameters
METHOD = "REGNET"
Epoch = args.epochs
BATCH_SIZE = args.batchsize
lr = args.learning_rate
split_rate = args.split_rate
resize = args.resize
resize_size = args.resize_size
num_classes = 801
valid_batch_size = args.validbatchsize


def load_label_dic(label_path):
    label_dic = {}
    f = open(label_path, 'r', encoding="utf-8")
    for idx, line in enumerate(f.readlines()):
        label_dic[line[0]] = idx
    return label_dic


def main():
    print("init data folder")

    if not os.path.exists(str('./checkpoints')):
        os.mkdir('checkpoints')
    if not os.path.exists(str('./checkpoints/' + METHOD)):
        os.mkdir('./checkpoints/' + METHOD)

    label_dic = load_label_dic(label_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = ChineseHandWriteDataset(root=image_path, label_dic=label_dic, transform=transform, resize=resize,
                                      resize_size=resize_size)

    train_set_size = int(len(dataset)*split_rate)
    valid_set_size = len(dataset) - train_set_size
    train_dataset, valid_dataset = data.random_split(
        dataset, [train_set_size, valid_set_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)

    valid_dataloader = DataLoader(
        valid_dataset, batch_size=valid_batch_size, pin_memory=True, num_workers=4)

    in_features = dataset[0][0].shape[0]
    # for resnet
    # model = ResNet18(in_features=in_features, num_classes=num_classes, pretrained=False)
    # for regnet
    model = RegNetx(in_features, num_classes, model='regnety_320', pretrained=True)

    # Efficient Net V1 B0
    model = EfficientNet.from_pretrained("efficientnet-b0",in_channels=1,num_classes=801)
    model.cuda()

    # in_features = dataset[0][0].shape[1]*dataset[0][0].shape[2]
    # model = Model(in_features=in_features).to(device)
    # summary(model, (1, resize_size, resize_size))
    loss = SmoothCrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    print("------------------ training start -----------------")

    result_param = {'training_loss': [], 'training_accuracy': [],
                    'validation_loss': [], 'validation_accuracy': []}

    for epoch in range(Epoch):
        train_bar = tqdm(train_dataloader)
        since = time.time()
        running_training_loss = 0
        running_training_correct = 0
        running_valid_loss = 0
        running_valid_correct = 0
        model.train()
        for imgs, label in train_bar:
            imgs = imgs.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss_val = loss(out, label)
            _, pred_class = torch.max(out.data, 1)
            running_training_correct += torch.sum(pred_class == label)
            running_training_loss += loss_val
            loss_val.backward()
            optimizer.step()
        with torch.no_grad():
            model.eval()
            for imgs, label in valid_dataloader:
                imgs = imgs.to(device)
                label = label.to(device)
                out = model(imgs)
                loss_val = loss(out, label)
                _, pred_class = torch.max(out.data, 1)
                running_valid_correct += torch.sum(pred_class == label)
                running_valid_loss += loss_val

        result_param['training_loss'].append(
            running_training_loss.item() / train_set_size)
        result_param['training_accuracy'].append(running_training_correct.item() /
                                                 train_set_size)
        result_param['validation_loss'].append(
            running_valid_loss.item() / valid_set_size)
        result_param['validation_accuracy'].append(running_valid_correct.item() /
                                                   valid_set_size)

        print("Epoch:{} Train Loss:{:.4f},  Train Accuracy:{:.4f},  Validation Loss:{:.4f},  Validation Accuracy:{:.4f}".format(
            epoch+1, result_param['training_loss'][-1], result_param['training_accuracy'][-1], result_param['validation_loss'][-1], result_param['validation_accuracy'][-1]))

        now_time = time.time() - since
        print("Training time is:{:.0f}m {:.0f}s".format(
            now_time // 60, now_time % 60))

        train_bar.set_description(desc='[%d/%d] | Train Loss:%.4f | Train Accuracy:%.4f | Validation Loss:%.4f '
                                       '| Validation Accuracy:%.4f | Training time:%.0f' %
                                       (epoch+1, Epoch, result_param['training_loss'][-1], result_param['training_accuracy'][-1],
                                        result_param['validation_loss'][-1],
                                        result_param['validation_accuracy'][-1], now_time))

        torch.save(model.state_dict(), str(
            './checkpoints/' + METHOD + '/' + "EPOCH_" + str(epoch) + ".pkl"))
        out_file = open(str(
            './checkpoints/' + METHOD + '/' + 'result_param.json'), "w+")
        json.dump(result_param, out_file, indent=4)

    plt.plot(range(1, Epoch + 1), total_training_loss)
    plt.title("loss value")
    plt.ylabel("loss value")
    plt.xlabel("epoch")
    plt.show()


if __name__ == "__main__":
    main()
