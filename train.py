import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from model import Model, SmoothCrossEntropyLoss
import torchvision.transforms as transforms
from DataLoader import ChineseHandWriteDataset
import time
import matplotlib.pyplot as plt
import argparse
import torch.utils.data as data

parser = argparse.ArgumentParser(
    description="ESun Competition HandWrite Recognition")
parser.add_argument("-e", "--epochs", type=int, default=100)
parser.add_argument("-se", "--start_epoch", type=int, default=0)
parser.add_argument("-b", "--batchsize", type=int, default=128)
parser.add_argument("-l", "--learning_rate", type=float, default=0.01)
parser.add_argument("-s", "--split_rate", type=float, default=0.8)
parser.add_argument("-r", "--resize", type=int, default=True)
parser.add_argument("-rs", "--resize_size", type=int, default=128)
# parser.add_argument("-g", "--gpu", type=int, default=0)
args = parser.parse_args()

# file path
image_path = './train_image'
path = './data'
label_path = 'training data dic.txt'

# Environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Hyper Parameters
Epoch = args.epochs
BATCH_SIZE = args.batchsize
lr = args.learning_rate
split_rate = args.split_rate
resize = args.resize
resize_size = args.resize_size

# get number of image
# def get_image_size():
#     return len(os.listdir(image_path))


def load_label_dic(label_path):
    label_dic = {}
    f = open(label_path, 'r', encoding="utf-8")
    for idx, line in enumerate(f.readlines()):
        label_dic[line[0]] = idx
    return label_dic


def main():
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
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=8)

    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_set_size)

    in_features = dataset[0][0].shape[1]*dataset[0][0].shape[2]
    model = Model(in_features=in_features).to(device)
    loss = SmoothCrossEntropyLoss().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    print("------------------ training start -----------------")
    total_training_loss = []
    total_training_correct = []
    total_valid_loss = []
    total_valid_correct = []
    for epoch in range(Epoch):
        since = time.time()
        running_training_loss = 0
        running_training_correct = 0
        running_valid_loss = 0
        running_valid_correct = 0
        for imgs, label in train_dataloader:
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
        for imgs, label in valid_dataloader:
            imgs = imgs.to(device)
            label = label.to(device)
            out = model(imgs)
            loss_val = loss(out, label)
            _, pred_class = torch.max(out.data, 1)
            running_valid_correct += torch.sum(pred_class == label)
            running_valid_loss += loss_val
        total_training_loss.append(
            running_training_loss.item() / train_set_size)
        total_training_correct.append(running_training_correct.item() /
                                      train_set_size)
        total_valid_loss.append(
            running_valid_loss.item() / valid_set_size)
        total_valid_correct.append(running_valid_correct.item() /
                                   valid_set_size)
        print("Epoch:{} Train Loss:{:.4f},  Train Accuracy:{:.4f},  Validation Loss:{:.4f},  Validation Accuracy:{:.4f}".format(
            epoch+1, total_training_loss[-1], total_training_correct[-1], total_valid_loss[-1], total_valid_correct[-1]))
        now_time = time.time() - since
        print("Training time is:{:.0f}m {:.0f}s".format(
            now_time // 60, now_time % 60))

    plt.plot(range(1, Epoch + 1), total_training_loss)
    plt.title("loss value")
    plt.ylabel("loss value")
    plt.xlabel("epoch")
    plt.show()


if __name__ == "__main__":
    main()
