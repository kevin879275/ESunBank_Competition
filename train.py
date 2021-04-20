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

path = './data'
label_file = 'training data dic.txt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

Epoch = 100
BATCH_SIZE = 128
lr = 0.01

transform = transforms.Compose([
    transforms.ToTensor(),
])


def load_label_list(file):
    label = []
    f = open(file, 'r', encoding="utf-8")
    for line in f.readlines():
        label.append(line[0])
    return label


def main():
    label_list = load_label_list(label_file)
    dataset = ChineseHandWriteDataset(root=path, transform=transform)
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Model(in_features=16384).to(device)
    loss = SmoothCrossEntropyLoss().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    print("------------------ training start -----------------")

    total_loss = []
    total_correct = []

    for epoch in range(Epoch):
        since = time.time()
        running_loss = 0
        running_correct = 0
        for imgs, label in train_dataloader:
            imgs = imgs.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss_val = loss(out, label)
            _, pred_class = torch.max(out.data, 1)
            running_correct += torch.sum(pred_class == label)
            running_loss += loss_val
            loss_val.backward()
            optimizer.step()
        total_loss.append(running_loss.item() / 68804)
        total_correct.append(running_correct.item() / 68804)
        print("Epoch:{} Loss:{:.4f},  Correct:{:.4f}".format(
            epoch+1, total_loss[-1], total_correct[-1]))
        now_time = time.time() - since
        print("Training time is:{:.0f}m {:.0f}s".format(
            now_time // 60, now_time % 60))


if __name__ == "__main__":
    main()
