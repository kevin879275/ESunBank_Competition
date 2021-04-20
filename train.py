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

Epoch = 1
BATCH_SIZE = 64
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
    loss = SmoothCrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("------------------ training start -----------------")
    for epoch in range(Epoch):
        since = time.time()
        loss_val_list = []
        total_loss = 0
        for imgs, label in train_dataloader:
            imgs = imgs.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss_val = loss(out, label)
            total_loss += loss_val
            loss_val.backward()
            optimizer.step()
        loss_val_list.append(total_loss.item() / len(train_dataloader))
        print("Epoch = ", epoch + 1, " loss = ", loss_val_list[-1])
        now_time = time.time() - since
        print("Training time is:{:.0f}m {:.0f}s".format(
            now_time // 60, now_time % 60))


if __name__ == "__main__":
    main()
