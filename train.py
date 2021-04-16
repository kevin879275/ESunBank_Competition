import os
import numpy as np
import torch
import torch.optim as optim
from torchsummary import summary
import torchvision.transforms as transforms
from DataLoader import ChineseHandWriteDataset

path = './data'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Epoch = 1
BATCH_SIZE = 1


def main():
    dataloader = ChineseHandWriteDataset(path)
    print(1)


if __name__ == "__main__":
    main()
