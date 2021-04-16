import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json


class ChineseHandWriteDataset(Dataset):
    def __init__(self, root="", transform=None):
        self.imgfile = []
        self.labelfile = []
        self.transform = transform
        for dir in os.listdir(root):
            dir_path = root + '/' + dir
            if dir == 'image':
                for file in os.listdir(dir_path):
                    img_path = dir_path + '/' + file
                    self.imgfile.append(self.imgfile)
            elif dir == 'label':
                for file in os.listdir(dir_path):
                    label_path = dir_path + '/' + file
                    self.labelfile.append(label_path)

    def __getitem__(self, index):
        return self.transform(np.load(self.imgfile[index//10000][index % 10000])), np.load(self.labelfile[index//10000][index % 10000])

    def __len__(self):
        return 1
