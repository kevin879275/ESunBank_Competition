import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json


class ChineseHandWriteDataset(Dataset):
    def __init__(self, root="", transform=None):
        self.data = []
        self.label = []
        for dir in os.listdir(root):
            dir_path = root + '/' + dir
            if dir == 'image':
                for file in os.listdir(dir_path):
                    img_path = dir_path + '/' + file
                    if len(self.data) == 0:
                        self.data = np.load(img_path)
                    else:
                        self.data = np.concatenate(
                            (self.data, np.load(img_path)), axis=0)
                    # img_path = root + '/' + file
                    # im = Image.open(img_path).convert("RGB")
                    # self.data.append(np.array(im))
                    # Image.fromarray(self.data[-1]).show()
            elif dir == 'label':
                for file in os.listdir(dir_path):
                    label_path = dir_path + '/' + file
                    if len(self.label) == 0:
                        self.label = np.load(label_path)
                    else:
                        self.label = np.concatenate(
                            (self.label, np.load(label_path)), axis=0)
        print(1)
        # self.label.append(img_path[-5:-4])
        # self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.data[index]), self.label[index]

    def __len__(self):
        return len(self.data)
