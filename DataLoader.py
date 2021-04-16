import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json


class ChineseHandWriteDataset(Dataset):
    def __init__(self, root="", transform=None):
        self.data = []
        self.label = []
        for file in os.listdir(root):
            img_path = root + '/' + file
            im = Image.open(img_path).convert("RGB")
            self.data.append(np.array(im))
            Image.fromarray(self.data[-1]).show()
            self.label.append(img_path[-5:-4])
        self.transform = transform
        np.save('train_data.npy', self.data)
        np.save('train_label.npy', self.data)

    def __getitem__(self, index):
        return self.transform(self.data[index]), self.label[index]

    def __len__(self):
        return len(self.data)
