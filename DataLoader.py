import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json

unsuitable_data = ['10412_家.jpg',
'18683_邦.jpg',
'21273_研.jpg',
'23165_學.jpg',
'31737_遠.jpg',
'34694_居.jpg',
'36591_茂.jpg',
'40121_苑.jpg',
'43255_普.jpg',
'4362_灣.jpg',
'48779_灣.jpg',
'57224_立.jpg',
'59937_興.jpg',
'60115_瑞.jpg',
'62199_格.jpg',
'62323_鈦.jpg',
'66241_悅.jpg']
class ChineseHandWriteDataset(Dataset):
    def __init__(self, root="", label_dic={}, transform=None, resize=True, resize_size=128):
        self.imgfile = []
        self.labelfile = []
        self.transform = transform
        self.root = root
        self.resize = resize
        self.resize_size = resize_size
        self.label_dic = label_dic
        self.img_file = os.listdir(self.root)
        # remove width < 10 or height < 10 image
        for i in unsuitable_data:
            if i in self.img_file:
                self.img_file.remove(i)
        # for dir in os.listdir(root):
        #     dir_path = root + '/' + dir
        #     if dir == 'image':
        #         for file in os.listdir(dir_path):
        #             img_path = dir_path + '/' + file
        #             self.imgfile.append(img_path)
        #             self.data = np.load(self.imgfile[0])
        #     elif dir == 'label':
        #         for file in os.listdir(dir_path):
        #             label_path = dir_path + '/' + file
        #             self.labelfile.append(label_path)
        #             self.label = np.load(self.labelfile[0])

    def __getitem__(self, index):
        img_path = self.root + '/' + self.img_file[index]
        img = Image.open(img_path).convert('L')
        label_chinese = img_path[-5:-4]
        if label_chinese in self.label_dic:
            label = self.label_dic[label_chinese]
        else:
            label = 801
        if self.resize:
            img = img.resize((self.resize_size, self.resize_size))

        return self.transform(img), label

    def __len__(self):
        return len(self.img_file)
