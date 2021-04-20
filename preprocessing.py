import os
import numpy as np
from PIL import Image

root = './train_image'

data = []
label = []

max_width = 0
max_height = 0
i = 0

label_list = {}

if not os.path.isdir('data'):
    os.mkdir('data')
    os.mkdir('data/image')
    os.mkdir('data/label')

f = open('training data dic.txt', 'r', encoding="utf-8")
for idx, line in enumerate(f.readlines()):
    label_list[line] = idx


for idx, file in enumerate(os.listdir(root)):
    img_path = root + '/' + file
    im = Image.open(img_path).convert('L')
    im_data = np.array(im)
    # print(1)
    pad_height = int((524 - im_data.shape[0]) / 2)
    pad_width = int((524 - im_data.shape[1]) / 2)
    im_data = np.pad(im_data, pad_width=((pad_height, 524 - im_data.shape[0] - pad_height), (pad_width, 524 - im_data.shape[1] - pad_width)),
                     mode='constant', constant_values=255)
    # max_width = max(max_width, im.size[0])
    # max_height = max(max_height, im.size[1])
    data.append(im_data)
    # Image.fromarray(data[-1]).show()
    l = img_path[-5:-4]
    if l in label_list:
        label.append(label_list[l])
    else:
        label.append(801)
    if idx % 10000 == 9999 or idx == len(os.listdir(root)) - 1:
        np.save('/data/image/train_data_' + str(i)+'.npy', data)
        np.save('/data/label/train_label_'+str(i)+'.npy', label)
        i = i + 1
        data = []
        label = []
# print(max_width)
# print(max_height)
# np.save('train_data.npy', data)
# np.save('train_label.npy', label)
