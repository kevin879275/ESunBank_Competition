import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from model import *
import torchvision.transforms as transforms
from DataLoader import ChineseHandWriteDataset
import time
import matplotlib.pyplot as plt
import argparse
import torch.utils.data as data
import json
from torchvision.datasets import ImageFolder
from pathlib import Path
#from xgboost import XGBClassifier

# Efficient Net V1
from efficientnet_pytorch import EfficientNet

try:
    from tqdm import tqdm
except ImportError:
    print('tqdm could not be imported. If you want to use progress bar during training,'
          'install tqdm from https://github.com/tqdm/tqdm.')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(
    description="ESun Competition HandWrite Recognition")
parser.add_argument("-e", "--epochs", type=int, default=100)
parser.add_argument("-b", "--batchsize", type=int, default=32)
parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
parser.add_argument("-s", "--split_rate", type=float, default=0.8)
parser.add_argument("-r", "--resize", type=int, default=True)
parser.add_argument("-rs", "--resize_size", type=int, default=128)
parser.add_argument("-vb", "--validbatchsize", type=int, default=32)
parser.add_argument('--use_gpu', dest='use_gpu',
                    type=str2bool, default=True, help='use gpu')
parser.add_argument("-nw", "--num_workers", type=int, default=1)
parser.add_argument("-sd", "--seed", type=int, default=1)  # spilt random Seed
### Checkpoint Path / Select Method ###
# Method save name and load name
parser.add_argument("-m", "--method", type=str, default="efficientnetV2")
# Method level e.g. b0, b1, b2, b3 or S, M, L
parser.add_argument("-ml", "--method_level", type=str, default="m")
# final save name => method + method_level , e.g. efficientNetb0

### Load Model Settings ###
# Load from epoch, -1 = final epoch in checkpoint
parser.add_argument("-se", "--start_epoch", type=int, default=-1)
parser.add_argument("-L", "--load_model", type=str2bool,
                    default=False)  # Load model or train from 0

args = parser.parse_args()

# file path
image_path = './train_image'
path = './data'
label_path = 'training data dic.txt'


# Hyper Parameters
if args.method == "efficientnet" or args.method == "efficientnetV2":
    METHOD = f"{args.method}-{args.method_level}"
elif args.method == 'regnet':
    METHOD = args.method
else:
    METHOD = args.method + args.method_level
Epoch = args.epochs
BATCH_SIZE = args.batchsize
lr = args.learning_rate
split_rate = args.split_rate
resize = args.resize
resize_size = args.resize_size
num_classes = 801
valid_batch_size = args.validbatchsize
START_EPOCH = 0
CHECKPOINT_FOLDER = './checkpoints/' + METHOD + '/'
# Environment
if args.use_gpu and torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')
    print('Warning! Using CPU.')


def getFinalEpoch():  # return last epoch num (final training saved)

    start_epoch = args.start_epoch
    p = Path(CHECKPOINT_FOLDER)
    if not p.exists():
        return None
    files = [x for x in filter(lambda x:re.match(
        f'.*EPOCH_\d+.pkl', x), os.listdir(CHECKPOINT_FOLDER))]
    if start_epoch == -1:
        start_epoch = len(files)-1
    if start_epoch > len(files)-1:
        print(f"input start epoch {start_epoch} no exist model")
        return None
    if start_epoch == -1:
        start_epoch = 0
    for file in files:
        r = re.match(r'EPOCH_(\d+).pkl', file)
        num = int(r.groups(1)[0])
        if num == start_epoch:
            return num
    return None


'''Argumentation'''

def mixup(x, y, alpha=1.0, use_cuda=args.use_gpu):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]

    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y

def getModelPath():
    num = getFinalEpoch()
    if num is not None:
        return f"{CHECKPOINT_FOLDER}EPOCH_{num}.pkl"
    return ""


def load_label_dic(label_path):
    label_dic = {}
    f = open(label_path, 'r', encoding="utf-8")
    for idx, line in enumerate(f.readlines()):
        label_dic[line[0]] = idx
    return label_dic


def switchModel(in_features=0):
    if args.method == "efficientnet":
        model = EfficientNet.from_pretrained(
            METHOD, in_channels=1, num_classes=num_classes)
    elif METHOD == "regnet":
        model = RegNetx(in_features, num_classes,
                        model='regnety_002', pretrained=True)
    elif re.match(r'efficientnetV2', METHOD):
        model = efficientnetV2[args.method_level]()
        #
        # model = globals()[METHOD](num_classes=num_classes)
    return model


START_EPOCH = getFinalEpoch() + 1 if getFinalEpoch() is not None else 0


def main():
    print("init data folder")

    label_dic = load_label_dic(label_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    clean_image_path = './data/'
    clean_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor(),
    ])
    #dataset = ImageFolder(clean_image_path, transform=clean_transform)
    dataset = ChineseHandWriteDataset(root=image_path, label_dic=label_dic, transform=transform, resize=resize,
                                      resize_size=resize_size)

    test_dataloader = DataLoader(
        dataset, batch_size=1, pin_memory=True, num_workers=args.num_workers)

    print(f"model is {METHOD}")
    model = switchModel(in_features=dataset[0][0].shape[0])
    if args.load_model:
        modelPath = getModelPath()
        if modelPath != "":
            model.load_state_dict(torch.load(modelPath))

    model.to(device)

    loss = SmoothCrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    print("------------------ testing start -----------------")

    with torch.no_grad():
        model.eval()
        accuracy = 0
        for imgs, label in test_dataloader:
            imgs = imgs.to(device)
            label = label.to(device)
            out = model(imgs)
            loss_val = loss(out, label)
            _, pred_class = torch.max(out.data, 1)
            accuracy += torch.sum(pred_class == label)

    print("Accuracy = ", accuracy)


if __name__ == "__main__":
    main()
