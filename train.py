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
from xgboost import XGBClassifier

##### Efficient Net V1
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
parser.add_argument('--use_gpu', dest='use_gpu', type=str2bool, default=True, help='use gpu')
parser.add_argument("-nw", "--num_workers", type=int, default=1)
parser.add_argument("-sd", "--seed", type=int, default=1) # spilt random Seed 
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
                    default=True)  # Load model or train from 0

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
START_EPOCH =  0
CHECKPOINT_FOLDER = './checkpoints/' + METHOD + '/'
# Environment
if args.use_gpu and torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')
    print('Warning! Using CPU.')


def getFinalEpoch(): #return last epoch num (final training saved)

    start_epoch = args.start_epoch
    p=Path(CHECKPOINT_FOLDER)
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




def getModelPath():
    num=getFinalEpoch()
    if num is not None :
        return f"{CHECKPOINT_FOLDER}EPOCH_{num}.pkl"
    return ""


def load_label_dic(label_path):
    label_dic = {}
    f = open(label_path, 'r', encoding="utf-8")
    for idx, line in enumerate(f.readlines()):
        label_dic[line[0]] = idx
    label_dic[800] = "is_null"
    return label_dic


def switchModel(in_features = 0):
    if args.method == "efficientnet":
        model = EfficientNet.from_pretrained(
            METHOD, in_channels=1, num_classes=num_classes)
    elif METHOD == "regnet":
        model = RegNetx(in_features, num_classes,
                model='regnety_002', pretrained=True)
    elif re.match(r'efficientnetV2',METHOD):
        model = efficientnetV2[args.method_level]()
        #
        # model = globals()[METHOD](num_classes=num_classes)
    return model

START_EPOCH = getFinalEpoch() + 1 if getFinalEpoch() is not None else 0

def main():
    print("init data folder")

    if not os.path.exists(str('./checkpoints')):
        os.mkdir('checkpoints')
    if not os.path.exists(str('./checkpoints/' + METHOD)):
        os.mkdir('./checkpoints/' + METHOD)

    label_dic = load_label_dic(label_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    clean_image_path = './data/'
    clean_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((resize_size,resize_size)),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(clean_image_path,transform=clean_transform)
    # dataset = ChineseHandWriteDataset(root=image_path, label_dic=label_dic, transform=transform, resize=resize,
    #                                   resize_size=resize_size)
        
    train_set_size = int(len(dataset)*split_rate)
    valid_set_size = len(dataset) - train_set_size
    train_dataset, valid_dataset = data.random_split(dataset, [train_set_size, valid_set_size],torch.Generator().manual_seed(args.seed))


    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    valid_dataloader = DataLoader(
         valid_dataset, batch_size=valid_batch_size, pin_memory=True, num_workers=args.num_workers)

    print(f"model is {METHOD}")
    model = switchModel(in_features = dataset[0][0].shape[0])
    if args.load_model:
        modelPath = getModelPath()
        if modelPath != "":
            model.load_state_dict(torch.load(modelPath))

    # for resnet
    # model = ResNet18(in_features=in_features, num_classes=num_classes, pretrained=False)
    # for regnet


    # Efficient Net V1 B0
    # model = EfficientNet.from_pretrained("efficientnet-b0",in_channels=1,num_classes=801)

    model.to(device)
    # in_features = dataset[0][0].shape[1]*dataset[0][0].shape[2]
    # model = Model(in_features=in_features).to(device)
    # summary(model, (1, resize_size, resize_size))
    loss = SmoothCrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    print("------------------ training start -----------------")

    result_param = {'training_loss': [], 'training_accuracy': [],
                    'validation_loss': [], 'validation_accuracy': []}

    for epoch in range(START_EPOCH, Epoch):
        since = time.time()
        running_training_loss = 0
        running_training_correct = 0
        running_valid_loss = 0
        running_valid_correct = 0
        model.train()
        train_bar = tqdm(train_dataloader)
        for imgs, label in train_bar:
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
            train_bar.set_description(desc='[%d/%d] | Train Loss:%.4f' %
                                           (epoch + 1, Epoch, loss_val.item()))
        with torch.no_grad():
            model.eval()
            val_bar = tqdm(valid_dataloader)
            for imgs, label in val_bar:
                imgs = imgs.to(device)
                label = label.to(device)
                out = model(imgs)
                loss_val = loss(out, label)
                val_bar.set_description(desc='[%d/%d] | Validation Loss:%.4f' % (epoch + 1, Epoch, loss_val.item()))
                _, pred_class = torch.max(out.data, 1)
                running_valid_correct += torch.sum(pred_class == label)
                running_valid_loss += loss_val

        result_param['training_loss'].append(
            running_training_loss.item() / train_set_size)
        result_param['training_accuracy'].append(running_training_correct.item() /
                                                 train_set_size)
        result_param['validation_loss'].append(
            running_valid_loss.item() / valid_set_size)
        result_param['validation_accuracy'].append(running_valid_correct.item() /
                                                   valid_set_size)

        print("Epoch:{} Train Loss:{:.4f},  Train Accuracy:{:.4f},  Validation Loss:{:.4f},  Validation Accuracy:{:.4f}".format(
            epoch+1, result_param['training_loss'][-1], result_param['training_accuracy'][-1], result_param['validation_loss'][-1], result_param['validation_accuracy'][-1]))

        now_time = time.time() - since
        print("Training time is:{:.0f}m {:.0f}s".format(
            now_time // 60, now_time % 60))

        torch.save(model.state_dict(), str(
            './checkpoints/' + METHOD + '/' + "EPOCH_" + str(epoch) + ".pkl"))
        out_file = open(str(
            './checkpoints/' + METHOD + '/' + 'result_param.json'), "w+")
        json.dump(result_param, out_file, indent=4)


if __name__ == "__main__":
    main()
