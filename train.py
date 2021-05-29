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
import xgboost
from utils import *
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
parser.add_argument("-sd", "--seed", type=int, default=1)  # spilt random Seed
parser.add_argument("-xgb", "--xgboost", type=str2bool, default=False) # use xgboost or not
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
PrograssiveModelDict=None
if args.method == "efficientnet" or args.method == "efficientnetV2":
    METHOD = f"{args.method}-{args.method_level}"
    PrograssiveModelDict = PrograssiveBounds[args.method][args.method_level]
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
is_useweight = True
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
    files = [x for x in filter(lambda x: re.match(
        f'.*EPOCH_\d+.pkl', x), os.listdir(CHECKPOINT_FOLDER))]
    nums = [int(re.match(r'EPOCH_(\d+).pkl', x).group(1)) for x in files]

    if len(files) == 0:
        if start_epoch != -1:
            print(
                f"<Warning> No such a Start epoch checkpoint file #{start_epoch} exists, which is file {CHECKPOINT_FOLDER}EPOCH_{start_epoch}.pkl")
        return None

    if start_epoch == -1:
        return max(nums)
    # search specific number
    if start_epoch in nums:
        return start_epoch
    else:
        print(
            f"<Warning> No such a Start epoch checkpoint file #{start_epoch} exists, which is file {CHECKPOINT_FOLDER}EPOCH_{start_epoch}.pkl")
    return None


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
    label_dic[800] = "isnull"
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


def getWeights(root):
    label_num = {}
    for i in range(801):
        label_num[str(i)] = None
    for idx, dir_ in enumerate(os.listdir(root)):
        nSamples = len(os.listdir(root + dir_))
        label_num[dir_] = nSamples * split_rate
    sorted_label_num = sorted(label_num.items(), key=lambda item: item[1])
    median_idx = int(len(sorted_label_num) / 2)
    median = sorted_label_num[median_idx][1]
    weights = []
    for i in label_num:
        weight = median / label_num[str(i)]
        weights.append(weight)
    weights = torch.FloatTensor(weights)
    return weights


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

    clean_image_path = './train_image/'
    # clean_transform = transforms.Compose([
    #     transforms.Grayscale(num_output_channels=1),
    #     transforms.Resize((resize_size, resize_size)),
    #     transforms.ToTensor(),
    # ])

    train_dataset = []
    valid_dataset = []
    for idx, dir_ in enumerate(os.listdir(clean_image_path)):
        dataset = ChineseHandWriteDataset(root=clean_image_path + dir_, label_dic=label_dic, transform=transform,
                                          resize=resize,
                                          resize_size=resize_size)
        train_set_size = int(len(dataset) * split_rate)
        valid_set_size = len(dataset) - train_set_size
        train_set, valid_set = data.random_split(dataset, [train_set_size, valid_set_size],
                                                 torch.Generator().manual_seed(args.seed))
        train_dataset.append(train_set)
        valid_dataset.append(valid_set)

    train_dataset = data.ConcatDataset(train_dataset)
    valid_dataset = data.ConcatDataset(valid_dataset)

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    valid_dataloader = DataLoader(
        valid_dataset, batch_size=valid_batch_size, pin_memory=True, num_workers=args.num_workers)

    print(f"model is {METHOD}")
    model = switchModel(in_features=train_dataset[0][0].shape[0])
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

    # get each class weight
    weights = None
    if is_useweight:
        weights = getWeights(root=clean_image_path)

    # Label smoothing
    # loss = SmoothCrossEntropyLoss(weight=weights).to(device)

    # Focal Loss
    loss = FocalLoss(weight=weights).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    print("------------------ training start -----------------")

    result_param = {'training_loss': [], 'training_accuracy': [],
                    'validation_loss': [], 'validation_accuracy': []}

    for epoch in range(START_EPOCH, Epoch):
        prograssive = None
        if PrograssiveModelDict is not None:
            prograssive = prograssiveNow(epoch, Epoch,PrograssiveModelDict)
        since = time.time()
        running_training_loss = 0
        running_training_correct = 0
        running_valid_loss = 0
        running_valid_correct = 0
        model.train()
        if prograssive is not None:
            train_dataloader.resize=int(prograssive["imgsize"])
        train_bar = tqdm(train_dataloader)
        for imgs, label in train_bar:
            imgs = imgs.to(device)
            label = label.to(device)
            if prograssive is not None:
                imgs, label = mixup(imgs, label, prograssive["mix"])
                setDropout(model, prograssive["drop"])
                
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
            running_training_loss.item() / len(train_dataset) * BATCH_SIZE)
        result_param['training_accuracy'].append(running_training_correct.item() /
                                                 len(train_dataset))
        result_param['validation_loss'].append(
            running_valid_loss.item() / len(valid_dataset) * valid_batch_size)
        result_param['validation_accuracy'].append(running_valid_correct.item() /
                                                   len(valid_dataset))

        print(
            "Epoch:{} Train Loss:{:.4f},  Train Accuracy:{:.4f},  Validation Loss:{:.4f},  Validation Accuracy:{:.4f}".format(
                epoch + 1, result_param['training_loss'][-1], result_param['training_accuracy'][-1],
                result_param['validation_loss'][-1], result_param['validation_accuracy'][-1]))

        now_time = time.time() - since
        print("Training time is:{:.0f}m {:.0f}s".format(
            now_time // 60, now_time % 60))

        torch.save(model.state_dict(), str(
            './checkpoints/' + METHOD + '/' + "EPOCH_" + str(epoch) + ".pkl"))
        out_file = open(str(
            './checkpoints/' + METHOD + '/' + 'result_param.json'), "w+")
        json.dump(result_param, out_file, indent=4)

    if args.xgboost:
        print("---------------Two stage - XGboost---------------------")
        with torch.no_grad():

            x_valid, y_valid = [], []
            val_bar = tqdm(valid_dataloader)
            for imgs, label in val_bar:
                imgs = imgs.to(device)
                label = label.to(device)
                # to numpy
                imgs = CustomPredict(model, imgs).cpu().detach().numpy()
                label = label.cpu().detach().numpy()
                if not len(x_valid):
                    x_valid, y_valid = imgs, label
                else:
                    x_valid, y_valid = np.concatenate((x_valid, imgs)), np.concatenate((y_valid, label))

            xgb_train, xgb_label = [], []
            train_bar = tqdm(train_dataloader)
            for imgs, label in train_bar:
                imgs = imgs.to(device)
                label = label.to(device)
                # to numpy
                imgs = CustomPredict(model, imgs).cpu().detach().numpy()
                label = label.cpu().detach().numpy()

                if not len(xgb_train):
                    xgb_train, xgb_label = imgs, label
                else:
                    xgb_train, xgb_label = np.concatenate((xgb_train, imgs)), np.concatenate((xgb_label, label))

            dval = xgboost.DMatrix(x_valid, y_valid)
            dtrain = xgboost.DMatrix(xgb_train, xgb_label)
            
            params = {
                'max_depth': 5,                 # the maximum depth of each tree
                'eta': lr,                     # the training step for each iteration
                'objective': 'multi:softmax',   # multiclass classification using the softmax objective
                'num_class': 801,                 # the number of classes that exist in this datset
                'updater' : 'grow_gpu_hist',
                'tree_method': 'gpu_hist',
            } 
            
            xgbmodel = xgboost.Booster()
            # xgbmodel.load_model('xgboost.model')
            xgbmodel = xgboost.train(params, dtrain, num_boost_round=100, evals=[(dval, 'val'), (dtrain, 'train')])        

            print(sum(xgbmodel.predict(dval) == y_valid) / len(y_valid))
            xgbmodel.save_model('xgboost.model')

if __name__ == "__main__":
    main()
