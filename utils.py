import os
import re
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from efficientnet_pytorch import EfficientNet

from model import *

'''Argumentation'''

PrograssiveBounds={
    "efficientnetV2":
    {
        "s":
        {
            "mixm":0.,
            "mixM":0., 
            "dropm":0.1,
            "dropM":0.3,
            "randargm":5,
            "randargM":15,
            "imgsizem":128,
            "imgsizeM":300
        },
        "m":
        {
            "mixm":0.,
            "mixM":0.2, 
            "dropm":0.1,
            "dropM":0.4,
            "randargm":5,
            "randargM":20,
            "imgsizem":128,
            "imgsizeM":300
        },
        "l":
        {
            "mixm":0.,
            "mixM":0.4, 
            "dropm":0.1,
            "dropM":0.5,
            "randargm":5,
            "randargM":25,
            "imgsizem":128,
            "imgsizeM":300
        },
        "xl":
        {
            "mixm":0.,
            "mixM":0.6, 
            "dropm":0.1,
            "dropM":0.6,
            "randargm":5,
            "randargM":30,
            "imgsizem":128,
            "imgsizeM":300
        }
    }
}

def getFinalEpoch(args,CHECKPOINT_FOLDER):  # return last epoch num (final training saved)

    start_epoch = args.start_epoch
    p = Path(CHECKPOINT_FOLDER)
    if not p.exists():
        return None
    files = [x for x in filter(lambda x: re.match(
        f'.*EPOCH_\d+.pkl', x), os.listdir(CHECKPOINT_FOLDER))]
    nums = [int(re.match(r'EPOCH_(\d+).pkl', x).group(1)) for x in files]
    if not args.load_model: 
        return None
    if len(files) == 0:
        if start_epoch != -1:
            print(
                f"<Warning> No such a Start epoch checkpoint file #{start_epoch} exists, which is file {CHECKPOINT_FOLDER}EPOCH_{start_epoch}.pkl")
        return None

    if start_epoch == -1:
        return max(nums)
    
    if start_epoch in nums:
        return start_epoch
    else:
        print(
            f"<Warning> No such a Start epoch checkpoint file #{start_epoch} exists, which is file {CHECKPOINT_FOLDER}EPOCH_{start_epoch}.pkl")
    return None

def getModelPath(CHECKPOINT_FOLDER,args):
    num = getFinalEpoch(CHECKPOINT_FOLDER=CHECKPOINT_FOLDER,args=args)
    if num is not None:
        return f"{CHECKPOINT_FOLDER}EPOCH_{num}.pkl"
    return ""


def load_label_dic(label_path):
    label_dic = {}
    f = open(label_path, 'r', encoding="utf-8")
    for idx, line in enumerate(f.readlines()):
        label_dic[line[0]] = idx
    label_dic[idx+1] = "isnull"
    return label_dic

def load_word_dic(label_path):
    label_dic = {}
    f = open(label_path, 'r', encoding="utf-8")
    for idx, line in enumerate(f.readlines()):
        label_dic[idx] = line[0]
    label_dic[idx+1] = "isnull"
    return label_dic

def switchModel(in_features,num_classes,args,METHOD):
    if args.method == "efficientnet":
        model = EfficientNet.from_pretrained(
            METHOD, in_channels=1, num_classes=num_classes)
    elif METHOD == "regnet":
        model = RegNetx(in_features, num_classes,
                        model='regnety_002', pretrained=True)
    elif re.match(r'efficientnetV2', METHOD):
        model = efficientnetV2[args.method_level]()
    return model


def getWeights(root,split_rate,len_data):
    label_num = {}
    for i in range(len_data):
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

def prograssiveNow(epoch,num_epochs,boudDictofModel):
    pa = epoch/(num_epochs -1)
    args = ["mix","drop","randarg","imgsize"]
    ansDict={}
    for arg in args:
        ansDict[arg]=boudDictofModel[f"{arg}m"]+(boudDictofModel[f"{arg}M"]-boudDictofModel[f"{arg}m"])*pa
    return ansDict

def setDropout(model, drop_rate=0.1):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Dropout2d):
            child.p = drop_rate
        setDropout(child, drop_rate=drop_rate)

def mixup(x, y, q=0., use_cuda=True):
    ''' x: image in [B, C, H, W]'''
    ''' y: label in [B, CLASSES] '''
    ''' mixup alpha (0 = no mix, 1 = no original image), p=1-q , o = i * p + mixupI * q '''
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    q = max(q,0.)
    q = min(q,1.)  # force range of q in [0,1]

    q = random.random()*q # random value [0,q)
    p = 1-q # random value [0,1-q)
    
    batch_size = x.size()[0]
    if use_cuda: 
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = p * x + q * x[index, :]
    '''y : [1,3,5] size : [B] '''
    if p>q:
        mixed_y=y
    else :
        mixed_y=y[index]
  
    return mixed_x, mixed_y

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def padding(images):
    tensors, labels, paths1, paths2 = list(zip(*images))
    max_size = (max([t.shape[-2] for t in tensors]), max([t.shape[-1] for t in tensors]))
    batch_shape = [len(tensors)] + list(tensors[0].shape[:-2]) + list((max_size))
    
    batched_imgs = tensors[0].new_full(batch_shape, 245.0 / 255.0)
    for img, pad_img in zip(tensors, batched_imgs):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)

    return batched_imgs, torch.Tensor(labels).long(), paths1, paths2