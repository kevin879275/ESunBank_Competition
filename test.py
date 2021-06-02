import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from model import *
import torchvision.transforms as transforms
from DataLoader import ChineseHandWriteDataset, CleanDataset
import time
import matplotlib.pyplot as plt
import argparse
import torch.utils.data as data
import json
from torchvision.datasets import ImageFolder
import xgboost
from utils import *
from torch_poly_lr_decay import PolynomialLRDecay
from tqdm import tqdm
from pathlib import Path


def main(args):
    
    # file path
    image_path = './train_image'
    path = './data'
    label_path = 'training data dic.txt'


    # Hyper Parameters
    PrograssiveModelDict=None
    if args.method == "efficientnet" or args.method == "efficientnetV2":
        METHOD = f"{args.method}-{args.method_level}"
        if args.method == "efficientnetV2":
            PrograssiveModelDict = PrograssiveBounds[args.method][args.method_level]
    elif args.method == 'regnet':
        METHOD = args.method
    else:
        METHOD = args.method + args.method_level

    # Environment
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        print('Warning! Using CPU.')

    Epoch = args.epochs
    BATCH_SIZE = args.batchsize
    lr = args.learning_rate
    split_rate = args.split_rate
    resize = args.resize
    resize_size = args.resize_size
    num_classes = 801

    CHECKPOINT_FOLDER = args.checkpoint_root + METHOD + '/'
    START_EPOCH = getFinalEpoch(args=args,CHECKPOINT_FOLDER=CHECKPOINT_FOLDER) + 1 if getFinalEpoch(args=args,CHECKPOINT_FOLDER=CHECKPOINT_FOLDER) is not None else 0

    is_useweight = True
    print("init data folder")

    Path(CHECKPOINT_FOLDER).mkdir(exist_ok=True,parents=True)
    
    label_dic = load_label_dic(label_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    clean_image_path = './train_image/'
    synthesis_path = './synthesis/'
    test_dataset="./EsunTestData/"
    # clean_transform = transforms.Compose([
    #     transforms.Grayscale(num_output_channels=1),
    #     transforms.Resize((resize_size, resize_size)),
    #     transforms.ToTensor(),
    # ])



    dataset = ChineseHandWriteDataset(root=test_dataset, label_dic=label_dic, transform=transform, resize=resize,
                                resize_size=resize_size, randaug=args.method=="efficientnetV2")
    # dataset = CleanDataset(root=synthesis_path + dir_, label_dic=label_dic, transform=transform, resize=resize,
    #                               resize_size=resize_size, randaug=args.method=="efficientnetV2")



    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=args.num_workers)



    print(f"model is {METHOD}")
    model = switchModel(in_features=dataset[0][0].shape[0],num_classes=num_classes,args=args,METHOD=METHOD)
    if args.load_model:
        modelPath = getModelPath(CHECKPOINT_FOLDER=CHECKPOINT_FOLDER,args=args)
        if modelPath != "":
            model.load_state_dict(torch.load(modelPath))



    model.to(device)


    # get each class weight
    weights = None
    if is_useweight:
        weights = getWeights(root=clean_image_path,split_rate=split_rate)

    # Label smoothing
    # loss = SmoothCrossEntropyLoss(weight=weights).to(device)

    # Focal Loss
    loss = FocalLoss(weight=weights).to(device)


    print("------------------ testing start -----------------")

    result_param = {'training_loss': [], 'training_accuracy': [],
                    'validation_loss': [], 'validation_accuracy': []}

        
    since = time.time()
    running_valid_loss = 0
    running_valid_correct = 0

    with torch.no_grad():
        dataset.eval()
        model.eval()
        val_bar = tqdm(dataloader)
        for imgs, label in val_bar:
        
            imgs = imgs.to(device)
            label = label.to(device)
            out = model(imgs)
            loss_val = loss(out, label)
            val_bar.set_description(desc='Test Loss:%.4f' % (loss_val.item()/
                                            len(imgs)))
            _, pred_class = torch.max(out.data, 1)
            running_valid_correct += torch.sum(pred_class == label)
            running_valid_loss += loss_val


    result_param['validation_loss'].append(
        running_valid_loss.item() / len(dataset) * BATCH_SIZE)
    result_param['validation_accuracy'].append(running_valid_correct.item() /
                                            len(dataset))

    print(
        "Test Loss:{:.4f},  Test Accuracy:{:.4f}".format(
            result_param['validation_loss'][-1], result_param['validation_accuracy'][-1]))

    now_time = time.time() - since






 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="ESun Competition HandWrite Recognition")
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-b", "--batchsize", type=int, default=1)
    parser.add_argument("-l", "--learning_rate", type=float, default=0.0001)
    parser.add_argument("-s", "--split_rate", type=float, default=0.8)
    parser.add_argument("-r", "--resize", type=int, default=True)
    parser.add_argument("-rs", "--resize_size", type=int, default=128)
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
    parser.add_argument("-cr","--checkpoint_root",type=str,default="./checkpoints/")
    main(parser.parse_args())
