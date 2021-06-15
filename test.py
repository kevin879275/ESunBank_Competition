import argparse
import json
import os
import time
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch_poly_lr_decay import PolynomialLRDecay

from model import *
from utils import *
from DataLoader import ChineseHandWriteDataset, CleanDataset, NameDataset, CommonWordDataset


def main(args):
    # ========================================================================================
    #   Variables
    # ========================================================================================
    if args.method == "efficientnet" or args.method == "efficientnetV2":
        METHOD = f"{args.method}-{args.method_level}"
    elif args.method == 'regnet':
        METHOD = args.method
    else:
        METHOD = args.method + args.method_level
    
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        print('Warning! Using CPU.')

    CHECKPOINT_FOLDER = args.checkpoint_root + METHOD + '/'
    
    BATCH_SIZE = args.batchsize
    
    test_dataset = 'ESunTestData/'
    path_label = 'datasets/training data dic.txt'
    
    NUM_WORKERS = args.num_workers
    WORD_TO_IDX_DICT = load_label_dic(path_label)
    IDX_TO_WORD_DICT = load_word_dic(path_label)
    RESIZE_SIZE = args.resize_size
    RESIZE = False if RESIZE_SIZE == 0 else True
    TRNASFORM = transforms.Compose([transforms.ToTensor()])
    USE_RANDAUG = (args.method=="efficientnetV2")

    NUM_CLASSES = len(WORD_TO_IDX_DICT)


    # ========================================================================================
    #   Data Loader
    # ========================================================================================
    dataset = ChineseHandWriteDataset(
        root=test_dataset, label_dic=WORD_TO_IDX_DICT, transform=TRNASFORM, resize=RESIZE,
        resize_size=RESIZE_SIZE, randaug=USE_RANDAUG)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=NUM_WORKERS)
    
    # ========================================================================================
    #   Testing
    # ========================================================================================
    print(f"model is {METHOD}")
    model = switchModel(in_features=dataset[0][0].shape[0],num_classes=NUM_CLASSES,args=args,METHOD=METHOD)
    if args.load_model:
        modelPath = getModelPath(CHECKPOINT_FOLDER=CHECKPOINT_FOLDER,args=args)
        if modelPath != "":
            model.load_state_dict(torch.load(modelPath))
    model.to(device)


    print("------------------ Start Testing  -----------------")
    with torch.no_grad():
        model.eval()
        val_bar = tqdm(dataloader)
        
        sum_test_correct = 0
        for batch_img, batch_label, _, _ in val_bar:
            batch_img, batch_label = batch_img.to(device), batch_label.to(device)
            output = model(batch_img)
            _, pred_class = torch.max(output, 1)
            
            sum_test_correct += torch.sum(pred_class == batch_label)

            val_bar.set_description()

    print("Test Accuracy:{:.4f}".format(sum_test_correct / len(dataset)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESun Competition HandWrite Recognition")
    
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-b", "--batchsize", type=int, default=16)
    parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-el", "--ending_learning_rate", type=float, default=0.00001)
    parser.add_argument("-s", "--split_rate", type=float, default=0.8)
    parser.add_argument("-rs", "--resize_size", type=int, default=128)
    parser.add_argument("-vb", "--validbatchsize", type=int, default=256)
    parser.add_argument('--use_gpu', dest='use_gpu', type=str2bool, default=True, help='use gpu')
    parser.add_argument("-nw", "--num_workers", type=int, default=1)
    parser.add_argument("-sd", "--seed", type=int, default=1)  # spilt random seed
    parser.add_argument("--use-padding", type=bool, default=True)                                                                                                                                                          )
    
    ### Checkpoint Path / Select Method ###
    ### final save name => method + method_level, e.g. efficientNet-b0
    parser.add_argument("-m", "--method", type=str, default="efficientnetV2") # Method save name and load name
    parser.add_argument("-ml", "--method_level", type=str, default="m") # Method level e.g. b0, b1, b2, b3 or S, M, L
    
    ### Load Model Settings ###
    ### Load from epoch, -1 = final epoch in checkpoint
    parser.add_argument("-se", "--start_epoch", type=int, default=-1)
    parser.add_argument("-L", "--load_model", type=str2bool, default=True)  # Load model or train from 0
    parser.add_argument("-cr","--checkpoint_root", type=str, default="./checkpoints/")
    parser.add_argument("-data","--dataset", type=str, default="")
    
    main(parser.parse_args())