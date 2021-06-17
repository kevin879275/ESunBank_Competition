import argparse
import json
import os
import time
from pathlib import Path
from PIL import Image
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
    
    test_dataset = 'EsunTestData/'
    # path_label = 'datasets/Esun_new_dic1_3.txt.'
    path_label = 'datasets/training data dic.txt'

    USE_PADDING = args.use_padding
    PADDING_FN = padding if USE_PADDING else None
    RESIZE_SIZE = args.resize_size
    RESIZE = False if RESIZE_SIZE == 0 or USE_PADDING else True

    NUM_WORKERS = args.num_workers
    WORD_TO_IDX_DICT = load_label_dic(path_label)
    IDX_TO_WORD_DICT = load_word_dic(path_label)
    TRNASFORM = transforms.Compose([transforms.ToTensor()])
    USE_RANDAUG = (args.method=="efficientnetV2")

    NUM_CLASSES = len(WORD_TO_IDX_DICT)

    ISNULL_THRESHOLD = args.threshold


    # ========================================================================================
    #   Data Loader
    # ========================================================================================
    dataset = NameDataset(
        root=test_dataset, label_dic=WORD_TO_IDX_DICT, transform=TRNASFORM, resize=RESIZE,
        resize_size=RESIZE_SIZE, randaug=USE_RANDAUG)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=NUM_WORKERS, collate_fn=PADDING_FN)
    
    # ========================================================================================
    #   Testing
    # ========================================================================================
    print(f"model is {METHOD}")
    model = switchModel(in_features=dataset[0][0].shape[0], num_classes=NUM_CLASSES, args=args, METHOD=METHOD)
    if args.load_model:
        modelPath = getModelPath(CHECKPOINT_FOLDER=CHECKPOINT_FOLDER, args=args)
        if modelPath != "":
            model.load_state_dict(torch.load(modelPath))
    model.to(device)


    print("------------------ Start Testing  -----------------")
    with torch.no_grad():
        model.eval()
        val_bar = tqdm(dataloader)
        
        sum_test_correct = 0
        for batch_img, batch_label, folder, filename, img_path in val_bar:
            batch_img, batch_label = batch_img.to(device), batch_label.to(device)
            
            output = model(batch_img)
            output = F.softmax(output, dim=1)
            
            pred_values, pred_classes = torch.max(output, 1)
            if ISNULL_THRESHOLD != -1:
                pred_classes[pred_values < ISNULL_THRESHOLD] = NUM_CLASSES - 1
            # batch_label[batch_label > 800] = NUM_CLASSES - 1
            # pred_classes[pred_classes > 800] = NUM_CLASSES - 1

            sum_test_correct += torch.sum(pred_classes == batch_label)

            # print()
            for i in range(len(batch_img)):
                # if pred_classes[i] != batch_label[i]:
                k = 1 
                pred_classes_topk_values, pred_classes_topk_indices = \
                    torch.topk(output.data, k, dim=1)
                
                fromp = f"{folder[i]}{filename[i]}"
                
                topk_probs = []
                probs = []
                for j in range(k):
                    pred_idx = pred_classes_topk_indices[i][j].item()
                    pred_word = IDX_TO_WORD_DICT[pred_idx]
                    value = pred_classes_topk_values[i][j].item()
                    word_prob = f"{pred_word}_{filename[i].split('.')[0].split('_')[1]}"
                    topk_probs.append(word_prob)
                    probs.append(round(value,3))
                topk_probs = ",".join(topk_probs) + ".png"
                # print(topk_probs)

                show_use_padding = lambda: "padding" if USE_PADDING else "resize"

                seg = 0
                if probs[0] < 0.2:
                    seg = '0-0.2'
                elif probs[0] < 0.5:
                    seg = '0.2-0.5'
                elif probs[0] < 0.75:
                    seg = '0.5-0.75'
                elif probs[0] < 0.9:
                    seg = '0.75-0.9'
                elif probs[0] < 0.95:
                    seg = '0.9-0.95'
                elif probs[0] < 0.99:
                    seg = '0.95-0.99'
                else:
                    seg = '0.99up'
                
                wront_output_folder = f'result/{show_use_padding()}-{ISNULL_THRESHOLD}-{seg}'
                wront_output_path = os.path.join(wront_output_folder, topk_probs)
                os.makedirs(wront_output_folder, exist_ok=True)
                ord_img = Image.open(img_path[i])
                ord_img.save(wront_output_path)

            val_bar.set_description()

    print("Test Accuracy:{:.4f}".format(sum_test_correct / len(dataset)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESun Competition HandWrite Recognition")
    
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-b", "--batchsize", type=int, default=32)
    parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-el", "--ending_learning_rate", type=float, default=0.00001)
    parser.add_argument("-s", "--split_rate", type=float, default=0.8)
    parser.add_argument("-rs", "--resize_size", type=int, default=128)
    parser.add_argument('--use_gpu', dest='use_gpu', type=str2bool, default=True, help='use gpu')
    parser.add_argument("-nw", "--num_workers", type=int, default=1)
    parser.add_argument("-sd", "--seed", type=int, default=1)  # spilt random seed
    parser.add_argument("-th", "--threshold", type=float, default=-1) # -1 to not set threshold
    parser.add_argument("--use-padding", type=str2bool, default=False)

    ### Checkpoint Path / Select Method ###
    ### final save name => method + method_level, e.g. efficientNet-b0
    parser.add_argument("-m", "--method", type=str, default="efficientnet") # Method save name and load name
    parser.add_argument("-ml", "--method_level", type=str, default="b7") # Method level e.g. b0, b1, b2, b3 or S, M, L
    
    ### Load Model Settings ###
    ### Load from epoch, -1 = final epoch in checkpoint
    parser.add_argument("-se", "--start_epoch", type=int, default = -1)
    parser.add_argument("-L", "--load_model", type=str2bool, default=True)  # Load model or train from 0
    parser.add_argument("-cr","--checkpoint_root", type=str, default="./checkpoints/")
    parser.add_argument("-data","--dataset", type=str, default="")
    
    main(parser.parse_args())