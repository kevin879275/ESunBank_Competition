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
import torchvision
import xgboost
from utils import *
from torch_poly_lr_decay import PolynomialLRDecay
from tqdm import tqdm
from pathlib import Path
from augmentations import RandAugment

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
    valid_batch_size = args.validbatchsize
    CHECKPOINT_FOLDER = args.checkpoint_root + METHOD + '/'
    START_EPOCH = getFinalEpoch(args=args,CHECKPOINT_FOLDER=CHECKPOINT_FOLDER) + 1 if getFinalEpoch(args=args,CHECKPOINT_FOLDER=CHECKPOINT_FOLDER) is not None else 0

    is_useweight = True
    print("init data folder")

    Path(CHECKPOINT_FOLDER).mkdir(exist_ok=True,parents=True)
    
    label_dic = load_label_dic(label_path)
    word_dic = load_word_dic(label_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    clean_image_path = './color_dataset/'
    synthesis_path = './synthesis/'
    # clean_transform = transforms.Compose([
    #     transforms.Grayscale(num_output_channels=1),
    #     transforms.Resize((resize_size, resize_size)),
    #     transforms.ToTensor(),
    # ])

    train_dataset = []
    valid_dataset = []
    for idx, dir_ in enumerate(os.listdir(clean_image_path)):
        # if args.pretrain_cleandataset:
        dataset = ChineseHandWriteDataset(root=clean_image_path + dir_, label_dic=label_dic, transform=transform, resize=resize,
                                    resize_size=resize_size)
            # dataset = CleanDataset(root=synthesis_path + dir_, label_dic=label_dic, transform=transform, resize=resize,
            #                             resize_size=resize_size, randaug=args.method=="efficientnetV2")
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
        valid_dataset, batch_size=valid_batch_size, pin_memory=False, num_workers=args.num_workers)

    print(f"model is {METHOD}")
    model = switchModel(in_features=train_dataset[0][0].shape[0],num_classes=num_classes,args=args,METHOD=METHOD)
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

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler_poly_lr_decay = PolynomialLRDecay(optimizer, max_decay_steps=100, end_learning_rate=args.ending_learning_rate, power=2.0)
    print("------------------ training start -----------------")

    result_param = {'training_loss': [], 'training_accuracy': [],
                    'validation_loss': [], 'validation_accuracy': []}
    
    for epoch in range(START_EPOCH, Epoch):
        batchI=0
        scheduler_poly_lr_decay.step()
        progressive = None
        if PrograssiveModelDict is not None:
            randaugment= RandAugment()
            
            progressive = prograssiveNow(epoch, Epoch,PrograssiveModelDict)
            randaugment.m=progressive["randarg"]
            

            
        since = time.time()
        running_training_loss = 0
        running_training_correct = 0
        running_valid_loss = 0
        running_valid_correct = 0
        dataset.train()
        model.train()

        train_bar = tqdm(train_dataloader)
        
        for imgst, label, folder, filename in train_bar:
            label = label.to(device)
            if progressive is not None:
                imgst, label = mixup(imgst, label, progressive["mix"])
                toPIL=transforms.ToPILImage()
                transform = transforms.Compose([
                    transforms.Resize((int(progressive["imgsize"]),int(progressive["imgsize"]))),
                    transforms.ToTensor(),
                ])
                imgs=torch.zeros((BATCH_SIZE,3,int(progressive["imgsize"]),int(progressive["imgsize"])))
                for i in range(imgst.size()[0]):
                    imgs[i]=transform(randaugment(toPIL(imgst[i])))
                imgs=imgs.to(device)
                # torchvision.utils.save_image(imgs,f"preprocessImgs/{epoch}-{batchI}.jpg")
                setDropout(model, progressive["drop"])
                
            optimizer.zero_grad()
            out = model(imgs)
            loss_val = loss(out, label)
            _, pred_class = torch.max(out.data, 1)
            running_training_correct += torch.sum(pred_class == label)
            running_training_loss += loss_val
            loss_val.backward()
            optimizer.step()
            train_bar.set_description(desc='[%d/%d] | Train Loss:%.4f' %
                                           (epoch + 1, Epoch, loss_val.item()/
                                                 len(imgs)))
        with torch.no_grad():
            dataset.eval()
            model.eval()
            if progressive is not None:
                setDropout(model,0)
            val_bar = tqdm(valid_dataloader)
            for imgs, label , folder, filename in val_bar:
                imgs = imgs.to(device)
                label = label.to(device)
                out = model(imgs)
                loss_val = loss(out, label)
                val_bar.set_description(desc='[%d/%d] | Validation Loss:%.4f' % (epoch + 1, Epoch, loss_val.item()/
                                                 len(imgs)))
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
    parser = argparse.ArgumentParser(
    description="ESun Competition HandWrite Recognition")
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-b", "--batchsize", type=int, default=32)
    parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-el", "--ending_learning_rate", type=float, default=0.00001)
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
    parser.add_argument("-cr","--checkpoint_root",type=str,default="./checkpoints/")
    parser.add_argument("-data","--dataset",type=str,default="")
    main(parser.parse_args())
