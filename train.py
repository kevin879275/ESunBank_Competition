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
    
    EPOCH = args.epochs
    LAST_EPOCH = getFinalEpoch(args=args, CHECKPOINT_FOLDER=CHECKPOINT_FOLDER)
    START_EPOCH = LAST_EPOCH + 1 if LAST_EPOCH is not None else 0
    BATCH_SIZE = args.batchsize
    
    path_color_image = 'datasets/esun_v2_color/color_dataset/'
    path_synthesis_rgb = 'datasets/SynthesisDataRGB/'
    path_name = 'datasets/name/' 
    path_common_word = 'datasets/common_word/'
    path_label = 'datasets/Esun_common.txt'
    
    NUM_WORKERS = args.num_workers
    SEED = args.seed
    WORD_TO_IDX_DICT = load_label_dic(path_label)
    RESIZE_SIZE = args.resize_size
    RESIZE = False if RESIZE_SIZE == 0 else True
    TRNASFORM = transforms.Compose([transforms.ToTensor()])
    USE_RANDAUG = (args.method=="efficientnetV2")

    print("init data folder")
    Path(CHECKPOINT_FOLDER).mkdir(exist_ok=True, parents=True)
    
    LR = args.learning_rate
    ENDING_LR = args.ending_learning_rate
    SPLIT_RATE = args.split_rate
    NUM_CLASSES = len(WORD_TO_IDX_DICT)
    FL_USE_WEIGHT = False

    # ========================================================================================
    #   Data Loader
    # ========================================================================================
    dataset_path_list = [path_color_image, path_synthesis_rgb, path_common_word]
    loader_list = []

    for img_path in dataset_path_list:
        train_dataset, valid_dataset = [], []
        for _, dir_ in enumerate(os.listdir(img_path)):
            if img_path == path_color_image:
                dataset = ChineseHandWriteDataset(
                    root=img_path + dir_, label_dic=WORD_TO_IDX_DICT, transform=TRNASFORM, resize=RESIZE,
                    resize_size=RESIZE_SIZE, randaug=USE_RANDAUG)
            elif img_path == path_synthesis_rgb:
                dataset = CleanDataset(
                    root=img_path + dir_, label_dic=WORD_TO_IDX_DICT, transform=TRNASFORM, resize=RESIZE,
                    resize_size=RESIZE_SIZE, randaug=USE_RANDAUG)
            elif img_path == path_name:
                dataset = NameDataset(
                    root=img_path + dir_, label_dic=WORD_TO_IDX_DICT, transform=TRNASFORM, resize=RESIZE,
                    resize_size=RESIZE_SIZE, randaug=USE_RANDAUG)
            elif img_path == path_common_word:
                dataset = CommonWordDataset(
                    root=img_path + dir_, label_dic=WORD_TO_IDX_DICT, transform=TRNASFORM, resize=RESIZE,
                    resize_size=RESIZE_SIZE, randaug=USE_RANDAUG)
            else:
                raise NotImplementedError('Dataset Is Not Found.')
            
            train_set_size = int(len(dataset) * SPLIT_RATE)
            valid_set_size = len(dataset) - train_set_size
            
            train_set, valid_set = random_split(
                dataset, [train_set_size, valid_set_size], torch.Generator().manual_seed(SEED))
            train_dataset.append(train_set)
            valid_dataset.append(valid_set)
        
        train_dataset = ConcatDataset(train_dataset)
        valid_dataset = ConcatDataset(valid_dataset)
        print('In {}, Num of Train Data: {}, Num of Validation Data: {}'.format(img_path.split('/')[-2], len(train_dataset), len(valid_dataset)))

        train_dataloader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=NUM_WORKERS)
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=BATCH_SIZE, pin_memory=False, num_workers=NUM_WORKERS)

        loader_list.append((train_dataloader, valid_dataloader))

    # ========================================================================================
    #   Training
    # ========================================================================================
    print(f"model is {METHOD}")
    
    model = switchModel(in_features=3, num_classes=NUM_CLASSES, args=args, METHOD=METHOD)
    if args.load_model:
        modelPath = getModelPath(CHECKPOINT_FOLDER=CHECKPOINT_FOLDER,args=args)
        if modelPath != "":
            model.load_state_dict(torch.load(modelPath))
    model.to(device)
    
    loss_weights = None
    if FL_USE_WEIGHT:
        pass
    criterion = FocalLoss(weight=loss_weights).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler_poly_lr_decay = PolynomialLRDecay(optimizer, max_decay_steps=100, end_learning_rate=ENDING_LR, power=2.0)
    
    print("------------------ Start Training -----------------")
    result_param = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(START_EPOCH, EPOCH):
        scheduler_poly_lr_decay.step(epoch)
        
        for train_dataloader, valid_dataloader in loader_list:
            since = time.time()
            sum_train_loss, sum_train_correct, sum_val_loss, sum_val_correct = 0, 0, 0, 0

            model.train()
            train_bar = tqdm(train_dataloader)
            
            for batch_img, batch_label, _, _ in train_bar:
                batch_img, batch_label = batch_img.to(device), batch_label.to(device)
                output = model(batch_img)

                loss = criterion(output, batch_label)
                _, pred_class = torch.max(output, 1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                sum_train_loss += loss
                sum_train_correct += torch.sum(pred_class == batch_label)
                train_bar.set_description(
                    desc='[%d/%d] | Train Loss:%.4f' % (epoch + 1, EPOCH, loss.item() / len(batch_img)))
            
            with torch.no_grad():
                model.eval()
                val_bar = tqdm(valid_dataloader)
                
                for batch_img, batch_label, _, _ in val_bar:
                    batch_img, batch_label = batch_img.to(device), batch_label.to(device)
                    output = model(batch_img)
                    
                    loss = criterion(output, batch_label)
                    _, pred_class = torch.max(output, 1)

                    val_bar.set_description(
                        desc='[%d/%d] | Validation Loss:%.4f' % (epoch + 1, EPOCH, loss.item() / len(batch_img)))
                    
                    sum_val_loss += loss
                    sum_val_correct += torch.sum(pred_class == batch_label)
            
            result_param['train_loss'].append(
                sum_train_loss.item() / len(train_dataloader.dataset))
            result_param['train_acc'].append(
                sum_train_correct.item() / len(train_dataloader.dataset))
            result_param['val_loss'].append(
                sum_val_loss.item() / len(valid_dataloader.dataset))
            result_param['val_acc'].append(
                sum_val_correct.item() / len(valid_dataloader.dataset))

            print("Epoch:{} Train Loss:{:.4f}, Train Accuracy:{:.4f}, Validation Loss:{:.4f}, Validation Accuracy:{:.4f}, Learning Rate:{:.4f}".format(
                epoch + 1, 
                result_param['train_loss'][-1], result_param['train_acc'][-1],
                result_param['val_loss'][-1], result_param['val_acc'][-1], 
                optimizer.param_groups[0]['lr'])
            )

            now_time = time.time() - since
            print("Training time is:{:.0f}m {:.0f}s".format(now_time // 60, now_time % 60))

            # Save Files
            path_model_save = os.path.join('checkpoints', METHOD, 'EPOCH_' + str(epoch) + '.pkl')
            path_out_file = os.path.join('checkpoints', METHOD, 'result_param.json')
            torch.save(model.state_dict(), path_model_save)
            with open(path_out_file, "w+") as out_file:
                json.dump(result_param, out_file, indent=4)

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