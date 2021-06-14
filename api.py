from argparse import ArgumentParser
import base64
import datetime
import hashlib

import cv2
from flask import Flask
from flask import request
from flask import jsonify
import numpy as np
app = Flask(__name__)
####### PUT YOUR INFORMATION HERE #######
CAPTAIN_EMAIL = 'gevin879275@gmail.com'          #
SALT = 'nawanawa'                        #
#########################################


from PIL import Image
from io import BytesIO
import base64
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
from testDataStealer import Stealer
from pathlib import Path
#### Efficient Net V1
from efficientnet_pytorch import EfficientNet
from utils import getFinalEpoch
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
parser.add_argument("-b", "--batchsize", type=int, default=128)
parser.add_argument("-l", "--learning_rate", type=float, default=0.01)
parser.add_argument("-s", "--split_rate", type=float, default=0.8)
parser.add_argument("-r", "--resize", type=int, default=True)
parser.add_argument("-rs", "--resize_size", type=int, default=128)
parser.add_argument("-vb", "--validbatchsize", type=int, default=64)
parser.add_argument('--use_gpu', dest='use_gpu', type=str2bool, default=True, help='use gpu')
parser.add_argument("-nw", "--num_workers", type=int, default=2)

### Checkpoint Path / Select Method ###
# Method save name and load name
parser.add_argument("-m", "--method", type=str, default="efficientnet")
# Method level e.g. b0, b1, b2, b3 or S, M, L
parser.add_argument("-ml", "--method_level", type=str, default="b7")
# final save name => method + method_level , e.g. efficientNetb0

### Load Model Settings ###
# Load from epoch, -1 = final epoch in checkpoint
parser.add_argument("-se", "--start_epoch", type=int, default=-1)
parser.add_argument("-L", "--load_model", type=str2bool,
                    default=True)  # Load model or train from 0

### Flask Args ###
parser.add_argument('-p', '--port', default=80, help='port')
parser.add_argument('-d', '--debug', default=False, help='debug')
parser.add_argument("-th", "--threshold", type=float, default=0.7)
args = parser.parse_args()

# file path
image_path = './train_image'
path = './data'
label_path = 'training data dic.txt'


# Hyper Parameters
if args.method == "efficientnet":
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
ISNULL_THRESHOLD = args.threshold
## save data from ESunTest api request
stealer = Stealer() 
stealer.start()

def load_label_dic(label_path):
    label_dic = {}
    f = open(label_path, 'r', encoding="utf-8")
    for idx, line in enumerate(f.readlines()):
        label_dic[idx] = line[0]
    label_dic[800] = "isnull"
    return label_dic

label_dic = load_label_dic(label_path)
# Environment
if args.use_gpu and torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')
    print('Warning! Using CPU.')


def getWordFromResult(result):
    _, pred_class = torch.max(result.data, 1)
    return label_dic[pred_class.item()]



def getModelPath():
    CHECKPOINT_FOLDER = './checkpoints/' + METHOD + '/'
    num = getFinalEpoch(CHECKPOINT_FOLDER=CHECKPOINT_FOLDER,args=args)
    if num is not None:
        return f"{CHECKPOINT_FOLDER}EPOCH_{num}.pkl"
    return ""




def switchModel(in_features = 0):
    if args.method == "efficientnet":
        model = EfficientNet.from_pretrained(
            METHOD, in_channels=1, num_classes=num_classes)
    elif METHOD == "regnet":
        model = RegNetx(in_features, num_classes,
                model='regnety_002', pretrained=True)
    elif METHOD == "efficientnetV2":
        model = efficientnetV2[args.method_level]()
        #
        # model = globals()[METHOD](num_classes=num_classes)
    return model
def evalModel():
    model = switchModel(in_features = 1)
    if args.load_model:
        modelPath = getModelPath()
        if modelPath != "":
            model.load_state_dict(torch.load(modelPath))
        else:
            print("<load model error>Check whether your method and method_level setting is right. Or set load_model as False without try to load checkpoint model.")
            exit(-1)

    model.to(device)
    model.eval()
    return model
model=evalModel()














def generate_server_uuid(input_string):
    """ Create your own server_uuid.

    @param:
        input_string (str): information to be encoded as server_uuid
    @returns:
        server_uuid (str): your unique server_uuid
    """
    s = hashlib.sha256()
    data = (input_string + SALT).encode("utf-8")
    s.update(data)
    server_uuid = s.hexdigest()
    return server_uuid


def base64_to_binary_for_cv2(image_64_encoded):
    """ Convert base64 to numpy.ndarray for cv2.

    @param:
        image_64_encode(str): image that encoded in base64 string format.
    @returns:
        image(numpy.ndarray): an image.
    """
    img_base64_binary = image_64_encoded.encode("utf-8")
    img_binary = base64.b64decode(img_base64_binary)
    image = cv2.imdecode(np.frombuffer(img_binary, np.uint8), cv2.IMREAD_COLOR)
    return image


def predict(image):
    """ Predict your model result.

    @param:
        image (numpy.ndarray): an image.
    @returns:
        prediction (str): a word.
    """

    ####### PUT YOUR MODEL INFERENCING CODE HERE #######
    prediction = '陳'


    ####################################################
    if _check_datatype_to_string(prediction):
        return prediction


def _check_datatype_to_string(prediction):
    """ Check if your prediction is in str type or not.
        If not, then raise error.

    @param:
        prediction: your prediction
    @returns:
        True or raise TypeError.
    """
    if isinstance(prediction, str):
        return True
    raise TypeError('Prediction is not in string type.')


def transformImage(pilImg):
    

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = pilImg.convert('L')
    image = image.resize((args.resize_size, args.resize_size))
    image = transform(image)
    return image
@app.route("/")
def index():
    return f" Server runing model {METHOD}"
@app.route('/inference', methods=['POST'])
def inference():
    """ API that return your model predictions when E.SUN calls this API. """
    data = request.get_json(force=True)

    # 自行取用，可紀錄玉山呼叫的 timestamp
    esun_timestamp = data['esun_timestamp']

    # 取 image(base64 encoded) 並轉成 cv2 可用格式
    image_64_encoded = data['image']
    image = Image.open(BytesIO(base64.b64decode(image_64_encoded))) #PIL

    stealer.pause=True
    stealer.imgs.append(image.copy())

    image =  transformImage(image)
    image = torch.tensor(image).to(device).unsqueeze(0)#batch
    
    output = model(image)
    if ISNULL_THRESHOLD != -1:
        output = F.softmax(output,dim = 1)
        pred_values, pred_classes = torch.max(output, dim=1)
        pred_classes[pred_values < ISNULL_THRESHOLD] = 800 #NUM_CLASSES
    result = output
    t = datetime.datetime.now()
    ts = str(int(t.utcnow().timestamp()))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL + ts)

    try:

        result = model(image)
    except TypeError as type_error:
        # You can write some log...
        raise type_error
    except Exception as e:
        # You can write some log...
        raise e
    answer=getWordFromResult(result)
    server_timestamp = time.time()

    stealer.pause=False

    return jsonify({'esun_uuid': data['esun_uuid'],
                    'server_uuid': server_uuid,
                    'answer': answer,
                    'server_timestamp': server_timestamp})

if __name__ == "__main__":

    app.run(host="0.0.0.0",debug=args.debug, port=args.port)
