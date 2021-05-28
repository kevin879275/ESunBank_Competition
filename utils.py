import numpy as np
import torch
import random
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





def prograssiveNow(epoch,num_epochs,boudDictofModel):
    pa = epoch/(num_epochs -1)
    args = ["mix","drop","randarg","imgsize"]
    ansDict={}
    for arg in args:
        ansDict[arg]=boudDictofModel[f"{arg}m"]+(boudDictofModel[f"{arg}M"]-boudDictofModel[f"{arg}m"])*pa
    return ansDict


def setDropout(model, drop_rate=0.1):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Dropout):
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
    mixed_y = p * y + q * y[index, :] 
    return mixed_x, mixed_y

