import torch
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
import torchvision
import timm

class ResNet18(nn.Module):
    def __init__(self, in_features, num_classes, pretrained=False):
        super(ResNet18, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=pretrained)
        num_filters = self.model.conv1.out_channels
        self.model.conv1 = nn.Conv2d(in_channels=in_features, out_channels=num_filters, kernel_size=(7, 7), stride=(2, 2),
                                     padding=(3, 3), bias=False)
        num_filters = self.model.fc.in_features
        self.model.fc = nn.Linear(num_filters, out_features=num_classes)

    def forward(self, x):
        y = self.model(x)
        return y


class MobileNetv2(nn.Module):
    def __init__(self, in_features, num_classes):
        super(MobileNetv2, self).__init__()
        self.model = torchvision.models.MobileNetV2(num_classes=num_classes)
        num_filters = self.model.features[0][0].out_channels
        self.model.features[0][0] = nn.Conv2d(in_channels=in_features, out_channels=num_filters, kernel_size=(3, 3), stride=(2, 2),
                                     padding=(1, 1), bias=False)

    def forward(self, x):
        y = self.model(x)
        return y


class RegNetx(nn.Module):
    def __init__(self, in_features, num_classes, model='regnetx_002', pretrained=True):
        """
        RegNet Models
        'regnetx_002', 'regnetx_004', 'regnetx_006', 'regnetx_008', 'regnetx_016', 'regnetx_032',
        'regnetx_040', 'regnetx_064', 'regnetx_080', 'regnetx_120', 'regnetx_160', 'regnetx_320',
        'regnety_002', 'regnety_004', 'regnety_006', 'regnety_008', 'regnety_016', 'regnety_032',
        'regnety_040', 'regnety_064', 'regnety_080', 'regnety_120', 'regnety_160', 'regnety_320'
        """
        super(RegNetx, self).__init__()
        self.model = timm.create_model(model, pretrained=pretrained)
        num_filters = self.model.stem.conv.out_channels
        self.model.stem.conv = nn.Conv2d(in_channels=in_features, out_channels=num_filters, kernel_size=(3, 3), stride=(2, 2),
                                     padding=(1, 1), bias=False)
        num_filters = self.model.head.fc.in_features
        self.model.head.fc = nn.Linear(in_features=num_filters, out_features=num_classes, bias=True)

    def forward(self, x):
        y = self.model(x)
        return y


class Model(nn.Module):
    def __init__(self, in_features):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, padding=1)
        self.conv2 = nn.Conv2d(2, 1, 3, padding=1)
        self.maxpooling = nn.MaxPool2d(2)
        # self.dense1 = nn.Linear(in_features, 2048)
        # self.dense2 = nn.Linear(2048, 1024)
        self.dense3 = nn.Linear(1024, 801)
        self.softmax = nn.Softmax()
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpooling(x)
        x = self.ReLU(x)
        x = self.conv2(x)
        x = self.maxpooling(x)
        x = self.ReLU(x)
        x = torch.flatten(x, start_dim=1)
        # x = self.dense1(x)
        # x = self.ReLU(x)
        # x = self.dense2(x)
        # x = self.ReLU(x)
        x = self.dense3(x)
        y = self.ReLU(x)
        # y = self.softmax(x)
        return y


class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def k_one_hot(self, targets: torch.Tensor, n_classes: int, smoothing=0.0):
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes-1)) \
                .scatter_(1, torch.cuda.LongTensor(targets.data.unsqueeze(1).to(torch.long)), 1.-smoothing)
        return targets

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
            if self.reduction == 'sum' else loss

    def forward(self, inputs, targets):
        assert 0 <= self.smoothing < 1

        targets = self.k_one_hot(targets, inputs.size(-1), self.smoothing)
        log_preds = F.log_softmax(inputs, -1)

        if self.weight is not None:
            log_preds = log_preds * self.weight.unsqueeze(0)

        return self.reduce_loss(-(targets * log_preds).sum(dim=-1))
