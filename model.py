import torch
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_features):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 3, padding=1)
        self.conv2 = nn.Conv2d(1, 1, 3, padding=1)
        self.maxpooling = nn.MaxPool2d(2)
        # self.dense1 = nn.Linear(in_features, 2048)
        # self.dense2 = nn.Linear(2048, 1024)
        self.dense3 = nn.Linear(1024, 801)
        self.softmax = nn.Softmax()
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpooling(x)
        x = self.conv2(x)
        x = self.maxpooling(x)
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
