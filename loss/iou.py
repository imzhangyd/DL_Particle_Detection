import torch
import torch.nn as nn


__author__ = "Yudong Zhang"


class cls_IOUloss(nn.Module):
    def __init__(self):
        super(cls_IOUloss,self).__init__()

    def forward(self, pred, label):
        b = pred.size()[0]
        pred = pred.view(b, -1)
        label = label.view(b, -1)
        inter = torch.sum(torch.mul(pred, label), dim=-1, keepdim=False)
        unit = torch.sum(torch.mul(pred, pred) + label, dim=-1, keepdim=False) - inter
        return torch.mean(1 - inter / (unit + 1e-10))

