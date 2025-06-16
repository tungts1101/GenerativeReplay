from torch import nn
from timm.models.layers.weight_init import trunc_normal_


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def weight_init(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def count_parameters(model, trainble=False):
    return (
        sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainble
        else sum(p.numel() for p in model.parameters())
    )
