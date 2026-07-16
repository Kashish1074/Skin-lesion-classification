import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


def build_model(num_classes, unfreeze_last_block=True):
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    # freeze entire backbone first
    for param in model.parameters():
        param.requires_grad = False

    # replace classifier head — this is always trainable
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    # optionally unfreeze the last inverted-residual block for fine-tuning
    if unfreeze_last_block:
        for param in model.features[-1].parameters():
            param.requires_grad = True

    return model


def get_optimizer(model, head_lr=1e-3, backbone_lr=1e-4):
    head_params = list(model.classifier.parameters())
    backbone_params = [p for p in model.features[-1].parameters() if p.requires_grad]

    param_groups = [
        {'params': head_params, 'lr': head_lr},
        {'params': backbone_params, 'lr': backbone_lr},
    ]

    optimizer = torch.optim.Adam(param_groups)
    return optimizer


def count_trainable_params(model):

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_device():
   
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")