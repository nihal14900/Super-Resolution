import torch
from torch import nn

def create_loss():
    return nn.L1Loss(reduction="mean")