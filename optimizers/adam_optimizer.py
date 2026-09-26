import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LRScheduler, StepLR

def create_optimizer(model, learning_rate):
    return Adam(model.parameters(), learning_rate)

def create_scheduler(optimizer, step_size, gamma):
    return StepLR(optimizer, step_size, gamma)