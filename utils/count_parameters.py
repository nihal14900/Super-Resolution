import torch
from torch import nn

def print_parameters(model):

    model_name = model.__class__.__name__

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad == True)
    non_trainable = total - trainable

    print("====================================================================================================")
    print(f"Model: {model_name}")
    print(f"Total parameters:         {total         / 1e6 :.2f} M")
    print(f"Trainable parameters:     {trainable     / 1e6 :.2f} M")
    print(f"Non-trainable paramaters: {non_trainable / 1e6 :.2f} M")
    print("====================================================================================================")