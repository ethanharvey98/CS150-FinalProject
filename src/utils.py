import torch

def inv_softplus(x):
    return x + torch.log(-torch.expm1(-x))
