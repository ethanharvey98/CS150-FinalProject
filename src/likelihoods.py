import torch

class GaussianLikelihood(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, labels, reduction):
        var = torch.ones_like(logits)
        return torch.nn.functional.gaussian_nll_loss(logits, labels, var, reduction=reduction, full=True)
