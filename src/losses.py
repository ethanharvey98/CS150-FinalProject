import torch

class SimpleLoss(torch.nn.Module):
    def __init__(self, model, likelihood):
        super().__init__()
        self.model = model
        self.likelihood = likelihood

    def forward(self, logits, labels, params, n):
        nll = self.likelihood(logits, labels, reduction="mean")
        return {"loss": nll, "nll": nll}
    
class WeightedLoss(torch.nn.Module):
    def __init__(self, model, likelihood):
        super().__init__()
        self.model = model
        self.likelihood = likelihood

    def forward(self, logits, labels, params, n):
        nll = labels.shape[2] * labels.shape[3] * self.likelihood(logits, labels, reduction="mean")
        return {"loss": nll, "nll": nll}
    