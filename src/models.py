import torch
# Importing our custom module(s)
import utils

class DenoisingDiffusionProbabilisticModel(torch.nn.Module):
    def __init__(self, eps_model, T=1000):
        super().__init__()
        self.eps_model = eps_model
        self.T = T
        # NOTE: Ho et al. (2020) used $\beta_1 = 1e-4$ and $\beta_T = 0.02$.
        self.register_buffer("raw_betas", utils.inv_softplus(torch.linspace(1e-4, 0.02, self.T)))
        # TODO: We should be able to learn betas with the ELBO.
        #self.raw_betas = torch.nn.Parameter(utils.inv_softplus(torch.linspace(1e-4, 0.02, self.T)))
        
    @property
    def betas(self):
        return torch.nn.functional.softplus(self.raw_betas)

    # TODO: @Jacob updated this forward process
    def forward(self, x_0):
        batch_size = len(x_0)
        t = torch.randint(0, self.T, (batch_size,), device=x_0.device)
        # Note: Calculating alphas and alphas_bar in forward pass so we can learn betas.
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alpha_bar_t = alphas_cumprod[t]
        eps = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_bar_t)[:, None, None, None] * x_0 + \
              torch.sqrt(1 - alpha_bar_t)[:, None, None, None] * eps
        logits = self.eps_model(x_t, t)
        return logits, eps
