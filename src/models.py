import torch
# Importing our custom module(s)
import utils

class DenoisingDiffusionProbabilisticModel(torch.nn.Module):
    def __init__(self, eps_model, T):
        super().__init__()
        self.eps_model = eps_model
        self.T = T
        # NOTE: Ho et al. (2020) used $\beta_1 = 1e-4$ and $\beta_T = 0.02$.
        self.register_buffer("raw_betas", utils.inv_softplus(torch.linspace(1e-4, 0.02, self.T)))
        #self.register_buffer("raw_betas", utils.inv_softplus(torch.linspace(1e-3, 0.2, self.T)))
        # TODO: We should be able to learn betas with the ELBO.
        #self.raw_betas = torch.nn.Parameter(utils.inv_softplus(torch.linspace(1e-4, 0.02, self.T)))
        
    @property
    def betas(self):
        return torch.nn.functional.softplus(self.raw_betas)

    def forward(self, x_0):
        batch_size = len(x_0)
        t = torch.randint(0, self.T, (1,), device=x_0.device, requires_grad=False)
        # Note: Calculating alphas and alphas_bar in forward pass so we can learn betas.
        # TODO: How does variance change?
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alpha_bar_t = alphas_cumprod[t]
        x_t_minus_1 = torch.nn.functional.avg_pool2d(x_0, kernel_size=2**t.item(), stride=2**t.item(), count_include_pad=False)
        eps = torch.randn_like(x_t_minus_1)
        x_t_minus_1 = torch.sqrt(alpha_bar_t)[:, None, None, None] * x_t_minus_1 + \
                      torch.sqrt(1 - alpha_bar_t)[:, None, None, None] * eps
        x_t = torch.nn.functional.avg_pool2d(x_t_minus_1, kernel_size=2, stride=2, count_include_pad=False)
        logits = self.eps_model(x_t, t)
        return logits, x_t_minus_1.detach()
