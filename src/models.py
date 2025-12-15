import copy
import torch
# Importing our custom module(s)
import utils

class DDPM(torch.nn.Module):
    def __init__(self, eps_model, T):
        super().__init__()
        self.eps_model = eps_model
        self.T = T
        # NOTE: Ho et al. (2020) used $\beta_1 = 1e-4$ and $\beta_T = 0.02$.
        self.register_buffer("raw_betas", utils.inv_softplus(torch.linspace(1e-4, 0.02, self.T)))
        #self.register_buffer("raw_betas", utils.inv_softplus(torch.tensor([0.00010001659393310547, 0.47586220502853394, 0.8500487804412842, 0.9573652148246765, 0.9879547357559204])))
        # TODO: We should be able to learn betas with the ELBO.
        #self.raw_betas = torch.nn.Parameter(utils.inv_softplus(torch.linspace(1e-4, 0.02, self.T)))
        
    @property
    def betas(self):
        return torch.nn.functional.softplus(self.raw_betas)
    
    def alpha_bar(self): 
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return alphas_cumprod

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

class UpscalingDDPM(torch.nn.Module):
    def __init__(self, eps_model, T):
        super().__init__()
        self.eps_model = eps_model
        self.T = T
        # NOTE: Ho et al. (2020) used $\beta_1 = 1e-4$ and $\beta_T = 0.02$.
        self.register_buffer("raw_betas", utils.inv_softplus(torch.linspace(1e-4, 0.02, self.T)))
        #self.register_buffer("raw_betas", utils.inv_softplus(torch.tensor([0.00010001659393310547, 0.47586220502853394, 0.8500487804412842, 0.9573652148246765, 0.9879547357559204])))
        # TODO: We should be able to learn betas with the ELBO.
        #self.raw_betas = torch.nn.Parameter(utils.inv_softplus(torch.linspace(1e-4, 0.02, self.T)))
        
    @property
    def betas(self):
        return torch.nn.functional.softplus(self.raw_betas)
    
    def alpha_bar(self): 
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return alphas_cumprod

    def forward(self, x_0):
        batch_size = len(x_0)
        t = torch.randint(0, self.T, (1,), device=x_0.device)
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alpha_bar_t = alphas_cumprod[t]        
        x_bar_0 = torch.nn.functional.avg_pool2d(x_0, kernel_size=2**t.item(), stride=2**t.item(), count_include_pad=False)
        down_x_bar_0 = torch.nn.functional.avg_pool2d(x_bar_0, kernel_size=2, stride=2, count_include_pad=False)
        eps = torch.randn_like(down_x_bar_0)
        x_t = torch.sqrt(alpha_bar_t)[:, None, None, None] * down_x_bar_0 + \
              torch.sqrt(1 - alpha_bar_t)[:, None, None, None] * eps
        up_x_t = torch.nn.functional.interpolate(x_t, scale_factor=2, mode="nearest")
        logits = self.eps_model(up_x_t, t)
        return logits, ((up_x_t - (torch.sqrt(alpha_bar_t)[:, None, None, None] * x_bar_0)) / torch.sqrt(1 - alpha_bar_t)[:, None, None, None]).detach()
    