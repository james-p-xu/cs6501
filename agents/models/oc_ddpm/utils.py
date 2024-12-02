
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mail_cfg


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RotaryPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "Embedding dimension must be even."
        self.dim = dim

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        batch_size, seq_len = x.shape

        t = torch.arange(seq_len, device=device, dtype=x.dtype).unsqueeze(1)  # Shape: [seq_len, 1]
        freqs = t * self.inv_freq  # Shape: [seq_len, dim/2]
        sinusoid = torch.cat((freqs.sin(), freqs.cos()), dim=-1)  # Shape: [seq_len, dim]

        sinusoid = sinusoid.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: [batch_size, seq_len, dim]

        x_emb = x.unsqueeze(-1).expand(-1, -1, self.dim)  # Shape: [batch_size, seq_len, dim]

        x1, x2 = torch.chunk(x_emb, 2, dim=-1)
        sin_part, cos_part = torch.chunk(sinusoid, 2, dim=-1)
        x_rotated = torch.cat((x1 * cos_part - x2 * sin_part, x1 * sin_part + x2 * cos_part), dim=-1)

        return x_rotated

def get_positional_embedding(dim: int):
    if mail_cfg.USE_ROPE:
        return RotaryPosEmb(dim)
    return SinusoidalPosEmb(dim)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2, dtype=torch.float32):
    betas = np.linspace(
        beta_start, beta_end, timesteps
    )
    return torch.tensor(betas, dtype=dtype)


def vp_beta_schedule(timesteps, dtype=torch.float32):
    t = np.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.
    b_min = 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return torch.tensor(betas, dtype=dtype)


class WeightedLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, targ, weights=1.0):
        '''
            pred, targ : tensor [ batch_size x action_dim ]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = (loss * weights).mean()
        return weighted_loss


class WeightedL1(WeightedLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class WeightedL2(WeightedLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')


Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
}