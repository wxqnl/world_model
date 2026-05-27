"""DDPM cosine noise schedule (eps-prediction) + DDIM sampler."""
from __future__ import annotations
import math
import torch


def _cosine_alphas(T: int) -> torch.Tensor:
    """Nichol-Dhariwal cosine schedule: alpha_bar(t) = cos^2((t/T + s)/(1+s) * pi/2), s=0.008."""
    s = 0.008
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T + s) / (1 + s)) * math.pi / 2.0) ** 2
    alphas_bar = (f / f[0]).clamp(min=1e-6, max=1.0)
    return alphas_bar  # length T+1, index 0 == 1.0


class CosineSchedule:
    def __init__(self, num_train_timesteps: int = 1000, device: str = "cuda"):
        self.T = num_train_timesteps
        alphas_bar = _cosine_alphas(num_train_timesteps)
        self.alphas_bar = alphas_bar[1:].to(device).float()             # alpha_bar at t = 1..T
        self.sqrt_alphas_bar = self.alphas_bar.sqrt()
        self.sqrt_one_minus_alphas_bar = (1.0 - self.alphas_bar).sqrt()

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # t: integer in [0, T-1]. shape [N]
        sa = self.sqrt_alphas_bar[t].view(-1, *([1] * (x0.dim() - 1)))
        som = self.sqrt_one_minus_alphas_bar[t].view(-1, *([1] * (x0.dim() - 1)))
        return sa * x0 + som * noise

    def sample_timesteps(self, n: int, device) -> torch.Tensor:
        return torch.randint(0, self.T, (n,), device=device, dtype=torch.long)

    @torch.no_grad()
    def ddim_sample(self, model_fn, shape, cond, n_steps: int = 25,
                    device: str = "cuda", dtype=torch.bfloat16,
                    shared_noise: bool = False) -> torch.Tensor:
        """DDIM sampling (eta=0). model_fn(x, t, cond) -> predicted eps.
        shape = [B, k, 4, 32, 32]; cond = [B, k, 64, cond_dim].
        If shared_noise: all k future frames start from the SAME noise tensor,
        which dramatically improves temporal coherence at the cost of some diversity."""
        step_idx = torch.linspace(self.T - 1, 0, n_steps + 1, device=device).long()
        if shared_noise:
            B, k = shape[0], shape[1]
            single = torch.randn(B, 1, *shape[2:], device=device, dtype=torch.float32)
            x = single.expand(B, k, *shape[2:]).contiguous()
        else:
            x = torch.randn(*shape, device=device, dtype=torch.float32)
        for i in range(n_steps):
            t = step_idx[i].expand(shape[0], shape[1])           # [B, k]
            t_next = step_idx[i + 1]
            a = self.alphas_bar[t]                                 # [B, k]
            a_next = self.alphas_bar[t_next]                       # scalar
            sa = a.sqrt().view(*t.shape, 1, 1, 1)
            som = (1 - a).sqrt().view(*t.shape, 1, 1, 1)
            with torch.autocast(device_type="cuda", dtype=dtype):
                eps = model_fn(x.to(dtype), t.to(device), cond.to(dtype)).float()
            x0 = (x - som * eps) / sa.clamp(min=1e-6)
            x0 = x0.clamp(-4.0, 4.0)                               # latent stability
            if i < n_steps - 1:
                a_next_b = a_next.view(1, 1, 1, 1, 1).expand_as(x)
                x = a_next_b.sqrt() * x0 + (1 - a_next_b).sqrt() * eps
            else:
                x = x0
        return x
