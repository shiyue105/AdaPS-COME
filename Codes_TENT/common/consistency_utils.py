"""Light consistency diagnostics for UC/CUC ablations."""
from __future__ import annotations

import torch


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-sample Jensen-Shannon divergence between probability vectors."""
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    p = p / p.sum(dim=1, keepdim=True).clamp_min(eps)
    q = q / q.sum(dim=1, keepdim=True).clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p) - torch.log(m.clamp_min(eps)))).sum(dim=1)
    kl_qm = (q * (torch.log(q) - torch.log(m.clamp_min(eps)))).sum(dim=1)
    return 0.5 * (kl_pm + kl_qm)


def light_augment_batch(x: torch.Tensor, hflip_prob: float = 0.50, max_shift: int = 2) -> torch.Tensor:
    """Cheap semantic-preserving augmentation used only for consistency scoring.

    It avoids adding ImageNet-C-style noise/blur/fog, so the consistency branch does
    not duplicate the tested corruption.
    """
    y = x.clone()
    b = y.shape[0]
    device = y.device

    if hflip_prob > 0:
        mask = torch.rand(b, device=device) < hflip_prob
        if mask.any():
            y[mask] = torch.flip(y[mask], dims=[3])

    if max_shift and max_shift > 0:
        shift_y = int(torch.randint(-max_shift, max_shift + 1, (1,), device=device).item())
        shift_x = int(torch.randint(-max_shift, max_shift + 1, (1,), device=device).item())
        if shift_y != 0 or shift_x != 0:
            y = torch.roll(y, shifts=(shift_y, shift_x), dims=(2, 3))

    return y
