"""Adaptive prediction-set weighting used by AdaPS-COME."""
from __future__ import annotations

import torch


def prediction_set_size(probs: torch.Tensor, q: torch.Tensor | float) -> torch.Tensor:
    """Smallest top-k size whose cumulative probability reaches q."""
    sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
    cumsum = sorted_probs.cumsum(dim=1)
    if not torch.is_tensor(q):
        q = torch.tensor(float(q), dtype=probs.dtype, device=probs.device)
    q = q.to(device=probs.device, dtype=probs.dtype)
    while q.ndim < cumsum.ndim:
        q = q.unsqueeze(-1)
    reached = cumsum >= q
    # argmax gives the first True because False=0, True=1 and there is always True at the end.
    return reached.float().argmax(dim=1).long() + 1


def weights_from_set_size(set_size: torch.Tensor, mode: str = "inverse_sqrt_set_size") -> torch.Tensor:
    s = set_size.float().clamp_min(1.0)
    if mode == "inverse_sqrt_set_size":
        w = 1.0 / torch.sqrt(s)
    elif mode == "inverse_set_size":
        w = 1.0 / s
    else:
        raise ValueError(f"Unknown weight_mode={mode}")
    return w.clamp_min(1e-8)
