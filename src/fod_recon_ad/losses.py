from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F


def cosine_residual(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(target.detach(), pred, dim=1).unsqueeze(1)).clamp_min(0.0)


def hard_weighted_cosine_loss(
    en: Iterable[torch.Tensor],
    de: Iterable[torch.Tensor],
    hard_quantile: float = 0.9,
    easy_weight: float = 0.1,
) -> torch.Tensor:
    """Normal-only feature reconstruction loss.

    The original Dinomaly code uses gradient hooks for hard mining. This version
    keeps the same intent with explicit residual weights, which makes it easier
    to reuse outside the original industrial AD training scripts.
    """

    losses: List[torch.Tensor] = []
    for target, pred in zip(en, de):
        residual = cosine_residual(target, pred)
        if 0.0 < hard_quantile < 1.0:
            flat = residual.detach().flatten(1)
            thresh = torch.quantile(flat, hard_quantile, dim=1).view(-1, 1, 1, 1)
            weights = torch.where(residual.detach() >= thresh, torch.ones_like(residual), easy_weight * torch.ones_like(residual))
            losses.append((weights * residual).mean())
        else:
            losses.append(residual.mean())
    return torch.stack(losses).mean()


def hard_weighted_masked_cosine_loss(
    en: Iterable[torch.Tensor],
    de: Iterable[torch.Tensor],
    mask: torch.Tensor,
    hard_quantile: float = 0.9,
    easy_weight: float = 0.1,
    band_weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Feature reconstruction loss restricted to structurally masked tokens."""

    losses: List[torch.Tensor] = []
    mask = mask.to(dtype=torch.bool)
    for target, pred in zip(en, de):
        residual = cosine_residual(target, pred)
        layer_mask = mask
        if layer_mask.shape[-2:] != residual.shape[-2:]:
            layer_mask = F.interpolate(layer_mask.float(), size=residual.shape[-2:], mode="nearest") > 0.5
        valid = layer_mask.expand_as(residual)
        selected = residual[valid]
        if selected.numel() == 0:
            losses.append(residual.mean() * 0.0)
            continue
        row_weight = None
        if band_weights:
            values = torch.as_tensor(band_weights, dtype=residual.dtype, device=residual.device)
            if values.numel() == 0:
                raise ValueError("band_weights must not be empty.")
            height = residual.shape[-2]
            row_ids = torch.div(
                torch.arange(height, device=residual.device) * values.numel(),
                height,
                rounding_mode="floor",
            ).clamp_max(values.numel() - 1)
            row_weight = values[row_ids].view(1, 1, height, 1).expand_as(residual)
            row_weight = row_weight / row_weight[valid].mean().clamp_min(1e-6)
        if 0.0 < hard_quantile < 1.0:
            flat = residual.detach().flatten(1)
            mask_flat = valid.flatten(1)
            weights = torch.full_like(residual, easy_weight)
            for idx in range(residual.shape[0]):
                values = flat[idx][mask_flat[idx]]
                if values.numel() == 0:
                    continue
                thresh = torch.quantile(values, hard_quantile)
                hard = residual[idx].detach() >= thresh
                weights[idx] = torch.where(hard, torch.ones_like(weights[idx]), weights[idx])
            if row_weight is not None:
                weights = weights * row_weight
            losses.append((weights[valid] * selected).mean())
        else:
            if row_weight is not None:
                losses.append((row_weight[valid] * selected).mean())
            else:
                losses.append(selected.mean())
    return torch.stack(losses).mean()


def distance_weighted_residual_loss(
    components: dict[str, torch.Tensor],
    risk: torch.Tensor,
    beta: float = 1.5,
) -> torch.Tensor:
    """Optional training-side loss. Default scripts keep this disabled.

    Previous FOD experiments showed that aggressive high-risk weighting can make
    the decoder reconstruct small abnormal-looking details too well. Use this
    only for controlled ablations.
    """

    base = components["base"]
    return ((1.0 + beta * risk.detach()) * base).mean()
