from __future__ import annotations

from functools import partial
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from .losses import hard_weighted_cosine_loss, hard_weighted_masked_cosine_loss


def _modify_grad(x: torch.Tensor, inds: torch.Tensor, factor: float = 0.0) -> torch.Tensor:
    inds = inds.expand_as(x)
    x[inds] *= factor
    return x


def _modify_grad_v2(x: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    factor = factor.expand_as(x)
    x *= factor
    return x


def plain_global_cosine_loss(en: Iterable[torch.Tensor], de: Iterable[torch.Tensor]) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    for target, pred in zip(en, de):
        losses.append((1.0 - F.cosine_similarity(target.detach().flatten(1), pred.flatten(1), dim=1, eps=1e-6)).mean())
    return torch.stack(losses).mean()


def dinomaly_global_cosine_hm_percent(
    en: Iterable[torch.Tensor],
    de: Iterable[torch.Tensor],
    p: float = 0.9,
    factor: float = 0.1,
) -> torch.Tensor:
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0.0
    count = 0
    for target, pred in zip(en, de):
        target_detached = target.detach()
        with torch.no_grad():
            point_dist = 1.0 - cos_loss(target_detached, pred).unsqueeze(1)
            keep = int(point_dist.numel() * (1.0 - p))
            keep = max(1, min(point_dist.numel(), keep))
            thresh = torch.topk(point_dist.reshape(-1), k=keep)[0][-1]
        loss = loss + torch.mean(
            1.0 - cos_loss(target_detached.reshape(target_detached.shape[0], -1), pred.reshape(pred.shape[0], -1))
        )
        pred.register_hook(partial(_modify_grad, inds=point_dist < thresh, factor=factor))
        count += 1
    if count == 0:
        raise ValueError("No feature pairs were provided for Dinomaly native loss.")
    return loss / count


def inpformer_global_cosine_hm_adaptive(
    en: Iterable[torch.Tensor],
    de: Iterable[torch.Tensor],
    y: float = 3.0,
) -> torch.Tensor:
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0.0
    count = 0
    for target, pred in zip(en, de):
        target_detached = target.detach()
        with torch.no_grad():
            point_dist = 1.0 - cos_loss(target_detached, pred).unsqueeze(1).detach()
            factor = (point_dist / point_dist.mean()).pow(y)
        loss = loss + torch.mean(
            1.0 - cos_loss(target_detached.reshape(target_detached.shape[0], -1), pred.reshape(pred.shape[0], -1))
        )
        pred.register_hook(partial(_modify_grad_v2, factor=factor))
        count += 1
    if count == 0:
        raise ValueError("No feature pairs were provided for INP-Former native loss.")
    return loss / count


class _GradientScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(factor)
        return inputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (factor,) = ctx.saved_tensors
        return grad_output * factor, None


def inpformer_soft_mining_loss(
    en: Iterable[torch.Tensor],
    de: Iterable[torch.Tensor],
    gamma: float = 3.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """INP-Former++ Eq. (5): independently mined cosine and MSE gradients."""

    cosine_losses: List[torch.Tensor] = []
    mse_losses: List[torch.Tensor] = []
    for target, pred in zip(en, de):
        target = target.detach()
        with torch.no_grad():
            cosine_map = (1.0 - F.cosine_similarity(target, pred, dim=1, eps=eps)).unsqueeze(1).clamp_min(0.0)
            mse_map = (target - pred).square().mean(dim=1, keepdim=True)
            cosine_factor = (cosine_map / cosine_map.mean().clamp_min(eps)).pow(gamma)
            mse_factor = (mse_map / mse_map.mean().clamp_min(eps)).pow(gamma)
        cosine_pred = _GradientScale.apply(pred, cosine_factor)
        mse_pred = _GradientScale.apply(pred, mse_factor)
        cosine_losses.append(
            (1.0 - F.cosine_similarity(target.flatten(1), cosine_pred.flatten(1), dim=1, eps=eps)).mean()
        )
        mse_losses.append(F.mse_loss(mse_pred, target))
    if not cosine_losses:
        raise ValueError("No feature pairs were provided for INP-Former++ soft-mining loss.")
    cosine_loss = torch.stack(cosine_losses).mean()
    mse_loss = torch.stack(mse_losses).mean()
    return cosine_loss + mse_loss, cosine_loss, mse_loss


def native_dinomaly_p(step: int, p_final: float = 0.9, warmup_iters: int = 1000) -> float:
    if warmup_iters <= 0:
        return float(p_final)
    return float(min(p_final * max(step, 1) / warmup_iters, p_final))


def _resize_mask(mask: torch.Tensor, shape: tuple[int, int], dtype: torch.dtype) -> torch.Tensor:
    layer_mask = mask.to(dtype=torch.bool)
    if layer_mask.shape[-2:] != shape:
        layer_mask = F.interpolate(layer_mask.float(), size=shape, mode="nearest") > 0.5
    return layer_mask.to(dtype=dtype)


def _masked_residual_mean(residual: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = _resize_mask(mask, residual.shape[-2:], residual.dtype)
    return (residual * weight).sum() / weight.sum().clamp_min(1.0)


def dinomaly_masked_cosine_hm_percent(
    en: Iterable[torch.Tensor],
    de: Iterable[torch.Tensor],
    mask: torch.Tensor,
    p: float = 0.9,
    factor: float = 0.1,
) -> torch.Tensor:
    cos_loss = torch.nn.CosineSimilarity()
    losses: List[torch.Tensor] = []
    for target, pred in zip(en, de):
        target_detached = target.detach()
        residual = (1.0 - cos_loss(target_detached, pred).unsqueeze(1)).clamp_min(0.0)
        weight = _resize_mask(mask, residual.shape[-2:], residual.dtype)
        with torch.no_grad():
            grad_factor = torch.zeros_like(residual)
            for idx in range(residual.shape[0]):
                values = residual[idx][weight[idx].bool()]
                if values.numel() == 0:
                    continue
                keep = max(1, min(values.numel(), int(values.numel() * (1.0 - p))))
                thresh = torch.topk(values.reshape(-1), k=keep).values[-1]
                easy = residual[idx] < thresh
                grad_factor[idx] = torch.where(easy, factor * weight[idx], weight[idx])
        pred.register_hook(partial(_modify_grad_v2, factor=grad_factor))
        losses.append(_masked_residual_mean(residual, mask))
    if not losses:
        raise ValueError("No feature pairs were provided for Dinomaly masked native loss.")
    return torch.stack(losses).mean()


def inpformer_masked_cosine_hm_adaptive(
    en: Iterable[torch.Tensor],
    de: Iterable[torch.Tensor],
    mask: torch.Tensor,
    y: float = 3.0,
) -> torch.Tensor:
    cos_loss = torch.nn.CosineSimilarity()
    losses: List[torch.Tensor] = []
    for target, pred in zip(en, de):
        target_detached = target.detach()
        residual = (1.0 - cos_loss(target_detached, pred).unsqueeze(1)).clamp_min(0.0)
        weight = _resize_mask(mask, residual.shape[-2:], residual.dtype)
        with torch.no_grad():
            denom = weight.flatten(1).sum(dim=1).view(-1, 1, 1, 1).clamp_min(1.0)
            mean = (residual.detach() * weight).flatten(1).sum(dim=1).view(-1, 1, 1, 1) / denom
            grad_factor = (residual.detach() / mean.clamp_min(1e-6)).pow(y) * weight
        pred.register_hook(partial(_modify_grad_v2, factor=grad_factor))
        losses.append(_masked_residual_mean(residual, mask))
    if not losses:
        raise ValueError("No feature pairs were provided for INP-Former masked native loss.")
    return torch.stack(losses).mean()


def compute_normal_loss(
    output: Tuple[torch.Tensor, ...] | tuple | list,
    architecture: str,
    step: int,
    normal_loss: str,
    prototype_loss_weight: float = 0.2,
    hard_quantile: float = 0.9,
    easy_weight: float = 0.1,
    dinomaly_p_final: float = 0.9,
    dinomaly_warmup_iters: int = 1000,
    dinomaly_factor: float = 0.1,
    inpformer_y: float = 3.0,
    inpformer_soft_mining_gamma: float = 3.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    en, de = output[:2]
    diagnostics: dict[str, float] = {}

    if normal_loss == "hard_weighted":
        loss = hard_weighted_cosine_loss(en, de, hard_quantile=hard_quantile, easy_weight=easy_weight)
    elif normal_loss == "legacy_plain":
        loss = plain_global_cosine_loss(en, de)
    elif normal_loss == "native":
        if architecture == "dinomaly":
            p = native_dinomaly_p(step, p_final=dinomaly_p_final, warmup_iters=dinomaly_warmup_iters)
            loss = dinomaly_global_cosine_hm_percent(en, de, p=p, factor=dinomaly_factor)
            diagnostics["dinomaly_native_p"] = p
        elif architecture == "inpformer":
            loss = inpformer_global_cosine_hm_adaptive(en, de, y=inpformer_y)
            diagnostics["inpformer_native_y"] = float(inpformer_y)
        else:
            raise ValueError(f"--normal-loss native is not defined for architecture={architecture!r}.")
    elif normal_loss == "inp_soft_mining":
        if architecture != "inpformer":
            raise ValueError("--normal-loss inp_soft_mining is only defined for INP-Former.")
        loss, cosine_loss, mse_loss = inpformer_soft_mining_loss(
            en,
            de,
            gamma=inpformer_soft_mining_gamma,
        )
        diagnostics["inp_soft_mining_cosine"] = float(cosine_loss.detach().cpu())
        diagnostics["inp_soft_mining_mse"] = float(mse_loss.detach().cpu())
        diagnostics["inp_soft_mining_gamma"] = float(inpformer_soft_mining_gamma)
    else:
        raise ValueError(f"Unsupported normal_loss={normal_loss!r}.")

    if architecture == "inpformer" and len(output) > 2:
        gather_loss = output[2]
        loss = loss + prototype_loss_weight * gather_loss
        diagnostics["gather_loss"] = float(gather_loss.detach().cpu())
        diagnostics["prototype_loss_weight"] = float(prototype_loss_weight)

    return loss, diagnostics


def compute_masked_normal_loss(
    en: Iterable[torch.Tensor],
    de: Iterable[torch.Tensor],
    mask: torch.Tensor,
    architecture: str,
    step: int,
    normal_loss: str,
    prototype_loss_weight: float = 0.2,
    gather_loss: torch.Tensor | None = None,
    hard_quantile: float = 0.9,
    easy_weight: float = 0.1,
    band_weights: Sequence[float] | None = None,
    dinomaly_p_final: float = 0.9,
    dinomaly_warmup_iters: int = 1000,
    dinomaly_factor: float = 0.1,
    inpformer_y: float = 3.0,
    inpformer_soft_mining_gamma: float = 3.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    diagnostics: dict[str, float] = {}

    if normal_loss == "hard_weighted":
        loss = hard_weighted_masked_cosine_loss(
            en,
            de,
            mask=mask,
            hard_quantile=hard_quantile,
            easy_weight=easy_weight,
            band_weights=band_weights,
        )
    elif normal_loss == "legacy_plain":
        losses: List[torch.Tensor] = []
        for target, pred in zip(en, de):
            residual = (1.0 - F.cosine_similarity(target.detach(), pred, dim=1).unsqueeze(1)).clamp_min(0.0)
            losses.append(_masked_residual_mean(residual, mask))
        loss = torch.stack(losses).mean()
    elif normal_loss == "native":
        if architecture == "dinomaly":
            p = native_dinomaly_p(step, p_final=dinomaly_p_final, warmup_iters=dinomaly_warmup_iters)
            loss = dinomaly_masked_cosine_hm_percent(en, de, mask=mask, p=p, factor=dinomaly_factor)
            diagnostics["dinomaly_native_p"] = p
        elif architecture == "inpformer":
            loss = inpformer_masked_cosine_hm_adaptive(en, de, mask=mask, y=inpformer_y)
            diagnostics["inpformer_native_y"] = float(inpformer_y)
        else:
            raise ValueError(f"--normal-loss native is not defined for architecture={architecture!r}.")
    elif normal_loss == "inp_soft_mining":
        raise ValueError("INP-Former++ soft mining is currently defined for full-token normal training, not masked loss.")
    else:
        raise ValueError(f"Unsupported normal_loss={normal_loss!r}.")

    if architecture == "inpformer" and gather_loss is not None:
        loss = loss + prototype_loss_weight * gather_loss
        diagnostics["gather_loss"] = float(gather_loss.detach().cpu())
        diagnostics["prototype_loss_weight"] = float(prototype_loss_weight)

    return loss, diagnostics


def compute_roi_masked_normal_loss(
    output: Tuple[torch.Tensor, ...] | tuple | list,
    roi_mask: torch.Tensor,
    architecture: str,
    step: int,
    normal_loss: str,
    prototype_loss_weight: float = 0.2,
    gather_distance: torch.Tensor | None = None,
    roi_aware_gather_loss: torch.Tensor | None = None,
    hard_quantile: float = 0.9,
    easy_weight: float = 0.1,
    dinomaly_p_final: float = 0.9,
    dinomaly_warmup_iters: int = 1000,
    dinomaly_factor: float = 0.1,
    inpformer_y: float = 3.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Normal reconstruction loss restricted to a spatial Clean-ROI mask.

    Unlike structural masked reconstruction, this mask denotes valid training
    support. By default, INP-Former reconstructs the prototype term from
    per-token nearest-prototype distance, preserving the historical Raw path.
    Explicit guided-loss ablations may instead supply a scalar that was already
    computed over the same strict ROI tokens.
    """

    en, de = output[:2]
    gather_loss = None
    if architecture == "inpformer" and roi_aware_gather_loss is not None:
        if roi_aware_gather_loss.ndim != 0:
            raise ValueError("ROI-aware model gather loss must be scalar.")
        gather_loss = roi_aware_gather_loss
        diagnostics_guided = True
    else:
        diagnostics_guided = False
    if gather_loss is None and architecture == "inpformer" and gather_distance is not None:
        if gather_distance.ndim != 2:
            raise ValueError("INP-Former gather_distance must have shape [B,N].")
        token_count = int(gather_distance.shape[1])
        side = int(token_count**0.5)
        if side * side != token_count:
            raise ValueError(f"Gather token count must be square, got {token_count}.")
        token_mask = F.adaptive_avg_pool2d(
            roi_mask.float(), (side, side)
        ).flatten(1)
        # Exclude any token cell intersecting the invalid region.
        token_mask = (token_mask >= 1.0 - 1e-6).to(gather_distance.dtype)
        gather_loss = (
            gather_distance * token_mask
        ).sum() / token_mask.sum().clamp_min(1.0)
    elif (
        gather_loss is None
        and architecture == "inpformer"
        and prototype_loss_weight > 0.0
    ):
        raise ValueError(
            "ROI-masked INP-Former training requires per-token gather_distance."
        )

    loss, diagnostics = compute_masked_normal_loss(
        en,
        de,
        mask=roi_mask,
        architecture=architecture,
        step=step,
        normal_loss=normal_loss,
        prototype_loss_weight=prototype_loss_weight,
        gather_loss=gather_loss,
        hard_quantile=hard_quantile,
        easy_weight=easy_weight,
        dinomaly_p_final=dinomaly_p_final,
        dinomaly_warmup_iters=dinomaly_warmup_iters,
        dinomaly_factor=dinomaly_factor,
        inpformer_y=inpformer_y,
    )
    diagnostics["roi_mask_fraction"] = float(roi_mask.detach().float().mean().cpu())
    if gather_loss is not None:
        diagnostics["roi_masked_gather_loss"] = float(gather_loss.detach().cpu())
        diagnostics["roi_aware_guided_gather"] = float(diagnostics_guided)
    return loss, diagnostics
