"""Target-on-Normal auxiliary losses for excluding pasted anomalies from prototypes.

These losses deliberately operate on a separate aggregation-only graph.  The
Object-Erasing decoder branch can therefore keep its historical frozen-prototype
semantics while the auxiliary objectives update prototype/attention parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _token_side(token_count: int) -> int:
    side = int(token_count**0.5)
    if side * side != int(token_count):
        raise ValueError(f"Token count must be square, got {token_count}.")
    return side


def token_occupancy(target_mask: torch.Tensor, token_count: int) -> torch.Tensor:
    """Return soft target area fraction for every token."""

    if target_mask.ndim != 4 or target_mask.shape[1] != 1:
        raise ValueError("Target mask must have shape [B,1,H,W].")
    side = _token_side(token_count)
    return F.adaptive_avg_pool2d(target_mask.float(), (side, side)).flatten(1).clamp(0.0, 1.0)


def pair_clean_batch(clean: torch.Tensor, target_batch: int) -> torch.Tensor:
    """Match composite i to clean Normal i modulo the clean batch size."""

    if clean.ndim < 1 or clean.shape[0] <= 0:
        raise ValueError("Clean tensor must have a non-empty batch dimension.")
    if target_batch <= 0:
        raise ValueError("Target batch must be positive.")
    indices = torch.arange(target_batch, device=clean.device) % int(clean.shape[0])
    return clean.index_select(0, indices)


def nearest_prototype_distance(tokens: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Cosine distance from each token to its nearest Normal prototype."""

    if tokens.ndim != 3 or prototypes.ndim != 3:
        raise ValueError("Tokens/prototypes must have shape [B,N,C] and [B,K,C].")
    if tokens.shape[0] != prototypes.shape[0] or tokens.shape[2] != prototypes.shape[2]:
        raise ValueError("Token/prototype batch or channel dimensions disagree.")
    token_norm = F.normalize(tokens.float(), dim=-1)
    prototype_norm = F.normalize(prototypes.float(), dim=-1)
    distance = (1.0 - token_norm @ prototype_norm.transpose(1, 2)).clamp_min(0.0)
    return distance.min(dim=-1).values


def prototype_invariance_loss(
    composite_prototypes: torch.Tensor,
    clean_prototypes: torch.Tensor,
    *,
    prototype_to_mode: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Keep composite prototypes aligned with their paired clean prototypes."""

    paired = pair_clean_batch(clean_prototypes, int(composite_prototypes.shape[0])).detach()
    if paired.shape != composite_prototypes.shape:
        raise ValueError(
            f"Paired prototype shape mismatch: {tuple(paired.shape)} vs "
            f"{tuple(composite_prototypes.shape)}."
        )
    cosine = F.cosine_similarity(composite_prototypes.float(), paired.float(), dim=-1)
    drift = 1.0 - cosine
    loss = drift.mean()
    diagnostics = {
        "l_target_proto_invariance": float(loss.detach().cpu()),
        "target_proto_cosine": float(cosine.mean().detach().cpu()),
        "target_proto_drift": float(drift.mean().detach().cpu()),
        "target_proto_worst_drift": float(drift.max().detach().cpu()),
    }
    if prototype_to_mode is not None:
        slot_to_mode = prototype_to_mode.detach().long().flatten().to(drift.device)
        if slot_to_mode.shape != (int(drift.shape[1]),):
            raise ValueError(
                "Prototype-to-mode mapping must contain one entry per physical prototype."
            )
        unique_modes = torch.unique(slot_to_mode, sorted=True)
        expected_modes = torch.arange(unique_modes.numel(), device=unique_modes.device)
        if not torch.equal(unique_modes, expected_modes):
            raise ValueError("Prototype-to-mode mapping must use contiguous mode indices.")
        for mode_index in range(int(unique_modes.numel())):
            selected = drift[:, slot_to_mode == mode_index]
            diagnostics[f"target_proto_mode_{mode_index}_drift"] = float(
                selected.mean().detach().cpu()
            )
    return loss, diagnostics


def prototype_repulsion_loss(
    target_tokens: torch.Tensor,
    composite_prototypes: torch.Tensor,
    target_mask: torch.Tensor,
    clean_tokens: torch.Tensor,
    clean_prototypes: torch.Tensor,
    *,
    normal_quantile: float = 0.99,
    margin_delta: float = 0.02,
    minimum_occupancy: float = 0.01,
    prototype_to_mode: torch.Tensor | None = None,
    minimum_normal_tokens_per_mode: int = 8,
    gradient_side: str = "prototype",
    mode_budget: str = "none",
    clean_valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Hinge-repel target cores beyond a Normal distance tail.

    ``prototype_to_mode`` optionally collapses physical prototype slots onto
    effective Normal modes.  Each target token then uses the Normal tail of its
    assigned effective mode instead of a single global tail.  Under-supported
    modes conservatively fall back to the global tail.
    """

    if not 0.0 < float(normal_quantile) <= 1.0:
        raise ValueError("Normal distance quantile must lie in (0,1].")
    if margin_delta < 0.0 or not 0.0 <= minimum_occupancy <= 1.0:
        raise ValueError("Repulsion margin delta/occupancy are invalid.")
    if int(minimum_normal_tokens_per_mode) <= 0:
        raise ValueError("Minimum Normal tokens per mode must be positive.")
    if gradient_side not in {"prototype", "target"}:
        raise ValueError("Repulsion gradient side must be 'prototype' or 'target'.")
    if mode_budget not in {"none", "clean_target_ratio"}:
        raise ValueError("Repulsion mode budget must be 'none' or 'clean_target_ratio'.")
    if mode_budget != "none" and prototype_to_mode is None:
        raise ValueError("Repulsion mode budget requires a prototype-to-mode mapping.")
    if gradient_side == "prototype":
        target_tokens = target_tokens.detach()
    else:
        composite_prototypes = composite_prototypes.detach()
    occupancy = token_occupancy(target_mask, int(target_tokens.shape[1])).to(target_tokens)
    weights = occupancy.sqrt() * (occupancy >= float(minimum_occupancy)).to(occupancy)
    target_norm = F.normalize(target_tokens.float(), dim=-1)
    composite_norm = F.normalize(composite_prototypes.float(), dim=-1)
    target_distance_all = (1.0 - target_norm @ composite_norm.transpose(1, 2)).clamp_min(0.0)
    target_distance, target_slot = target_distance_all.min(dim=-1)
    paired_clean_prototypes = pair_clean_batch(clean_prototypes, int(clean_tokens.shape[0]))
    clean_norm = F.normalize(clean_tokens.float(), dim=-1)
    clean_prototype_norm = F.normalize(paired_clean_prototypes.float(), dim=-1)
    normal_distance_all = (1.0 - clean_norm @ clean_prototype_norm.transpose(1, 2)).clamp_min(0.0)
    normal_distance, normal_slot = normal_distance_all.min(dim=-1)
    if clean_valid_mask is not None:
        if clean_valid_mask.ndim != 4 or clean_valid_mask.shape[0] != clean_tokens.shape[0]:
            raise ValueError("Clean valid mask must be [B,1,H,W] and match clean tokens.")
        clean_side = _token_side(int(clean_tokens.shape[1]))
        clean_valid = (
            F.adaptive_avg_pool2d(
                clean_valid_mask.float(), (clean_side, clean_side)
            ).flatten(1)
            >= (1.0 - 1e-6)
        )
        if not bool(clean_valid.any()):
            raise ValueError("Prototype repulsion received an empty clean valid mask.")
    else:
        clean_valid = torch.ones_like(normal_distance, dtype=torch.bool)
    normal_tail = torch.quantile(
        normal_distance.detach().float()[clean_valid], float(normal_quantile)
    )
    mode_tail_min = normal_tail
    mode_tail_max = normal_tail
    fallback_modes = 0
    mode_diagnostics: dict[str, float] = {}
    token_budget = torch.ones_like(target_distance)
    if prototype_to_mode is None:
        margin = normal_tail + float(margin_delta)
        effective_mode_count = 0
    else:
        slot_to_mode = prototype_to_mode.detach().long().flatten().to(target_slot.device)
        prototype_count = int(composite_prototypes.shape[1])
        if slot_to_mode.shape != (prototype_count,):
            raise ValueError(
                "Prototype-to-mode mapping must contain one entry per physical prototype."
            )
        if bool((slot_to_mode < 0).any()):
            raise ValueError("Prototype-to-mode mapping contains a negative mode index.")
        unique_modes = torch.unique(slot_to_mode, sorted=True)
        expected_modes = torch.arange(unique_modes.numel(), device=unique_modes.device)
        if not torch.equal(unique_modes, expected_modes):
            raise ValueError("Prototype-to-mode mapping must use contiguous mode indices.")
        effective_mode_count = int(unique_modes.numel())
        normal_mode = slot_to_mode.index_select(0, normal_slot.detach().flatten()).view_as(normal_slot)
        target_mode = slot_to_mode.index_select(0, target_slot.detach().flatten()).view_as(target_slot)
        mode_tails = normal_tail.repeat(effective_mode_count)
        supported_tails = []
        for mode_index in range(effective_mode_count):
            selected = normal_distance.detach()[
                (normal_mode == mode_index) & clean_valid
            ].float()
            if int(selected.numel()) < int(minimum_normal_tokens_per_mode):
                fallback_modes += 1
                continue
            mode_tail = torch.quantile(selected, float(normal_quantile))
            mode_tails[mode_index] = mode_tail
            supported_tails.append(mode_tail)
        if supported_tails:
            stacked_tails = torch.stack(supported_tails)
            mode_tail_min = stacked_tails.min()
            mode_tail_max = stacked_tails.max()
        margin = mode_tails.index_select(0, target_mode.flatten()).view_as(target_distance)
        margin = margin + float(margin_delta)
        normal_counts = torch.stack(
            [(normal_mode == mode_index).float().sum() for mode_index in range(effective_mode_count)]
        )
        normal_shares = normal_counts / normal_counts.sum().clamp_min(1.0)
        target_masses = torch.stack(
            [
                weights[target_mode == mode_index].sum()
                for mode_index in range(effective_mode_count)
            ]
        )
        target_shares = target_masses / target_masses.sum().clamp_min(1e-6)
        suggested_mode_weights = torch.minimum(
            torch.ones_like(normal_shares),
            normal_shares / target_shares.clamp_min(1e-6),
        ).detach()
        applied_mode_weights = (
            suggested_mode_weights
            if mode_budget == "clean_target_ratio"
            else torch.ones_like(suggested_mode_weights)
        )
        token_budget = applied_mode_weights.index_select(
            0, target_mode.flatten()
        ).view_as(target_distance)
        for mode_index in range(effective_mode_count):
            selected_weights = weights[target_mode == mode_index]
            selected_hinge = F.relu(
                margin[target_mode == mode_index] - target_distance[target_mode == mode_index]
            )
            selected_denom = selected_weights.sum()
            if float(selected_denom.detach().cpu()) > 0.0:
                mode_loss = (selected_hinge * selected_weights).sum() / selected_denom
                mode_active = (
                    (selected_hinge > 0.0).to(selected_weights) * selected_weights
                ).sum() / selected_denom
            else:
                mode_loss = selected_denom
                mode_active = selected_denom
            prefix = f"target_proto_mode_{mode_index}"
            mode_diagnostics.update(
                {
                    f"{prefix}_normal_share": float(normal_shares[mode_index].detach().cpu()),
                    f"{prefix}_target_share": float(target_shares[mode_index].detach().cpu()),
                    f"{prefix}_suggested_budget_weight": float(
                        suggested_mode_weights[mode_index].cpu()
                    ),
                    f"{prefix}_applied_budget_weight": float(
                        applied_mode_weights[mode_index].cpu()
                    ),
                    f"{prefix}_repulsion_loss": float(mode_loss.detach().cpu()),
                    f"{prefix}_repulsion_active": float(mode_active.detach().cpu()),
                }
            )
    hinge = F.relu(margin - target_distance)
    per_image_denom = weights.sum(dim=1)
    valid = per_image_denom > 0.0
    if bool(valid.any()):
        per_image = (hinge * weights * token_budget).sum(dim=1) / per_image_denom.clamp_min(1e-6)
        loss = per_image[valid].mean()
        target_mean = ((target_distance * weights).sum(dim=1) / per_image_denom.clamp_min(1e-6))[valid].mean()
        active = (((hinge > 0.0).to(weights) * weights).sum(dim=1) / per_image_denom.clamp_min(1e-6))[valid].mean()
    else:
        loss = target_distance.sum() * 0.0
        target_mean = target_distance.detach().new_tensor(0.0)
        active = target_distance.detach().new_tensor(0.0)
    diagnostics = {
        "l_target_proto_repulsion": float(loss.detach().cpu()),
        "target_proto_repulsion_margin": float(margin.detach().float().mean().cpu()),
        "target_proto_repulsion_margin_min": float(mode_tail_min.detach().cpu()) + float(margin_delta),
        "target_proto_repulsion_margin_max": float(mode_tail_max.detach().cpu()) + float(margin_delta),
        "target_proto_effective_mode_count": float(effective_mode_count),
        "target_proto_margin_fallback_modes": float(fallback_modes),
        "target_proto_distance_mean": float(target_mean.detach().cpu()),
        "target_proto_repulsion_active": float(active.detach().cpu()),
        "normal_proto_distance_mean": float(normal_distance.mean().detach().cpu()),
        "normal_proto_distance_q95": float(torch.quantile(normal_distance.detach().float(), 0.95).cpu()),
        "normal_proto_distance_q99": float(normal_tail.cpu()),
        "target_proto_core_fraction": float((weights > 0.0).float().mean().detach().cpu()),
        "target_proto_occupancy_mean": float(occupancy.mean().detach().cpu()),
        "target_proto_gradient_side_target": float(gradient_side == "target"),
        "target_proto_mode_budget_enabled": float(mode_budget != "none"),
    }
    diagnostics.update(mode_diagnostics)
    return loss, diagnostics


def aggregation_attention_exclusion_loss(
    composite_attention: Sequence[torch.Tensor],
    clean_attention: Sequence[torch.Tensor],
    target_mask: torch.Tensor,
    *,
    target_to_background_ratio: float = 0.25,
    background_anchor_weight: float = 1.0,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Suppress target aggregation density and preserve clean background attention."""

    if not 0.0 <= target_to_background_ratio <= 1.0 or background_anchor_weight < 0.0:
        raise ValueError("Aggregation-attention ratio/anchor weight is invalid.")
    if not composite_attention or len(composite_attention) != len(clean_attention):
        raise ValueError("Clean/composite aggregation attention layers must be non-empty and aligned.")
    suppress_losses = []
    anchor_losses = []
    target_mass = []
    density_ratio = []
    for composite, clean in zip(composite_attention, clean_attention):
        if composite.ndim != 4 or clean.ndim != 4:
            raise ValueError("Aggregation attention must have shape [B,H,K,N].")
        paired_clean = pair_clean_batch(clean, int(composite.shape[0])).detach().to(composite)
        occupancy = token_occupancy(target_mask, int(composite.shape[-1])).to(composite)
        mask = occupancy[:, None, None, :]
        mass = (composite * mask).sum(dim=-1).mean()
        fraction = occupancy.mean().clamp_min(epsilon)
        background_mass = (composite * (1.0 - mask)).sum(dim=-1).mean()
        background_fraction = (1.0 - occupancy).mean().clamp_min(epsilon)
        target_density = mass / fraction
        background_density = background_mass / background_fraction
        suppress_losses.append(
            F.relu(target_density - float(target_to_background_ratio) * background_density)
        )
        background_mask = 1.0 - mask
        anchor_losses.append(
            F.smooth_l1_loss(
                composite * background_mask,
                paired_clean * background_mask,
                reduction="sum",
            )
            / background_mask.sum().clamp_min(epsilon)
        )
        target_mass.append(mass)
        density_ratio.append(target_density / background_density.clamp_min(epsilon))
    suppress = torch.stack(suppress_losses).mean()
    anchor = torch.stack(anchor_losses).mean()
    loss = suppress + float(background_anchor_weight) * anchor
    return loss, {
        "l_target_aggregation_attention": float(loss.detach().cpu()),
        "l_target_aggregation_suppress": float(suppress.detach().cpu()),
        "l_target_aggregation_background_anchor": float(anchor.detach().cpu()),
        "target_aggregation_attention_mass": float(torch.stack(target_mass).mean().detach().cpu()),
        "target_aggregation_attention_density_ratio": float(torch.stack(density_ratio).mean().detach().cpu()),
        "target_aggregation_attention_layers": float(len(suppress_losses)),
    }


def decoder_read_attention_exclusion_loss(
    composite_attention: Sequence[torch.Tensor],
    clean_attention: Sequence[torch.Tensor],
    target_mask: torch.Tensor,
    *,
    target_to_background_ratio: float = 0.25,
    background_anchor_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Suppress target reads while anchoring untouched background read attention."""

    if not 0.0 <= target_to_background_ratio <= 1.0 or background_anchor_weight < 0.0:
        raise ValueError("Read-attention ratio/anchor weight is invalid.")
    if not composite_attention or len(composite_attention) != len(clean_attention):
        raise ValueError("Clean/composite decoder attention layers must be non-empty and aligned.")
    suppress_losses = []
    anchor_losses = []
    target_means = []
    background_means = []
    for composite, clean in zip(composite_attention, clean_attention):
        if composite.ndim != 4 or clean.ndim != 4:
            raise ValueError("Decoder attention must have shape [B,H,N,K].")
        paired_clean = pair_clean_batch(clean, int(composite.shape[0])).detach().to(composite)
        occupancy = token_occupancy(target_mask, int(composite.shape[2])).to(composite)
        target_weight = occupancy[:, None, :, None]
        background_weight = 1.0 - target_weight
        target_mean = (composite * target_weight).sum() / (
            target_weight.sum() * composite.shape[1] * composite.shape[3]
        ).clamp_min(1e-6)
        background_mean = (composite * background_weight).sum() / (
            background_weight.sum() * composite.shape[1] * composite.shape[3]
        ).clamp_min(1e-6)
        suppress_losses.append(F.relu(target_mean - float(target_to_background_ratio) * background_mean))
        anchor_losses.append(
            F.smooth_l1_loss(
                composite * background_weight,
                paired_clean * background_weight,
                reduction="sum",
            )
            / (background_weight.sum() * composite.shape[1] * composite.shape[3]).clamp_min(1e-6)
        )
        target_means.append(target_mean)
        background_means.append(background_mean)
    suppress = torch.stack(suppress_losses).mean()
    anchor = torch.stack(anchor_losses).mean()
    target_mean = torch.stack(target_means).mean()
    background_mean = torch.stack(background_means).mean()
    loss = suppress + float(background_anchor_weight) * anchor
    return loss, {
        "l_target_read_attention": float(loss.detach().cpu()),
        "l_target_read_suppress": float(suppress.detach().cpu()),
        "l_target_read_background_anchor": float(anchor.detach().cpu()),
        "target_read_attention_mean": float(target_mean.detach().cpu()),
        "background_read_attention_mean": float(background_mean.detach().cpu()),
        "target_read_attention_ratio": float((target_mean / background_mean.clamp_min(1e-6)).detach().cpu()),
        "target_read_attention_layers": float(len(suppress_losses)),
    }


@dataclass
class DecoderAttentionCapture:
    """Capture differentiable INP-Former decoder attention outputs."""

    blocks: Iterable[nn.Module]

    def __post_init__(self) -> None:
        self.outputs: list[torch.Tensor] = []
        self.inputs: list[tuple[nn.Module, torch.Tensor, torch.Tensor]] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, module: nn.Module, inputs: tuple, output: object) -> None:
        if isinstance(output, tuple) and len(output) >= 2 and isinstance(output[1], torch.Tensor):
            self.outputs.append(output[1])
        if len(inputs) >= 2 and isinstance(inputs[0], torch.Tensor) and isinstance(inputs[1], torch.Tensor):
            self.inputs.append((module, inputs[0].detach(), inputs[1].detach()))

    def start(self) -> None:
        if self._handles:
            raise RuntimeError("Decoder attention capture is already active.")
        for block in self.blocks:
            attention = getattr(block, "attn", None)
            if not isinstance(attention, nn.Module):
                raise ValueError("Decoder block does not expose an attention module.")
            self._handles.append(attention.register_forward_hook(self._hook))

    def clear(self) -> None:
        self.outputs.clear()
        self.inputs.clear()

    def recompute_last(self, layer_count: int) -> tuple[torch.Tensor, ...]:
        """Recompute attention with detached inputs so gradients stay in decoder attention."""

        if layer_count <= 0 or len(self.inputs) < layer_count:
            raise RuntimeError("Captured decoder inputs do not cover all requested layers.")
        recomputed = []
        for attention, tokens, prototypes in self.inputs[-layer_count:]:
            batch, token_count, channels = tokens.shape
            prototype_count = prototypes.shape[1]
            heads = int(getattr(attention, "num_heads"))
            query = attention.q(tokens).reshape(
                batch, token_count, heads, channels // heads
            ).permute(0, 2, 1, 3)
            key_value = attention.kv(prototypes).reshape(
                batch, prototype_count, 2, heads, channels // heads
            ).permute(2, 0, 3, 1, 4)
            key = key_value[0]
            query = F.normalize(query, dim=-1)
            key = F.normalize(key, dim=-1)
            recomputed.append(
                F.relu((query @ key.transpose(-2, -1)) * attention.learn_scale)
            )
        return tuple(recomputed)

    def stop(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
