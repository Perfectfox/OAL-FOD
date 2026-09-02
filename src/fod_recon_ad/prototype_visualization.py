from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


GROUP_NAMES = ("background", "texture", "object_like")


@dataclass(frozen=True)
class PrototypeAssignments:
    prototype_similarity: torch.Tensor
    prototype_index: torch.Tensor
    prototype_margin: torch.Tensor
    group_similarity: torch.Tensor
    group_index: torch.Tensor
    group_margin: torch.Tensor
    prototype_matrix: torch.Tensor


@dataclass(frozen=True)
class FamiliarityGateMaps:
    """Token-wise components of the familiarity aggregation gate."""

    objectness: torch.Tensor
    novelty: torch.Tensor
    risk: torch.Tensor
    guided_weight: torch.Tensor
    token_weight: torch.Tensor
    suppression: torch.Tensor


def group_slices(groups: tuple[int, int, int]) -> tuple[slice, slice, slice]:
    if len(groups) != 3 or any(int(value) <= 0 for value in groups):
        raise ValueError(f"Expected three positive prototype group sizes, got {groups}.")
    bg, texture, object_like = (int(value) for value in groups)
    return (
        slice(0, bg),
        slice(bg, bg + texture),
        slice(bg + texture, bg + texture + object_like),
    )


def prototype_assignments(
    tokens: torch.Tensor,
    prototypes: torch.Tensor,
    groups: tuple[int, int, int] = (2, 2, 2),
) -> PrototypeAssignments:
    """Compute token-to-prototype and token-to-semantic-group cosine assignments."""

    if tokens.ndim != 3 or prototypes.ndim != 3:
        raise ValueError("tokens and prototypes must have shapes [B,N,C] and [B,P,C].")
    if tokens.shape[0] != prototypes.shape[0] or tokens.shape[2] != prototypes.shape[2]:
        raise ValueError(
            f"Incompatible token/prototype shapes: {tuple(tokens.shape)} vs {tuple(prototypes.shape)}."
        )
    if sum(groups) != prototypes.shape[1]:
        raise ValueError(
            f"Prototype groups {groups} sum to {sum(groups)}, but P={prototypes.shape[1]}."
        )

    normalized_tokens = F.normalize(tokens.float(), dim=-1)
    normalized_prototypes = F.normalize(prototypes.float(), dim=-1)
    similarity = normalized_tokens @ normalized_prototypes.transpose(1, 2)
    top2 = similarity.topk(k=min(2, similarity.shape[-1]), dim=-1)
    prototype_index = top2.indices[:, :, 0]
    prototype_margin = (
        top2.values[:, :, 0] - top2.values[:, :, 1]
        if top2.values.shape[-1] > 1
        else torch.ones_like(top2.values[:, :, 0])
    )
    per_group = torch.stack(
        [similarity[:, :, current].max(dim=-1).values for current in group_slices(groups)],
        dim=-1,
    )
    group_top2 = per_group.topk(k=2, dim=-1)
    group_index = group_top2.indices[:, :, 0]
    group_margin = group_top2.values[:, :, 0] - group_top2.values[:, :, 1]
    prototype_matrix = normalized_prototypes @ normalized_prototypes.transpose(1, 2)
    return PrototypeAssignments(
        prototype_similarity=similarity,
        prototype_index=prototype_index,
        prototype_margin=prototype_margin,
        group_similarity=per_group,
        group_index=group_index,
        group_margin=group_margin,
        prototype_matrix=prototype_matrix,
    )


def familiarity_gate_maps(
    objectness: torch.Tensor,
    novelty: torch.Tensor,
    *,
    min_weight: float,
    power: float,
    alpha: float = 1.0,
) -> FamiliarityGateMaps:
    """Reproduce the gate used by ``_aggregate_guided_prototypes``.

    ``guided_weight`` is the full-strength familiarity gate. ``token_weight``
    additionally includes the native-anchor interpolation used at runtime.
    """

    if objectness.shape != novelty.shape:
        raise ValueError(
            f"objectness and novelty must have the same shape, got "
            f"{tuple(objectness.shape)} and {tuple(novelty.shape)}."
        )
    if not 0.0 <= float(min_weight) <= 1.0:
        raise ValueError("min_weight must be in [0, 1].")
    if float(power) <= 0.0:
        raise ValueError("power must be positive.")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1].")

    risk = (objectness * novelty).clamp(0.0, 1.0)
    guided_weight = float(min_weight) + (1.0 - float(min_weight)) * (
        1.0 - risk
    ).pow(float(power))
    token_weight = 1.0 + float(alpha) * (guided_weight - 1.0)
    return FamiliarityGateMaps(
        objectness=objectness,
        novelty=novelty,
        risk=risk,
        guided_weight=guided_weight,
        token_weight=token_weight,
        suppression=1.0 - token_weight,
    )


def first_block_attention_mass(
    block: torch.nn.Module,
    prototype_token: torch.Tensor,
    tokens: torch.Tensor,
    token_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean token attention mass before/after the gate in block one.

    The same prototype queries and token keys are used on both sides, so the
    difference isolates the direct gate effect rather than later query drift.
    Results have shape ``[B, N]`` and each row sums to one.
    """

    if tokens.ndim != 3 or prototype_token.ndim not in {2, 3}:
        raise ValueError("Expected tokens [B,N,C] and prototype_token [P,C] or [B,P,C].")
    if token_weight.shape != tokens.shape[:2]:
        raise ValueError(
            f"token_weight shape {tuple(token_weight.shape)} does not match "
            f"tokens {tuple(tokens.shape[:2])}."
        )
    batch, token_count, channels = tokens.shape
    prototype = (
        prototype_token.unsqueeze(0).expand(batch, -1, -1)
        if prototype_token.ndim == 2
        else prototype_token
    )
    if prototype.shape[0] != batch or prototype.shape[2] != channels:
        raise ValueError("Prototype batch/channel dimensions do not match tokens.")

    attention = block.attn
    heads = int(attention.num_heads)
    normalized_prototype = block.norm1(prototype)
    normalized_tokens = block.norm1(tokens)
    query = attention.q(normalized_prototype).reshape(
        batch, prototype.shape[1], heads, channels // heads
    ).permute(0, 2, 1, 3)
    key_value = attention.kv(normalized_tokens).reshape(
        batch, token_count, 2, heads, channels // heads
    ).permute(2, 0, 3, 1, 4)
    logits = (query @ key_value[0].transpose(-2, -1)) * attention.scale
    before = logits.softmax(dim=-1)
    after = (logits + token_weight.clamp_min(1e-6).log()[:, None, None, :]).softmax(
        dim=-1
    )
    return before.mean(dim=(1, 2)), after.mean(dim=(1, 2))

def tokenize_binary_mask(
    mask: np.ndarray,
    model_input_size: int,
    token_side: int,
    *,
    mode: str = "any",
    threshold: float = 0.5,
) -> np.ndarray:
    """Project an image-space binary mask to the ViT grid.

    ``any`` preserves sub-token FOD targets. ``fraction`` is appropriate for
    validity/ROI masks and applies ``threshold`` to per-token coverage.
    """

    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape={mask.shape}.")
    if model_input_size <= 0 or token_side <= 0:
        raise ValueError("model_input_size and token_side must be positive.")
    if model_input_size % token_side != 0:
        raise ValueError(
            f"Model input {model_input_size} is not divisible by token side {token_side}."
        )
    tensor = torch.from_numpy(mask.astype(np.float32, copy=False))[None, None]
    resized = F.interpolate(
        tensor,
        size=(model_input_size, model_input_size),
        mode="nearest",
    )
    kernel = model_input_size // token_side
    if mode == "any":
        pooled = F.max_pool2d(resized, kernel_size=kernel, stride=kernel)
        result = pooled > 0.0
    elif mode == "fraction":
        pooled = F.avg_pool2d(resized, kernel_size=kernel, stride=kernel)
        result = pooled >= float(threshold)
    else:
        raise ValueError(f"Unknown token mask mode: {mode!r}.")
    return result[0, 0].cpu().numpy().astype(bool)


def normalized_entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    if total <= 0.0 or counts.size <= 1:
        return 0.0
    probabilities = counts[counts > 0.0] / total
    return float(-(probabilities * np.log(probabilities)).sum() / math.log(counts.size))


def summarize_assignments(
    assignments: PrototypeAssignments,
    valid_mask: np.ndarray,
    gt_mask: np.ndarray,
    prior_group_index: np.ndarray | None = None,
    groups: tuple[int, int, int] = (2, 2, 2),
) -> dict[str, float | list[float]]:
    """Summarize one batch-size-one crop without changing model state."""

    if assignments.prototype_index.shape[0] != 1:
        raise ValueError("summarize_assignments currently expects batch size one.")
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    gt = np.asarray(gt_mask, dtype=bool).reshape(-1) & valid
    if valid.shape[0] != assignments.prototype_index.shape[1]:
        raise ValueError("Token mask size does not match assignment count.")
    background = valid & ~gt
    prototype_index = assignments.prototype_index[0].detach().cpu().numpy()
    group_index = assignments.group_index[0].detach().cpu().numpy()
    prototype_margin = assignments.prototype_margin[0].detach().cpu().numpy()
    group_margin = assignments.group_margin[0].detach().cpu().numpy()
    prototype_count = assignments.prototype_matrix.shape[-1]

    def fractions(index: np.ndarray, selection: np.ndarray, count: int) -> list[float]:
        selected = index[selection]
        if selected.size == 0:
            return [0.0] * count
        values = np.bincount(selected, minlength=count).astype(np.float64)
        return (values / values.sum()).tolist()

    matrix = assignments.prototype_matrix[0].detach().cpu().numpy()
    diagonal = np.eye(prototype_count, dtype=bool)
    all_pairs = matrix[~diagonal]
    if sum(groups) != prototype_count:
        raise ValueError(f"Prototype groups {groups} do not match P={prototype_count}.")
    prototype_groups = np.concatenate(
        [np.full(size, group_index, dtype=np.int64) for group_index, size in enumerate(groups)]
    )
    upper = np.triu(np.ones((prototype_count, prototype_count), dtype=bool), k=1)
    within = upper & (prototype_groups[:, None] == prototype_groups[None, :])
    between = upper & (prototype_groups[:, None] != prototype_groups[None, :])
    within_mean = float(matrix[within].mean()) if within.any() else 0.0
    between_mean = float(matrix[between].mean()) if between.any() else 0.0
    result: dict[str, float | list[float]] = {
        "valid_tokens": float(valid.sum()),
        "gt_tokens": float(gt.sum()),
        "prototype_occupancy": fractions(prototype_index, valid, prototype_count),
        "prototype_usage_entropy": normalized_entropy(
            np.bincount(prototype_index[valid], minlength=prototype_count)
        ),
        "group_occupancy": fractions(group_index, valid, 3),
        "gt_group_occupancy": fractions(group_index, gt, 3),
        "background_group_occupancy": fractions(group_index, background, 3),
        "prototype_margin_mean": float(prototype_margin[valid].mean()) if valid.any() else 0.0,
        "group_margin_mean": float(group_margin[valid].mean()) if valid.any() else 0.0,
        "prototype_offdiag_cosine_mean": float(all_pairs.mean()) if all_pairs.size else 0.0,
        "prototype_within_group_cosine_mean": within_mean,
        "prototype_between_group_cosine_mean": between_mean,
        "prototype_group_separation_gap": within_mean - between_mean,
    }
    if prior_group_index is not None:
        prior = np.asarray(prior_group_index).reshape(-1)
        if prior.shape != group_index.shape:
            raise ValueError("prior_group_index does not match token assignments.")
        result["prior_affinity_group_agreement"] = (
            float((prior[valid] == group_index[valid]).mean()) if valid.any() else 0.0
        )
    return result
