from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PerspectiveMaskConfig:
    """Token-level masks for feature reconstruction.

    ``strategy="perspective"`` keeps the previous fixed row-band structural
    masks. ``strategy="uniform"`` keeps four complementary phases at one
    row-independent block size. ``strategy="uniform_single"`` uses one fixed
    token-grid mask with no row-distance prior. ``strategy="adaptive"`` uses a
    learned planner to rank feature tokens and masks rank segments without any
    row-distance prior.
    """

    strategy: str = "perspective"
    band_block_sizes: tuple[int, ...] = (1, 1, 2, 2, 4, 4)
    fill: str = "visible_mean"
    prototype_source: str = "masked"
    adaptive_mask_ratio: float = 0.25
    adaptive_segments: int = 4
    adaptive_temperature: float = 1.0
    adaptive_dilate: int = 0
    inp_mask_mode: str = "standard"
    local_context_radius: int = 2
    candidate_mask_ratio: float = 0.08
    candidate_segments: int = 4
    candidate_min_score: float = 0.0
    candidate_dilate: int = 1
    candidate_prior_weight: float = 0.60
    candidate_prompt: bool = False
    prototype_grid_ratio: float = 0.20
    prototype_grid_threshold: float = 0.0
    prototype_grid_block_size: int = 2
    prototype_handling: str = "standard"


def parse_block_sizes(text: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(text, str):
        values = [int(part) for part in text.replace(",", " ").split() if part.strip()]
    else:
        values = [int(value) for value in text]
    if not values:
        raise ValueError("At least one block size is required.")
    if any(value <= 0 for value in values):
        raise ValueError(f"Block sizes must be positive: {values}")
    return tuple(values)


class AdaptiveMaskPlanner(nn.Module):
    """Predict token mask scores from frozen normal patch features."""

    def __init__(self, embed_dim: int, hidden_dim: int = 192) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, spatial_tokens: torch.Tensor, side: int) -> torch.Tensor:
        batch, token_count, channels = spatial_tokens.shape
        if token_count != side * side:
            raise RuntimeError(f"Expected {side * side} spatial tokens, got {token_count}.")
        x = self.norm(spatial_tokens)
        x = x.transpose(1, 2).reshape(batch, channels, side, side).contiguous()
        return self.net(x)


def attach_adaptive_mask_planner(model, hidden_dim: int = 192) -> nn.Module:
    """Attach a trainable planner module to a Dinomaly/INP model."""

    if hasattr(model, "adaptive_mask_planner"):
        return model.adaptive_mask_planner
    embed_dim = getattr(model.encoder, "embed_dim", None)
    if embed_dim is None:
        embed_dim = getattr(model.encoder, "num_features", None)
    if embed_dim is None:
        patch_embed = getattr(model.encoder, "patch_embed", None)
        proj = getattr(patch_embed, "proj", None)
        embed_dim = getattr(proj, "out_channels", None)
    if embed_dim is None:
        raise RuntimeError("Cannot infer encoder feature dimension for AdaptiveMaskPlanner.")
    planner = AdaptiveMaskPlanner(int(embed_dim), hidden_dim=hidden_dim).to(next(model.parameters()).device)
    model.add_module("adaptive_mask_planner", planner)
    return planner


class LocalContextReconstructor(nn.Module):
    """Rebuild selected INP-Former tokens from visible local neighbors."""

    def __init__(self, embed_dim: int, radius: int = 2) -> None:
        super().__init__()
        self.radius = int(radius)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, tokens: torch.Tensor, spatial_mask: torch.Tensor, start: int, fill: str) -> torch.Tensor:
        batch, token_count, channels = tokens.shape
        spatial = tokens[:, start:, :]
        side = int(math.sqrt(spatial.shape[1]))
        if side * side != spatial.shape[1]:
            raise RuntimeError(f"Cannot infer square token grid from spatial token count={spatial.shape[1]}.")

        query_tokens = _masked_tokens(tokens, spatial_mask, start=start, fill=fill)[:, start:, :]
        q = self.q(self.norm_q(query_tokens))
        kv_tokens = self.norm_kv(spatial)
        k = self.k(kv_tokens)
        v = self.v(kv_tokens)

        kernel = max(3, self.radius * 2 + 1)
        padding = kernel // 2

        def unfold_token_map(values: torch.Tensor) -> torch.Tensor:
            fmap = values.transpose(1, 2).reshape(batch, channels, side, side).contiguous()
            patches = F.unfold(fmap, kernel_size=kernel, padding=padding)
            patches = patches.transpose(1, 2).reshape(batch, side * side, kernel * kernel, channels)
            return patches

        k_neigh = unfold_token_map(k)
        v_neigh = unfold_token_map(v)
        visible = (~spatial_mask[:, 0].bool()).to(dtype=tokens.dtype).unsqueeze(1)
        visible_neigh = F.unfold(visible, kernel_size=kernel, padding=padding).transpose(1, 2) > 0.5

        scores = (q.unsqueeze(2) * k_neigh).sum(dim=-1) * self.scale
        scores = scores.masked_fill(~visible_neigh, -1.0e4)
        attn = torch.softmax(scores, dim=-1).unsqueeze(-1)
        context = (attn * v_neigh).sum(dim=2)

        has_visible = visible_neigh.any(dim=-1, keepdim=True)
        visible_flat = visible.flatten(2).transpose(1, 2)
        fallback = (spatial * visible_flat).sum(dim=1, keepdim=True) / visible_flat.sum(dim=1, keepdim=True).clamp_min(1.0)
        context = torch.where(has_visible, context, fallback.expand_as(context))
        local = self.proj(context)

        if start <= 0:
            return local
        return torch.cat([tokens[:, :start, :], local], dim=1)


class PromptedLocalContextLayer(nn.Module):
    """Layer-wise local visible-neighbor aggregation for masked INP tokens."""

    def __init__(self, embed_dim: int, radius: int = 2) -> None:
        super().__init__()
        self.radius = int(radius)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm_mlp = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.scale = embed_dim ** -0.5

    def _local_visible_mean(
        self,
        values: torch.Tensor,
        visible: torch.Tensor,
        side: int,
    ) -> torch.Tensor:
        batch, token_count, channels = values.shape
        if token_count != side * side:
            raise RuntimeError(f"Expected {side * side} spatial tokens, got {token_count}.")
        kernel = max(3, self.radius * 2 + 1)
        padding = kernel // 2
        fmap = values.transpose(1, 2).reshape(batch, channels, side, side).contiguous()
        masked_fmap = fmap * visible
        channel_kernel = fmap.new_ones((channels, 1, kernel, kernel))
        count_kernel = fmap.new_ones((1, 1, kernel, kernel))
        sums = F.conv2d(masked_fmap, channel_kernel, padding=padding, groups=channels)
        counts = F.conv2d(visible, count_kernel, padding=padding).clamp_min(1.0)
        context = sums / counts
        return context.flatten(2).transpose(1, 2).contiguous()

    def forward(self, tokens: torch.Tensor, spatial_mask: torch.Tensor, start: int) -> torch.Tensor:
        spatial = tokens[:, start:, :]
        side = int(math.sqrt(spatial.shape[1]))
        if side * side != spatial.shape[1]:
            raise RuntimeError(f"Cannot infer square token grid from spatial token count={spatial.shape[1]}.")

        q = self.q(self.norm_q(spatial))
        kv_tokens = self.norm_kv(spatial)
        v = self.v(kv_tokens)
        visible = (~spatial_mask[:, 0].bool()).to(dtype=tokens.dtype).unsqueeze(1)
        context = self._local_visible_mean(v, visible, side)
        gate = torch.sigmoid((q * self.k(kv_tokens)).sum(dim=-1, keepdim=True) * self.scale)
        out = self.proj(context + gate * q)
        out = out + self.mlp(self.norm_mlp(out))
        if start <= 0:
            return out
        return torch.cat([tokens[:, :start, :], out], dim=1)


class PromptedContextMaskDecoder(nn.Module):
    """Prompted context branch used inside each INP-Former decoder layer.

    Visible tokens keep the original INP prototype-guided decoder path. Masked
    tokens use local aggregation over nearby visible tokens at the same decoder
    depth. Type prompts tell the shared decoder loop whether a spatial token is
    visible or masked without exposing the masked token's original feature.
    """

    def __init__(self, embed_dim: int, num_layers: int, radius: int = 2) -> None:
        super().__init__()
        self.visible_prompt = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_prompt = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.layers = nn.ModuleList(
            [PromptedLocalContextLayer(embed_dim, radius=radius) for _ in range(int(num_layers))]
        )
        nn.init.normal_(self.visible_prompt, mean=0.0, std=1e-3)
        nn.init.normal_(self.mask_prompt, mean=0.0, std=1e-3)

    def add_prompts(self, tokens: torch.Tensor, spatial_mask: torch.Tensor, start: int) -> torch.Tensor:
        mask_flat = spatial_mask[:, 0].flatten(1).to(dtype=tokens.dtype).unsqueeze(-1)
        spatial = tokens[:, start:, :]
        prompts = self.visible_prompt * (1.0 - mask_flat) + self.mask_prompt * mask_flat
        spatial = spatial + prompts
        if start <= 0:
            return spatial
        return torch.cat([tokens[:, :start, :], spatial], dim=1)

    def context(self, layer_index: int, tokens: torch.Tensor, spatial_mask: torch.Tensor, start: int) -> torch.Tensor:
        if layer_index >= len(self.layers):
            raise RuntimeError(f"Missing context layer {layer_index}; module has {len(self.layers)} layers.")
        return self.layers[layer_index](tokens, spatial_mask, start=start)


def attach_local_context_reconstructor(model, radius: int = 2) -> nn.Module:
    """Attach the prompted local-context decoder branch used by INP-Former."""

    if hasattr(model, "local_context_reconstructor"):
        return model.local_context_reconstructor
    embed_dim = getattr(model.encoder, "embed_dim", None)
    if embed_dim is None:
        embed_dim = getattr(model.encoder, "num_features", None)
    if embed_dim is None:
        patch_embed = getattr(model.encoder, "patch_embed", None)
        proj = getattr(patch_embed, "proj", None)
        embed_dim = getattr(proj, "out_channels", None)
    if embed_dim is None:
        prototype = getattr(model, "prototype_token", None)
        embed_dim = getattr(prototype, "shape", [None, None])[-1] if prototype is not None else None
    if embed_dim is None:
        raise RuntimeError("Cannot infer encoder feature dimension for LocalContextReconstructor.")
    decoder_layers = getattr(model, "decoder", None)
    num_layers = len(decoder_layers) if decoder_layers is not None else 1
    reconstructor = PromptedContextMaskDecoder(int(embed_dim), num_layers=num_layers, radius=radius).to(
        next(model.parameters()).device
    )
    model.add_module("local_context_reconstructor", reconstructor)
    return reconstructor


class CandidateMaskPrompt(nn.Module):
    """A tiny token prompt that marks candidate-masked INP positions."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.prompt = nn.Parameter(torch.zeros(1, 1, int(embed_dim)))
        nn.init.normal_(self.prompt, mean=0.0, std=1e-3)


def attach_candidate_mask_prompt(model) -> nn.Module:
    """Attach the prompt used by candidate-style irregular masked reconstruction."""

    if hasattr(model, "candidate_mask_prompt"):
        return model.candidate_mask_prompt
    embed_dim = getattr(model.encoder, "embed_dim", None)
    if embed_dim is None:
        embed_dim = getattr(model.encoder, "num_features", None)
    if embed_dim is None:
        patch_embed = getattr(model.encoder, "patch_embed", None)
        proj = getattr(patch_embed, "proj", None)
        embed_dim = getattr(proj, "out_channels", None)
    if embed_dim is None:
        prototype = getattr(model, "prototype_token", None)
        embed_dim = getattr(prototype, "shape", [None, None])[-1] if prototype is not None else None
    if embed_dim is None:
        raise RuntimeError("Cannot infer encoder feature dimension for CandidateMaskPrompt.")
    prompt = CandidateMaskPrompt(int(embed_dim)).to(next(model.parameters()).device)
    model.add_module("candidate_mask_prompt", prompt)
    return prompt


def load_adaptive_mask_planner_checkpoint(
    model,
    checkpoint: str | Path,
    device: torch.device,
    hidden_dim: int = 192,
    strict: bool = True,
) -> nn.Module:
    """Attach and load a separately trained adaptive mask planner."""

    planner = attach_adaptive_mask_planner(model, hidden_dim=hidden_dim)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict):
        if "planner" in state:
            state = state["planner"]
        elif "adaptive_mask_planner" in state:
            state = state["adaptive_mask_planner"]
        elif "model" in state:
            prefix = "adaptive_mask_planner."
            state = {
                key[len(prefix) :]: value
                for key, value in state["model"].items()
                if key.startswith(prefix)
            }
    if not state:
        raise RuntimeError(f"No adaptive mask planner weights found in {checkpoint}")
    planner.load_state_dict(state, strict=strict)
    return planner


def _normalize_map(values: torch.Tensor) -> torch.Tensor:
    flat = values.flatten(1)
    vmin = flat.min(dim=1).values.view(-1, 1, 1, 1)
    vmax = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return ((values - vmin) / (vmax - vmin).clamp_min(1e-6)).clamp(0.0, 1.0)


def feature_adaptive_target(spatial_tokens: torch.Tensor, side: int) -> torch.Tensor:
    """Build a normal-feature pseudo target without perspective/location priors."""

    batch, token_count, channels = spatial_tokens.shape
    if token_count != side * side:
        raise RuntimeError(f"Expected {side * side} spatial tokens, got {token_count}.")
    fmap = F.normalize(spatial_tokens.detach(), dim=-1).transpose(1, 2).reshape(batch, channels, side, side)
    local_mean = F.avg_pool2d(fmap, kernel_size=3, stride=1, padding=1, count_include_pad=False)
    context_error = (1.0 - F.cosine_similarity(fmap, local_mean, dim=1, eps=1e-6).unsqueeze(1)).clamp_min(0.0)
    mean_sq = F.avg_pool2d(fmap * fmap, kernel_size=3, stride=1, padding=1, count_include_pad=False)
    local_var = (mean_sq - local_mean * local_mean).mean(dim=1, keepdim=True).clamp_min(0.0)
    return _normalize_map(context_error + 0.5 * _normalize_map(local_var))


def _rank_segment_mask(logits: torch.Tensor, ratio: float, pattern: int, segments: int) -> torch.Tensor:
    batch, _, height, width = logits.shape
    flat = logits.flatten(1)
    total = flat.shape[1]
    k = max(1, int(round(total * float(ratio))))
    segments = max(1, int(segments))
    segment = int(pattern) % segments
    offset = min(segment * k, max(total - 1, 0))
    end = min(offset + k, total)
    if end <= offset:
        offset = max(0, total - k)
        end = total
    order = torch.argsort(flat, dim=1, descending=True)
    selected = order[:, offset:end]
    mask = torch.zeros((batch, total), dtype=torch.bool, device=logits.device)
    mask.scatter_(1, selected, True)
    return mask.view(batch, 1, height, width)


def perspective_structural_mask(
    batch: int,
    side: int,
    pattern: int,
    config: PerspectiveMaskConfig,
    device: torch.device,
) -> torch.Tensor:
    """Return a boolean mask of shape ``[B, 1, side, side]``.

    Patterns 0..3 select one quadrant in each 2x2 block grid. Across four
    patterns, every token is masked once for a fixed block size. Per-row-band
    block sizes make the mask finer in far runway rows and coarser nearby.
    """

    bands = config.band_block_sizes
    phase_y = int(pattern) // 2
    phase_x = int(pattern) % 2
    mask = torch.zeros((side, side), dtype=torch.bool, device=device)
    yy = torch.arange(side, device=device).view(side, 1).expand(side, side)
    xx = torch.arange(side, device=device).view(1, side).expand(side, side)
    for band, block in enumerate(bands):
        y0 = int(round(band * side / len(bands)))
        y1 = int(round((band + 1) * side / len(bands)))
        if y1 <= y0:
            continue
        local_y = yy[y0:y1] - y0
        local_x = xx[y0:y1]
        selected = ((local_y // block) % 2 == phase_y) & ((local_x // block) % 2 == phase_x)
        mask[y0:y1] = selected
    return mask.view(1, 1, side, side).expand(batch, 1, side, side)


def uniform_single_structural_mask(
    batch: int,
    side: int,
    config: PerspectiveMaskConfig,
    device: torch.device,
) -> torch.Tensor:
    """Return one position-agnostic structural mask.

    A single block size must be supplied. The mask selects the upper-left
    quadrant in every 2x2 block grid, matching one Full-mask pass (25% token
    density) while removing both row-conditioned scale and complementary
    multi-pattern coverage.
    """

    if len(config.band_block_sizes) != 1:
        raise ValueError(
            "uniform_single requires exactly one --mask-band-block-sizes value; "
            f"got {config.band_block_sizes}."
        )
    block = int(config.band_block_sizes[0])
    yy = torch.arange(side, device=device).view(side, 1).expand(side, side)
    xx = torch.arange(side, device=device).view(1, side).expand(side, side)
    selected = ((yy // block) % 2 == 0) & ((xx // block) % 2 == 0)
    return selected.view(1, 1, side, side).expand(batch, 1, side, side)


def uniform_structural_mask(
    batch: int,
    side: int,
    pattern: int,
    config: PerspectiveMaskConfig,
    device: torch.device,
) -> torch.Tensor:
    """Return one of four complementary masks at a uniform token scale."""

    if len(config.band_block_sizes) != 1:
        raise ValueError(
            "uniform requires exactly one --mask-band-block-sizes value; "
            f"got {config.band_block_sizes}."
        )
    block = int(config.band_block_sizes[0])
    phase_y = int(pattern) // 2
    phase_x = int(pattern) % 2
    if not 0 <= int(pattern) <= 3:
        raise ValueError(f"uniform supports complementary patterns 0..3, got {pattern}.")
    yy = torch.arange(side, device=device).view(side, 1).expand(side, side)
    xx = torch.arange(side, device=device).view(1, side).expand(side, side)
    selected = ((yy // block) % 2 == phase_y) & ((xx // block) % 2 == phase_x)
    return selected.view(1, 1, side, side).expand(batch, 1, side, side)


def feature_adaptive_mask(
    model,
    target_tokens: torch.Tensor,
    start: int,
    side: int,
    pattern: int,
    config: PerspectiveMaskConfig,
) -> torch.Tensor:
    if not hasattr(model, "adaptive_mask_planner"):
        raise RuntimeError("Adaptive mask strategy requires attach_adaptive_mask_planner(model) before forward.")
    spatial_tokens = target_tokens[:, start:, :].detach()
    logits = model.adaptive_mask_planner(spatial_tokens, side)
    temperature = max(float(config.adaptive_temperature), 1e-3)
    probs = torch.sigmoid(logits / temperature)
    target = feature_adaptive_target(spatial_tokens, side)
    mask = _rank_segment_mask(
        logits=logits,
        ratio=config.adaptive_mask_ratio,
        pattern=pattern,
        segments=config.adaptive_segments,
    )
    if config.adaptive_dilate > 0:
        radius = int(config.adaptive_dilate)
        kernel = radius * 2 + 1
        mask = F.max_pool2d(mask.float(), kernel_size=kernel, stride=1, padding=radius) > 0
    model._adaptive_mask_state = {
        "logits": logits,
        "probs": probs,
        "target": target,
        "mask": mask,
    }
    return mask


def candidate_guided_mask(
    model,
    target_tokens: torch.Tensor,
    start: int,
    side: int,
    pattern: int,
    config: PerspectiveMaskConfig,
) -> torch.Tensor:
    """Build candidate-like irregular masks from current normal token priors.

    This intentionally mirrors inference-time candidate selection: high local
    feature contrast and high guided objectness get masked first, while
    texture-like normal areas are less likely to be selected.
    """

    spatial_tokens = target_tokens[:, start:, :].detach()
    feature_score = feature_adaptive_target(spatial_tokens, side)
    score = feature_score

    guided_config = getattr(model, "_guided_prototype_config", None)
    if guided_config is not None:
        try:
            from .prototype_guidance import _build_trainable_priors

            with torch.no_grad():
                priors, _ = _build_trainable_priors(
                    model,
                    spatial_tokens,
                    guided_config,
                    getattr(model, "_guided_prototype_image", None),
                )
            prior_map = priors.view(spatial_tokens.shape[0], side, side, 3).permute(0, 3, 1, 2).contiguous()
            objectness = prior_map[:, 2:3].clamp(0.0, 1.0)
            texture = prior_map[:, 1:2].clamp(0.0, 1.0)
            prior_score = _normalize_map(objectness * (1.0 - texture).clamp_min(0.0))
            prior_weight = min(max(float(config.candidate_prior_weight), 0.0), 1.0)
            score = (1.0 - prior_weight) * feature_score + prior_weight * prior_score
            score = _normalize_map(score)
        except Exception:
            # Candidate masks are a training/eval aid; fall back to pure feature
            # contrast if a non-guided model or partial checkpoint lacks priors.
            score = feature_score

    mask = _rank_segment_mask(
        logits=score,
        ratio=float(config.candidate_mask_ratio),
        pattern=pattern,
        segments=max(1, int(config.candidate_segments)),
    )
    if config.candidate_min_score > 0.0:
        mask = mask & (score >= float(config.candidate_min_score))
    if config.candidate_dilate > 0:
        radius = int(config.candidate_dilate)
        kernel = radius * 2 + 1
        mask = F.max_pool2d(mask.float(), kernel_size=kernel, stride=1, padding=radius) > 0
    model._candidate_mask_state = {
        "score": score.detach(),
        "mask": mask.detach(),
    }
    return mask


def prototype_guided_grid_mask(
    model,
    target_tokens: torch.Tensor,
    start: int,
    side: int,
    pattern: int,
    config: PerspectiveMaskConfig,
) -> torch.Tensor:
    """Mask complementary phases only inside high-risk prototype grid cells."""

    if not 0 <= int(pattern) <= 3:
        raise ValueError(f"prototype_grid supports complementary patterns 0..3, got {pattern}.")
    risk = getattr(model, "_guided_decoder_risk", None)
    if not isinstance(risk, torch.Tensor):
        raise RuntimeError(
            "prototype_grid requires a preliminary Guided Prototype aggregation "
            "to expose _guided_decoder_risk."
        )
    spatial_risk = risk[:, start:]
    if spatial_risk.shape[1] != side * side:
        raise RuntimeError(
            f"Prototype risk/token mismatch: risk={tuple(risk.shape)} start={start} side={side}."
        )
    block = int(config.prototype_grid_block_size)
    if block <= 0 or side % block:
        raise ValueError(
            f"prototype_grid block size must divide token side: block={block}, side={side}."
        )
    risk_map = spatial_risk.reshape(spatial_risk.shape[0], 1, side, side)
    block_score = F.max_pool2d(risk_map, kernel_size=block, stride=block)
    flat = block_score.flatten(1)
    max_ratio = min(max(float(config.prototype_grid_ratio), 0.0), 1.0)
    keep_count = min(flat.shape[1], max(0, int(math.ceil(max_ratio * flat.shape[1]))))
    support_flat = flat >= float(config.prototype_grid_threshold)
    if keep_count == 0:
        support_flat.zero_()
    elif keep_count < flat.shape[1]:
        top_indices = flat.topk(keep_count, dim=1, largest=True, sorted=False).indices
        cap = torch.zeros_like(support_flat)
        cap.scatter_(1, top_indices, True)
        support_flat &= cap
    block_support = support_flat.reshape_as(block_score)
    support = block_support.repeat_interleave(block, dim=2).repeat_interleave(block, dim=3)

    yy = torch.arange(side, device=target_tokens.device).view(side, 1)
    xx = torch.arange(side, device=target_tokens.device).view(1, side)
    phase_y = int(pattern) // 2
    phase_x = int(pattern) % 2
    phase = ((yy % 2) == phase_y) & ((xx % 2) == phase_x)
    mask = support & phase.view(1, 1, side, side)
    model._prototype_grid_mask_state = {
        "score": risk_map.detach(),
        "block_score": block_score.detach(),
        "support": support.detach(),
        "mask": mask.detach(),
    }
    return mask


def make_spatial_mask(
    model,
    target_tokens: torch.Tensor,
    start: int,
    batch: int,
    side: int,
    pattern: int,
    config: PerspectiveMaskConfig,
    device: torch.device,
) -> torch.Tensor:
    if config.strategy == "perspective":
        return perspective_structural_mask(batch, side, pattern, config, device)
    if config.strategy == "uniform":
        return uniform_structural_mask(batch, side, pattern, config, device)
    if config.strategy == "uniform_single":
        return uniform_single_structural_mask(batch, side, config, device)
    if config.strategy == "adaptive":
        return feature_adaptive_mask(model, target_tokens, start, side, pattern, config)
    if config.strategy == "candidate":
        return candidate_guided_mask(model, target_tokens, start, side, pattern, config)
    if config.strategy == "prototype_grid":
        return prototype_guided_grid_mask(model, target_tokens, start, side, pattern, config)
    raise ValueError(f"Unsupported mask strategy: {config.strategy}")


def _resolve_spatial_mask(
    spatial_mask: torch.Tensor | None,
    model,
    target_tokens: torch.Tensor,
    start: int,
    batch: int,
    side: int,
    pattern: int,
    config: PerspectiveMaskConfig,
    device: torch.device,
) -> torch.Tensor:
    if spatial_mask is None:
        return make_spatial_mask(
            model,
            target_tokens,
            start=start,
            batch=batch,
            side=side,
            pattern=pattern,
            config=config,
            device=device,
        )
    if spatial_mask.shape != (batch, 1, side, side):
        raise ValueError(f"Expected spatial_mask shape {(batch, 1, side, side)}, got {tuple(spatial_mask.shape)}.")
    return spatial_mask.to(device=device, dtype=torch.bool)


def _masked_tokens(tokens: torch.Tensor, spatial_mask: torch.Tensor, start: int, fill: str) -> torch.Tensor:
    out = tokens.clone()
    mask_flat = spatial_mask[:, 0].flatten(1)
    spatial = out[:, start:, :]
    if fill == "zero":
        fill_values = torch.zeros((tokens.shape[0], 1, tokens.shape[2]), dtype=tokens.dtype, device=tokens.device)
    elif fill == "visible_mean":
        visible = (~mask_flat).to(tokens.dtype).unsqueeze(-1)
        denom = visible.sum(dim=1, keepdim=True).clamp_min(1.0)
        fill_values = (spatial * visible).sum(dim=1, keepdim=True) / denom
    else:
        raise ValueError(f"Unsupported mask fill: {fill}")
    spatial = torch.where(mask_flat.unsqueeze(-1), fill_values.expand_as(spatial), spatial)
    out[:, start:, :] = spatial
    return out


def _add_candidate_prompt(
    model,
    tokens: torch.Tensor,
    spatial_mask: torch.Tensor,
    start: int,
    config: PerspectiveMaskConfig,
) -> torch.Tensor:
    if config.strategy != "candidate" or not config.candidate_prompt:
        return tokens
    prompt_module = getattr(model, "candidate_mask_prompt", None)
    if prompt_module is None:
        prompt_module = attach_candidate_mask_prompt(model)
    mask_flat = spatial_mask[:, 0].flatten(1).to(dtype=tokens.dtype).unsqueeze(-1)
    out = tokens.clone()
    out[:, start:, :] = out[:, start:, :] + mask_flat * prompt_module.prompt.to(dtype=tokens.dtype)
    return out


def _tokens_to_maps(tokens: Iterable[torch.Tensor], batch: int, side: int) -> list[torch.Tensor]:
    return [
        value.permute(0, 2, 1).reshape(batch, -1, side, side).contiguous()
        for value in tokens
    ]


def _forward_masked_vitill(
    model,
    images: torch.Tensor,
    pattern: int,
    config: PerspectiveMaskConfig,
    spatial_mask: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Run a ViTill-style model with structural masks before the bottleneck."""

    x = model.encoder.prepare_tokens(images)
    en_list = []
    for i, blk in enumerate(model.encoder.blocks):
        if i <= model.target_layers[-1]:
            if i in model.encoder_require_grad_layer:
                x = blk(x)
            else:
                with torch.no_grad():
                    x = blk(x)
        else:
            continue
        if i in model.target_layers:
            en_list.append(x)
    if not en_list:
        raise RuntimeError("No encoder target features were captured.")

    start = 1 + model.encoder.num_register_tokens
    side = int(math.sqrt(en_list[0].shape[1] - start))
    if side * side != en_list[0].shape[1] - start:
        raise RuntimeError(f"Cannot infer square token grid from shape {en_list[0].shape}.")
    batch = images.shape[0]

    if model.remove_class_token:
        en_list = [e[:, start:, :] for e in en_list]
        start = 0

    target_tokens = model.fuse_feature(en_list)
    spatial_mask = _resolve_spatial_mask(
        spatial_mask,
        model,
        target_tokens,
        start=start,
        batch=batch,
        side=side,
        pattern=pattern,
        config=config,
        device=images.device,
    )
    x = _masked_tokens(target_tokens, spatial_mask, start=start, fill=config.fill)
    x = _add_candidate_prompt(model, x, spatial_mask, start=start, config=config)

    for blk in model.bottleneck:
        x = blk(x)

    attn_mask = model.generate_mask(side, x.device) if model.mask_neighbor_size > 0 else None
    de_list = []
    for blk in model.decoder:
        x = blk(x, attn_mask=attn_mask)
        de_list.append(x)
    de_list = de_list[::-1]

    en = [model.fuse_feature([en_list[idx] for idx in idxs]) for idxs in model.fuse_layer_encoder]
    de = [model.fuse_feature([de_list[idx] for idx in idxs]) for idxs in model.fuse_layer_decoder]

    if not model.remove_class_token:
        en = [e[:, start:, :] for e in en]
        de = [d[:, start:, :] for d in de]

    return _tokens_to_maps(en, batch, side), _tokens_to_maps(de, batch, side), spatial_mask


def _forward_masked_inpformer(
    model,
    images: torch.Tensor,
    pattern: int,
    config: PerspectiveMaskConfig,
    spatial_mask: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Run an INP-Former model with structural masks before prototype decoding.

    The encoder target remains the unmasked fused normal feature. The masked
    feature is used both to aggregate prototypes and as decoder input, which
    keeps the reconstruction path from directly copying masked tokens.
    """

    if getattr(model, "_guided_prototype_enabled", False):
        model._guided_prototype_image = images.detach()

    x = model.encoder.prepare_tokens(images)
    en_list = []
    for i, blk in enumerate(model.encoder.blocks):
        if i <= model.target_layers[-1]:
            if i in model.encoder_require_grad_layer:
                x = blk(x)
            else:
                with torch.no_grad():
                    x = blk(x)
        else:
            continue
        if i in model.target_layers:
            en_list.append(x)
    if not en_list:
        raise RuntimeError("No encoder target features were captured.")

    original_start = 1 + model.encoder.num_register_tokens
    side = int(math.sqrt(en_list[0].shape[1] - original_start))
    if side * side != en_list[0].shape[1] - original_start:
        raise RuntimeError(f"Cannot infer square token grid from shape {en_list[0].shape}.")
    batch = images.shape[0]

    start = original_start
    if model.remove_class_token:
        en_list = [e[:, original_start:, :] for e in en_list]
        start = 0

    target_tokens = model.fuse_feature(en_list)
    frozen_mask_prototype = config.prototype_handling in {
        "masked_frozen",
        "exclude_detach",
    }
    if config.strategy == "prototype_grid":
        if not hasattr(model, "aggregate_guided_prototypes"):
            raise RuntimeError("prototype_grid is only implemented for Guided INP-Former.")
        previous_update_stats = getattr(model, "_guided_update_prior_stats", None)
        model._guided_update_prior_stats = False
        try:
            with torch.no_grad():
                model.aggregate_guided_prototypes(
                    target_tokens,
                    images,
                    prototype_context=target_tokens,
                )
        finally:
            if previous_update_stats is None:
                delattr(model, "_guided_update_prior_stats")
            else:
                model._guided_update_prior_stats = previous_update_stats
    spatial_mask = _resolve_spatial_mask(
        spatial_mask,
        model,
        target_tokens,
        start=start,
        batch=batch,
        side=side,
        pattern=pattern,
        config=config,
        device=images.device,
    )
    masked_tokens = _masked_tokens(target_tokens, spatial_mask, start=start, fill=config.fill)
    masked_tokens = _add_candidate_prompt(model, masked_tokens, spatial_mask, start=start, config=config)

    if config.prototype_source == "masked":
        prototype_context = masked_tokens
    elif config.prototype_source == "full":
        prototype_context = target_tokens
    else:
        raise ValueError(f"Unsupported INP prototype source: {config.prototype_source}")

    if config.prototype_handling == "exclude_detach":
        grid_state = getattr(model, "_prototype_grid_mask_state", None)
        if not isinstance(grid_state, dict) or "support" not in grid_state:
            raise RuntimeError("exclude_detach requires a prototype_grid support mask.")
        exclusion = grid_state["support"][:, 0].flatten(1)
        if start:
            prefix = exclusion.new_zeros((batch, start))
            exclusion = torch.cat([prefix, exclusion], dim=1)
        previous_update_stats = getattr(model, "_guided_update_prior_stats", None)
        model._guided_update_prior_stats = False
        model._guided_external_aggregation_exclusion = exclusion
        try:
            with torch.no_grad():
                agg_prototype = model.aggregate_guided_prototypes(
                    target_tokens,
                    images,
                    prototype_context=target_tokens,
                )
        finally:
            delattr(model, "_guided_external_aggregation_exclusion")
            if previous_update_stats is None:
                delattr(model, "_guided_update_prior_stats")
            else:
                model._guided_update_prior_stats = previous_update_stats
        agg_prototype = agg_prototype.detach()
    elif hasattr(model, "aggregate_guided_prototypes"):
        aggregation_context = (
            torch.no_grad() if frozen_mask_prototype else torch.enable_grad()
        )
        previous_update_stats = getattr(model, "_guided_update_prior_stats", None)
        if frozen_mask_prototype:
            model._guided_update_prior_stats = False
        try:
            with aggregation_context:
                agg_prototype = model.aggregate_guided_prototypes(
                    target_tokens,
                    images,
                    prototype_context=prototype_context,
                )
        finally:
            if frozen_mask_prototype:
                if previous_update_stats is None:
                    delattr(model, "_guided_update_prior_stats")
                else:
                    model._guided_update_prior_stats = previous_update_stats
        if frozen_mask_prototype:
            agg_prototype = agg_prototype.detach()
    else:
        agg_prototype = model.prototype_token
        for blk in model.aggregation:
            agg_prototype = blk(agg_prototype.unsqueeze(0).repeat((batch, 1, 1)), prototype_context)
    model._masked_gather_loss = (
        None
        if frozen_mask_prototype
        else model.gather_loss(target_tokens, agg_prototype)
    )

    x = masked_tokens
    for blk in model.bottleneck:
        x = blk(x)

    de_list = []
    for blk in model.decoder:
        x = blk(x, agg_prototype)
        de_list.append(x)
    de_list = de_list[::-1]

    en = [model.fuse_feature([en_list[idx] for idx in idxs]) for idxs in model.fuse_layer_encoder]
    de = [model.fuse_feature([de_list[idx] for idx in idxs]) for idxs in model.fuse_layer_decoder]

    if not model.remove_class_token:
        en = [e[:, original_start:, :] for e in en]
        de = [d[:, original_start:, :] for d in de]

    return _tokens_to_maps(en, batch, side), _tokens_to_maps(de, batch, side), spatial_mask


def _forward_local_context_masked_inpformer(
    model,
    images: torch.Tensor,
    pattern: int,
    config: PerspectiveMaskConfig,
    spatial_mask: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """INP-Former masked reconstruction with a prompted dual-source decoder.

    Masked token features are hidden before the decoder input. At every decoder
    depth, visible positions keep the original INP prototype-guided output,
    while masked positions are produced from nearby visible context tokens at
    that same depth.
    """

    if getattr(model, "_guided_prototype_enabled", False):
        model._guided_prototype_image = images.detach()

    if not hasattr(model, "local_context_reconstructor"):
        attach_local_context_reconstructor(model, radius=config.local_context_radius)

    x = model.encoder.prepare_tokens(images)
    en_list = []
    for i, blk in enumerate(model.encoder.blocks):
        if i <= model.target_layers[-1]:
            if i in model.encoder_require_grad_layer:
                x = blk(x)
            else:
                with torch.no_grad():
                    x = blk(x)
        else:
            continue
        if i in model.target_layers:
            en_list.append(x)
    if not en_list:
        raise RuntimeError("No encoder target features were captured.")

    original_start = 1 + model.encoder.num_register_tokens
    side = int(math.sqrt(en_list[0].shape[1] - original_start))
    if side * side != en_list[0].shape[1] - original_start:
        raise RuntimeError(f"Cannot infer square token grid from shape {en_list[0].shape}.")
    batch = images.shape[0]

    start = original_start
    if model.remove_class_token:
        en_list = [e[:, original_start:, :] for e in en_list]
        start = 0

    target_tokens = model.fuse_feature(en_list)
    spatial_mask = _resolve_spatial_mask(
        spatial_mask,
        model,
        target_tokens,
        start=start,
        batch=batch,
        side=side,
        pattern=pattern,
        config=config,
        device=images.device,
    )
    masked_tokens = _masked_tokens(target_tokens, spatial_mask, start=start, fill=config.fill)
    masked_tokens = _add_candidate_prompt(model, masked_tokens, spatial_mask, start=start, config=config)

    if config.prototype_source == "masked":
        prototype_context = masked_tokens
    elif config.prototype_source == "full":
        prototype_context = target_tokens
    else:
        raise ValueError(f"Unsupported INP prototype source: {config.prototype_source}")

    if hasattr(model, "aggregate_guided_prototypes"):
        agg_prototype = model.aggregate_guided_prototypes(
            target_tokens,
            images,
            prototype_context=prototype_context,
        )
    else:
        agg_prototype = model.prototype_token
        for blk in model.aggregation:
            agg_prototype = blk(agg_prototype.unsqueeze(0).repeat((batch, 1, 1)), prototype_context)
    model._masked_gather_loss = model.gather_loss(target_tokens, agg_prototype)

    x = masked_tokens
    for blk in model.bottleneck:
        x = blk(x)

    proto_de_list = []
    mask_flat = spatial_mask[:, 0].flatten(1).to(dtype=target_tokens.dtype).unsqueeze(-1)
    if start > 0:
        full_mask = torch.cat([torch.zeros_like(target_tokens[:, :start, :1]), mask_flat], dim=1)
    else:
        full_mask = mask_flat

    for layer_index, blk in enumerate(model.decoder):
        prompted = model.local_context_reconstructor.add_prompts(x, spatial_mask, start=start)
        proto_tokens = blk(prompted, agg_prototype)
        context_tokens = model.local_context_reconstructor.context(layer_index, prompted, spatial_mask, start=start)
        x = proto_tokens * (1.0 - full_mask) + context_tokens * full_mask
        proto_de_list.append(x)
    de_list = proto_de_list[::-1]

    en = [model.fuse_feature([en_list[idx] for idx in idxs]) for idxs in model.fuse_layer_encoder]
    de = [model.fuse_feature([de_list[idx] for idx in idxs]) for idxs in model.fuse_layer_decoder]

    if not model.remove_class_token:
        en = [e[:, original_start:, :] for e in en]
        de = [d[:, original_start:, :] for d in de]

    return _tokens_to_maps(en, batch, side), _tokens_to_maps(de, batch, side), spatial_mask


def forward_masked_reconstruction(
    model,
    images: torch.Tensor,
    pattern: int,
    config: PerspectiveMaskConfig,
    spatial_mask: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    if hasattr(model, "forward_masked"):
        if spatial_mask is not None:
            raise ValueError("Custom spatial_mask is not supported by model.forward_masked.")
        return model.forward_masked(images, pattern=pattern, config=config)
    if hasattr(model, "generate_mask") and hasattr(model, "mask_neighbor_size"):
        return _forward_masked_vitill(model, images, pattern, config, spatial_mask=spatial_mask)
    if hasattr(model, "aggregation") and hasattr(model, "prototype_token"):
        if config.inp_mask_mode == "local_context":
            return _forward_local_context_masked_inpformer(model, images, pattern, config, spatial_mask=spatial_mask)
        return _forward_masked_inpformer(model, images, pattern, config, spatial_mask=spatial_mask)
    raise TypeError(f"Unsupported masked reconstruction model type: {type(model)!r}")


def forward_spatial_masked_reconstruction(
    model,
    images: torch.Tensor,
    spatial_mask: torch.Tensor,
    config: PerspectiveMaskConfig,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    return forward_masked_reconstruction(model, images, pattern=0, config=config, spatial_mask=spatial_mask)


def cosine_residuals(en: Sequence[torch.Tensor], de: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    return [
        (1.0 - F.cosine_similarity(target.detach(), pred, dim=1).unsqueeze(1)).clamp_min(0.0)
        for target, pred in zip(en, de)
    ]


def adaptive_mask_planner_loss(
    model,
    en: Sequence[torch.Tensor],
    de: Sequence[torch.Tensor],
    ratio_weight: float = 5.0,
    prior_weight: float = 1.0,
    tv_weight: float = 0.05,
    binary_weight: float = 0.01,
    difficulty_weight: float = 0.2,
) -> torch.Tensor:
    state = getattr(model, "_adaptive_mask_state", None)
    if not state:
        raise RuntimeError("No adaptive mask state found. Run adaptive masked forward before planner loss.")
    logits = state["logits"]
    probs = state["probs"]
    target = state["target"].to(dtype=probs.dtype, device=probs.device)
    mask = state["mask"].to(dtype=probs.dtype, device=probs.device)
    mask_ratio = mask.flatten(1).mean(dim=1).detach()

    prior = F.binary_cross_entropy_with_logits(logits, target)
    ratio = (probs.flatten(1).mean(dim=1) - mask_ratio).pow(2).mean()
    tv = torch.zeros((), dtype=probs.dtype, device=probs.device)
    if probs.shape[-1] > 1:
        tv = tv + (probs[:, :, :, 1:] - probs[:, :, :, :-1]).abs().mean()
    if probs.shape[-2] > 1:
        tv = tv + (probs[:, :, 1:, :] - probs[:, :, :-1, :]).abs().mean()
    binary = (probs * (1.0 - probs)).mean()

    residual = torch.stack(cosine_residuals(en, de), dim=0).mean(dim=0).detach()
    residual = _normalize_map(residual)
    difficulty = -(probs * residual).flatten(1).mean(dim=1).mean()

    return (
        prior_weight * prior
        + ratio_weight * ratio
        + tv_weight * tv
        + binary_weight * binary
        + difficulty_weight * difficulty
    )


def masked_reconstruction_components(
    model,
    images: torch.Tensor,
    patterns: Sequence[int],
    config: PerspectiveMaskConfig,
) -> dict[str, torch.Tensor]:
    accum: dict[str, torch.Tensor] = {}
    denom = None
    for pattern in patterns:
        en, de, mask = forward_masked_reconstruction(model, images, pattern, config)
        residuals = cosine_residuals(en, de)
        components = {
            "base": torch.stack(residuals, dim=0).mean(dim=0),
            "shallow": residuals[0],
            "deep": residuals[-1],
        }
        mask_f = mask.to(components["base"].dtype)
        denom = mask_f if denom is None else denom + mask_f
        for key, value in components.items():
            accum[key] = value * mask_f if key not in accum else accum[key] + value * mask_f
    if denom is None:
        raise ValueError("At least one mask pattern is required.")
    support = denom.clamp(0.0, 1.0)
    denom = denom.clamp_min(1.0)
    out = {key: value / denom for key, value in accum.items()}
    out["disagreement"] = (out["shallow"] - out["deep"]).abs()
    out["mask_support"] = support
    return out
