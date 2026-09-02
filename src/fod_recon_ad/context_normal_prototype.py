from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ContextNormalPrototypeConfig:
    mix: float = 0.5
    memory_size: int = 2048
    candidates_per_image: int = 8
    candidates_per_group: int = 32
    context_radius: int = 1
    topk: int = 5
    temperature: float = 0.07
    query_chunk_size: int = 1024
    memory_build_batches: int = 0
    freeze_decoder: bool = False


def add_context_normal_prototype_args(parser) -> None:
    parser.add_argument(
        "--context-normal-prototype",
        action="store_true",
        help=(
            "Build INP prototypes from a blend of current tokens and normal-memory values "
            "retrieved by local context. Decoder inputs remain unchanged."
        ),
    )
    parser.add_argument(
        "--context-normal-prototype-mix",
        type=float,
        default=0.5,
        help="Blend factor: 0 is Native prototype aggregation; 1 uses retrieved normal values.",
    )
    parser.add_argument("--context-normal-memory-size", type=int, default=2048)
    parser.add_argument("--context-normal-candidates-per-image", type=int, default=8)
    parser.add_argument("--context-normal-candidates-per-group", type=int, default=32)
    parser.add_argument("--context-normal-context-radius", type=int, default=1)
    parser.add_argument("--context-normal-topk", type=int, default=5)
    parser.add_argument("--context-normal-temperature", type=float, default=0.07)
    parser.add_argument("--context-normal-query-chunk-size", type=int, default=1024)
    parser.add_argument(
        "--context-normal-memory-build-batches",
        type=int,
        default=0,
        help="Normal batches used to build the frozen memory; 0 scans one full loader epoch.",
    )
    parser.add_argument(
        "--context-normal-freeze-decoder",
        action="store_true",
        help="Freeze bottleneck and decoder during the warm-start prototype adaptation.",
    )


def context_normal_config_from_args(args) -> ContextNormalPrototypeConfig:
    config = ContextNormalPrototypeConfig(
        mix=float(args.context_normal_prototype_mix),
        memory_size=int(args.context_normal_memory_size),
        candidates_per_image=int(args.context_normal_candidates_per_image),
        candidates_per_group=int(args.context_normal_candidates_per_group),
        context_radius=int(args.context_normal_context_radius),
        topk=int(args.context_normal_topk),
        temperature=float(args.context_normal_temperature),
        query_chunk_size=int(args.context_normal_query_chunk_size),
        memory_build_batches=int(args.context_normal_memory_build_batches),
        freeze_decoder=bool(args.context_normal_freeze_decoder),
    )
    if not 0.0 <= config.mix <= 1.0:
        raise ValueError("--context-normal-prototype-mix must be in [0, 1].")
    if config.memory_size <= 0:
        raise ValueError("--context-normal-memory-size must be positive.")
    if config.candidates_per_image <= 0 or config.candidates_per_group <= 0:
        raise ValueError("Context-normal candidate counts must be positive.")
    if config.context_radius <= 0:
        raise ValueError("--context-normal-context-radius must be positive.")
    if config.topk <= 0 or config.topk > config.memory_size:
        raise ValueError("--context-normal-topk must be positive and no larger than memory size.")
    if config.temperature <= 0.0:
        raise ValueError("--context-normal-temperature must be positive.")
    if config.query_chunk_size <= 0:
        raise ValueError("--context-normal-query-chunk-size must be positive.")
    if config.memory_build_batches < 0:
        raise ValueError("--context-normal-memory-build-batches cannot be negative.")
    return config


def context_ring_descriptor(tokens: torch.Tensor, radius: int) -> torch.Tensor:
    """Describe each token by its local neighbors, explicitly excluding the center token."""

    if tokens.ndim != 3:
        raise ValueError(f"Expected [B,N,C] tokens, got {tuple(tokens.shape)}.")
    batch, token_count, channels = tokens.shape
    side = int(math.isqrt(token_count))
    if side * side != token_count:
        raise ValueError(f"Context memory expects a square token grid, got N={token_count}.")
    if radius <= 0:
        raise ValueError(f"Context radius must be positive, got {radius}.")

    feature = tokens.transpose(1, 2).reshape(batch, channels, side, side)
    kernel_side = 2 * radius + 1
    kernel = feature.new_ones(1, 1, kernel_side, kernel_side)
    kernel[..., radius, radius] = 0.0
    channel_kernel = kernel.expand(channels, 1, -1, -1)
    context_sum = F.conv2d(feature, channel_kernel, padding=radius, groups=channels)
    counts = F.conv2d(
        feature.new_ones(batch, 1, side, side),
        kernel,
        padding=radius,
    )
    context = context_sum / counts.clamp_min(1.0)
    return context.flatten(2).transpose(1, 2).contiguous()


def blend_prototype_context(
    target_tokens: torch.Tensor,
    normal_tokens: torch.Tensor,
    mix: float,
) -> torch.Tensor:
    if target_tokens.shape != normal_tokens.shape:
        raise ValueError(
            f"Target and normal token shapes differ: {tuple(target_tokens.shape)} vs "
            f"{tuple(normal_tokens.shape)}."
        )
    return torch.lerp(target_tokens, normal_tokens, float(mix))


class ContextKeyValueMemory(nn.Module):
    """Frozen normal memory: neighbor context is the key and its center token is the value."""

    def __init__(
        self,
        dim: int,
        size: int,
        topk: int,
        temperature: float,
        query_chunk_size: int,
    ) -> None:
        super().__init__()
        self.size = int(size)
        self.topk = int(topk)
        self.temperature = float(temperature)
        self.query_chunk_size = int(query_chunk_size)
        self.register_buffer("keys", torch.zeros(self.size, dim), persistent=True)
        self.register_buffer("values", torch.zeros(self.size, dim), persistent=True)
        self.register_buffer("group_ids", torch.full((self.size,), -1, dtype=torch.long), persistent=True)
        self.register_buffer("count", torch.zeros((), dtype=torch.long), persistent=True)

    @property
    def ready(self) -> bool:
        return int(self.count.item()) > 0

    @torch.no_grad()
    def set_entries(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> None:
        if keys.ndim != 2 or values.shape != keys.shape:
            raise ValueError("Memory keys and values must have the same [M,C] shape.")
        if group_ids.shape != (keys.shape[0],):
            raise ValueError("Memory group_ids must have shape [M].")
        take = min(self.size, keys.shape[0])
        self.keys.zero_()
        self.values.zero_()
        self.group_ids.fill_(-1)
        self.keys[:take].copy_(F.normalize(keys[:take].detach().float(), dim=-1).to(self.keys))
        self.values[:take].copy_(values[:take].detach().float().to(self.values))
        self.group_ids[:take].copy_(group_ids[:take].detach().long().to(self.group_ids))
        self.count.fill_(take)

    def retrieve(
        self,
        query_context: torch.Tensor,
        source_group_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        count = int(self.count.item())
        if count == 0:
            raise RuntimeError("Context-normal memory is empty. Build or load it before forward().")
        if source_group_ids is not None and source_group_ids.shape != (query_context.shape[0],):
            raise ValueError(
                f"Expected source_group_ids shape {(query_context.shape[0],)}, "
                f"got {tuple(source_group_ids.shape)}."
            )

        keys = self.keys[:count]
        values = self.values[:count]
        memory_groups = self.group_ids[:count]
        query = F.normalize(query_context.float(), dim=-1)
        batch, token_count, channels = query.shape
        flat_query = query.reshape(-1, channels)
        flat_batch = torch.arange(batch, device=query.device).repeat_interleave(token_count)
        outputs = []
        best_similarities = []
        excluded_rows = 0
        for start in range(0, flat_query.shape[0], self.query_chunk_size):
            stop = min(start + self.query_chunk_size, flat_query.shape[0])
            similarity = flat_query[start:stop] @ keys.T
            if source_group_ids is not None:
                query_groups = source_group_ids[flat_batch[start:stop]].to(memory_groups.device)
                valid = memory_groups.unsqueeze(0) != query_groups.unsqueeze(1)
                valid = valid & (memory_groups.unsqueeze(0) >= 0)
                has_valid = valid.any(dim=1)
                excluded_rows += int((~has_valid).sum().item())
                similarity = torch.where(has_valid.unsqueeze(1), similarity.masked_fill(~valid, -1e4), similarity)
            k = min(self.topk, count)
            top_similarity, top_indices = similarity.topk(k, dim=1)
            weights = F.softmax(top_similarity / self.temperature, dim=1)
            selected_values = values[top_indices]
            outputs.append((weights.unsqueeze(-1) * selected_values).sum(dim=1))
            best_similarities.append(top_similarity[:, 0])

        normal = torch.cat(outputs, dim=0).reshape(batch, token_count, channels)
        best = torch.cat(best_similarities, dim=0)
        diagnostics = {
            "context_normal_memory_count": float(count),
            "context_normal_similarity": float(best.mean().detach().cpu()),
            "context_normal_unexcluded_ratio": float(excluded_rows / max(1, flat_query.shape[0])),
        }
        return normal.to(query_context.dtype), diagnostics


def extract_inpformer_tokens(
    model: nn.Module,
    images: torch.Tensor,
) -> tuple[list[torch.Tensor], torch.Tensor, int, int]:
    """Run the frozen encoder and return captured features plus fused INP target tokens."""

    x = model.encoder.prepare_tokens(images)
    encoder_features = []
    for index, block in enumerate(model.encoder.blocks):
        if index > model.target_layers[-1]:
            continue
        if index in model.encoder_require_grad_layer:
            x = block(x)
        else:
            with torch.no_grad():
                x = block(x)
        if index in model.target_layers:
            encoder_features.append(x)
    if not encoder_features:
        raise RuntimeError("No INP-Former encoder features were captured.")

    token_start = 1 + model.encoder.num_register_tokens
    side = int(math.isqrt(encoder_features[0].shape[1] - token_start))
    if model.remove_class_token:
        encoder_features = [feature[:, token_start:, :] for feature in encoder_features]
    target_tokens = model.fuse_feature(encoder_features)
    return encoder_features, target_tokens, side, token_start


def _context_normal_inpformer_forward(self, images: torch.Tensor) -> tuple:
    encoder_features, target_tokens, side, token_start = extract_inpformer_tokens(self, images)
    config: ContextNormalPrototypeConfig = self._context_normal_config
    if config.mix == 0.0:
        normal_tokens = target_tokens
        memory_diag = {
            "context_normal_memory_count": float(self.context_normal_memory.count.item()),
            "context_normal_similarity": 0.0,
            "context_normal_unexcluded_ratio": 0.0,
        }
    else:
        context = context_ring_descriptor(target_tokens, config.context_radius)
        normal_tokens, memory_diag = self.context_normal_memory.retrieve(
            context,
            getattr(self, "_context_normal_source_group_ids", None),
        )
    prototype_tokens = blend_prototype_context(target_tokens, normal_tokens, config.mix)

    batch = images.shape[0]
    prototype = self.prototype_token.unsqueeze(0).repeat(batch, 1, 1)
    for block in self.aggregation:
        prototype = block(prototype, prototype_tokens)
    gather_loss = self.gather_loss(target_tokens, prototype)

    x = target_tokens
    for block in self.bottleneck:
        x = block(x)
    decoder_features = []
    for block in self.decoder:
        x = block(x, prototype)
        decoder_features.append(x)
    decoder_features = decoder_features[::-1]

    encoder_output = [
        self.fuse_feature([encoder_features[index] for index in indices])
        for indices in self.fuse_layer_encoder
    ]
    decoder_output = [
        self.fuse_feature([decoder_features[index] for index in indices])
        for indices in self.fuse_layer_decoder
    ]
    if not self.remove_class_token:
        encoder_output = [feature[:, token_start:, :] for feature in encoder_output]
        decoder_output = [feature[:, token_start:, :] for feature in decoder_output]
    encoder_maps = [
        feature.permute(0, 2, 1).reshape(batch, -1, side, side).contiguous()
        for feature in encoder_output
    ]
    decoder_maps = [
        feature.permute(0, 2, 1).reshape(batch, -1, side, side).contiguous()
        for feature in decoder_output
    ]
    shift = 1.0 - F.cosine_similarity(target_tokens.float(), normal_tokens.float(), dim=-1)
    self._context_normal_diag = {
        **memory_diag,
        "context_normal_mix": float(config.mix),
        "context_normal_shift": float(shift.mean().detach().cpu()),
    }
    return encoder_maps, decoder_maps, gather_loss


def configure_context_normal_prototype(model: nn.Module, args, architecture: str) -> None:
    if not getattr(args, "context_normal_prototype", False):
        return
    if architecture != "inpformer":
        raise ValueError("--context-normal-prototype is only implemented for --architecture inpformer.")
    if getattr(args, "guided_prototype", False):
        raise ValueError("Context-normal and Guided Prototype alter the same aggregation path; enable only one.")
    config = context_normal_config_from_args(args)
    prototype_token = getattr(model, "prototype_token", None)
    if prototype_token is None:
        raise ValueError("INP-Former model does not expose prototype_token.")
    if hasattr(model, "context_normal_memory"):
        raise RuntimeError("Context-normal prototype has already been configured on this model.")

    model.context_normal_memory = ContextKeyValueMemory(
        dim=int(prototype_token.shape[-1]),
        size=config.memory_size,
        topk=config.topk,
        temperature=config.temperature,
        query_chunk_size=config.query_chunk_size,
    ).to(prototype_token.device)
    model._context_normal_config = config
    model._context_normal_enabled = True
    model._context_normal_diag = {}
    model._context_normal_source_group_ids = None
    model._context_normal_original_forward = model.forward
    model.forward = types.MethodType(_context_normal_inpformer_forward, model)

    if config.freeze_decoder:
        for module_name in ("bottleneck", "decoder"):
            for parameter in getattr(model, module_name).parameters():
                parameter.requires_grad_(False)


def set_context_normal_source_groups(
    model: nn.Module,
    source_group_ids: torch.Tensor | None,
) -> None:
    if getattr(model, "_context_normal_enabled", False):
        model._context_normal_source_group_ids = (
            None if source_group_ids is None else source_group_ids.detach()
        )


def get_context_normal_diag(model: nn.Module) -> dict[str, float]:
    return dict(getattr(model, "_context_normal_diag", {}) or {})


def context_normal_memory_ready(model: nn.Module) -> bool:
    memory = getattr(model, "context_normal_memory", None)
    return isinstance(memory, ContextKeyValueMemory) and memory.ready


def _balanced_round_robin(
    buckets: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
    limit: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected_keys = []
    selected_values = []
    selected_groups = []
    offsets = {group_id: 0 for group_id in buckets}
    active = sorted(buckets)
    while active and len(selected_keys) < limit:
        next_active = []
        for group_id in active:
            offset = offsets[group_id]
            values = buckets[group_id]
            if offset < len(values) and len(selected_keys) < limit:
                key, value = values[offset]
                selected_keys.append(key)
                selected_values.append(value)
                selected_groups.append(group_id)
                offsets[group_id] = offset + 1
            if offsets[group_id] < len(values):
                next_active.append(group_id)
        active = next_active
    if not selected_keys:
        raise RuntimeError("No normal token candidates were collected for context memory.")
    return (
        torch.stack(selected_keys),
        torch.stack(selected_values),
        torch.tensor(selected_groups, dtype=torch.long),
    )


@torch.no_grad()
def fit_context_normal_memory(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    source_group_resolver,
) -> dict[str, float]:
    """Build a frozen, source-balanced memory from one pass over normal training images."""

    if not getattr(model, "_context_normal_enabled", False):
        return {}
    config: ContextNormalPrototypeConfig = model._context_normal_config
    memory: ContextKeyValueMemory = model.context_normal_memory
    if memory.ready:
        return {
            "context_normal_memory_count": float(memory.count.item()),
            "context_normal_memory_groups": float(torch.unique(memory.group_ids[: memory.count]).numel()),
        }

    was_training = model.training
    model.eval()
    buckets: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    generator = torch.Generator(device="cpu").manual_seed(20260715)
    batches = 0
    for images, _, paths in loader:
        images = images.to(device, non_blocking=True)
        _, target_tokens, _, _ = extract_inpformer_tokens(model, images)
        contexts = context_ring_descriptor(target_tokens, config.context_radius)
        group_ids = source_group_resolver.ids(paths).tolist()
        token_count = target_tokens.shape[1]
        take = min(config.candidates_per_image, token_count)
        for image_index, group_id in enumerate(group_ids):
            bucket = buckets.setdefault(int(group_id), [])
            remaining = config.candidates_per_group - len(bucket)
            if remaining <= 0:
                continue
            indices = torch.randperm(token_count, generator=generator)[: min(take, remaining)]
            keys = contexts[image_index, indices.to(contexts.device)].detach().float().cpu()
            values = target_tokens[image_index, indices.to(target_tokens.device)].detach().float().cpu()
            bucket.extend(zip(keys.unbind(0), values.unbind(0)))
        batches += 1
        if config.memory_build_batches > 0 and batches >= config.memory_build_batches:
            break

    keys, values, group_ids = _balanced_round_robin(buckets, config.memory_size)
    memory.set_entries(keys.to(device), values.to(device), group_ids.to(device))
    if was_training:
        model.train()
    return {
        "context_normal_memory_count": float(memory.count.item()),
        "context_normal_memory_groups": float(torch.unique(memory.group_ids[: memory.count]).numel()),
        "context_normal_memory_batches": float(batches),
    }
