from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlindSpotContextHead(nn.Module):
    """Predict each target token from its neighbours and adaptive prototypes.

    The token at the predicted location is excluded both from the local
    key/value set and from the neighbourhood descriptor used to select
    prototypes. Inputs are detached so this auxiliary head cannot update the
    Adaptive encoder, prototype aggregation, or reconstruction decoder.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        hidden_dim: int = 128,
        radius: int = 2,
        prototype_temperature: float = 0.10,
    ) -> None:
        super().__init__()
        if embed_dim <= 0 or hidden_dim <= 0:
            raise ValueError("embed_dim and hidden_dim must be positive.")
        if radius < 1:
            raise ValueError("Blind-spot radius must be at least one.")
        if prototype_temperature <= 0:
            raise ValueError("Prototype temperature must be positive.")
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.radius = int(radius)
        self.prototype_temperature = float(prototype_temperature)
        self.token_norm = nn.LayerNorm(embed_dim)
        self.prototype_norm = nn.LayerNorm(embed_dim)
        self.query = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.key = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.value = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.context_out = nn.Linear(hidden_dim, embed_dim)
        self.prototype_out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.output_norm = nn.LayerNorm(embed_dim)

    def _neighbour_mean(self, tokens: torch.Tensor, side: int) -> torch.Tensor:
        batch, token_count, channels = tokens.shape
        if token_count != side * side:
            raise ValueError(
                f"Token grid is not side x side: tokens={token_count}, side={side}."
            )
        kernel = 2 * self.radius + 1
        token_map = tokens.transpose(1, 2).reshape(batch, channels, side, side)
        neighbour_kernel = torch.ones(
            (channels, 1, kernel, kernel),
            dtype=tokens.dtype,
            device=tokens.device,
        )
        neighbour_kernel[:, :, self.radius, self.radius] = 0
        summed = F.conv2d(
            token_map,
            neighbour_kernel,
            stride=1,
            padding=self.radius,
            groups=channels,
        )
        valid = torch.ones(
            (batch, 1, side, side), dtype=tokens.dtype, device=tokens.device
        )
        count_kernel = neighbour_kernel[:1]
        count = F.conv2d(
            valid,
            count_kernel,
            stride=1,
            padding=self.radius,
        ).clamp_min(1.0)
        return (summed / count).flatten(2).transpose(1, 2)

    def forward(
        self,
        target_tokens: torch.Tensor,
        adaptive_prototypes: torch.Tensor,
        *,
        side: int,
    ) -> torch.Tensor:
        tokens = target_tokens.detach()
        prototypes = adaptive_prototypes.detach()
        if tokens.ndim != 3 or prototypes.ndim != 3:
            raise ValueError("Tokens and prototypes must have shape [B, N/K, C].")
        if tokens.shape[0] != prototypes.shape[0] or tokens.shape[2] != self.embed_dim:
            raise ValueError("Blind-spot token/prototype shapes are incompatible.")

        neighbour_mean = self._neighbour_mean(tokens, side)
        neighbour_unit = F.normalize(neighbour_mean.float(), dim=-1)
        prototype_unit = F.normalize(prototypes.float(), dim=-1)
        assignment = torch.softmax(
            torch.einsum("bnc,bkc->bnk", neighbour_unit, prototype_unit)
            / self.prototype_temperature,
            dim=-1,
        ).to(dtype=tokens.dtype)
        prototype_context = torch.einsum("bnk,bkc->bnc", assignment, prototypes)

        norm_tokens = self.token_norm(tokens)
        query = self.query(self.prototype_norm(prototype_context))
        key_map = self.key(norm_tokens).transpose(1, 2).reshape(
            tokens.shape[0], self.hidden_dim, side, side
        )
        value_map = self.value(norm_tokens).transpose(1, 2).reshape_as(key_map)
        kernel = 2 * self.radius + 1
        neighbourhood = kernel * kernel
        keys = F.unfold(key_map, kernel_size=kernel, padding=self.radius)
        values = F.unfold(value_map, kernel_size=kernel, padding=self.radius)
        keys = keys.reshape(
            tokens.shape[0], self.hidden_dim, neighbourhood, side * side
        ).permute(0, 3, 2, 1)
        values = values.reshape_as(
            keys.permute(0, 3, 2, 1)
        ).permute(0, 3, 2, 1)

        valid_map = torch.ones(
            (tokens.shape[0], 1, side, side),
            dtype=tokens.dtype,
            device=tokens.device,
        )
        valid = F.unfold(valid_map, kernel_size=kernel, padding=self.radius)
        valid = valid.transpose(1, 2).bool()
        valid[:, :, neighbourhood // 2] = False
        logits = torch.einsum("bnh,bnkh->bnk", query, keys) / math.sqrt(
            self.hidden_dim
        )
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        local_context = torch.einsum("bnk,bnkh->bnh", weights, values)
        prediction = self.context_out(local_context) + self.prototype_out(
            prototype_context
        )
        return self.output_norm(prediction)


def _infer_embed_dim(model: nn.Module) -> int:
    prototype_token = getattr(model, "prototype_token", None)
    if isinstance(prototype_token, nn.Parameter):
        return int(prototype_token.shape[-1])
    if isinstance(prototype_token, (nn.ParameterList, nn.ModuleList)) and len(prototype_token):
        return int(prototype_token[0].shape[-1])
    encoder = getattr(model, "encoder", None)
    embed_dim = getattr(encoder, "embed_dim", None)
    if embed_dim is None:
        raise ValueError("Could not infer INP-Former embedding dimension.")
    return int(embed_dim)


def attach_blindspot_context_head(
    model: nn.Module,
    *,
    hidden_dim: int = 128,
    radius: int = 2,
    prototype_temperature: float = 0.10,
) -> BlindSpotContextHead:
    head = BlindSpotContextHead(
        _infer_embed_dim(model),
        hidden_dim=hidden_dim,
        radius=radius,
        prototype_temperature=prototype_temperature,
    )
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    head = head.to(device)
    model.add_module("blindspot_context_head", head)
    return head


def blindspot_context_score(
    head: BlindSpotContextHead,
    target_tokens: torch.Tensor,
    adaptive_prototypes: torch.Tensor,
    *,
    side: int,
) -> torch.Tensor:
    prediction = head(target_tokens, adaptive_prototypes, side=side)
    score = 1.0 - F.cosine_similarity(
        prediction.float(), target_tokens.detach().float(), dim=-1
    )
    return score.clamp_min(0.0).reshape(target_tokens.shape[0], 1, side, side)


def fuse_blindspot_scores(
    adaptive_score: torch.Tensor,
    context_score: torch.Tensor,
    *,
    mode: str,
    adaptive_q99: float,
    context_q99: float,
    boost_alpha: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    if adaptive_score.shape != context_score.shape:
        raise ValueError("Adaptive and context score maps must have identical shapes.")
    if adaptive_q99 <= 0 or context_q99 <= 0:
        raise ValueError("Normal Q99 calibration values must be positive.")
    if mode == "boost":
        return adaptive_score + float(boost_alpha) * F.relu(
            context_score - float(context_q99)
        )
    if mode == "agreement":
        adaptive_norm = adaptive_score / max(float(adaptive_q99), eps)
        context_norm = context_score / max(float(context_q99), eps)
        return torch.minimum(adaptive_norm, context_norm) * float(adaptive_q99)
    raise ValueError(f"Unsupported blind-spot fusion mode: {mode}")


def load_blindspot_calibration(path: str | Path) -> Mapping[str, float]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    adaptive_q99 = float(payload["adaptive"]["q99"])
    context_q99 = float(payload["context"]["q99"])
    if adaptive_q99 <= 0 or context_q99 <= 0:
        raise ValueError(f"Invalid blind-spot calibration in {path}.")
    return {"adaptive_q99": adaptive_q99, "context_q99": context_q99}


def add_blindspot_context_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--blindspot-context", action="store_true")
    parser.add_argument("--blindspot-context-hidden-dim", type=int, default=128)
    parser.add_argument("--blindspot-context-radius", type=int, default=2)
    parser.add_argument(
        "--blindspot-context-prototype-temperature", type=float, default=0.10
    )
    parser.add_argument(
        "--blindspot-fusion", choices=["boost", "agreement"], default="boost"
    )
    parser.add_argument("--blindspot-normal-calibration", type=Path, default=None)
    parser.add_argument("--blindspot-boost-alpha", type=float, default=0.5)
