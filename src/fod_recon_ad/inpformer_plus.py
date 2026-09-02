from __future__ import annotations

import math
import types
from dataclasses import asdict, dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_inp_coherence_loss(
    query: torch.Tensor,
    prototypes: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Paper-equivalent soft INP coherence loss from INP-Former++ Eq. (3)."""

    if query.ndim != 3 or prototypes.ndim != 3:
        raise ValueError("query and prototypes must have shape [B, N, C] and [B, M, C].")
    if query.shape[0] != prototypes.shape[0] or query.shape[2] != prototypes.shape[2]:
        raise ValueError(f"Incompatible query/prototype shapes: {query.shape}, {prototypes.shape}.")

    query_normalized = F.normalize(query, dim=-1, eps=eps)
    prototype_normalized = F.normalize(prototypes, dim=-1, eps=eps)
    similarities = torch.matmul(query_normalized, prototype_normalized.transpose(1, 2))
    assignment = similarities.softmax(dim=-1)
    reconstructed = torch.matmul(assignment, prototypes)
    return (1.0 - F.cosine_similarity(query.flatten(1), reconstructed.flatten(1), dim=1, eps=eps)).mean()


def configure_inp_coherence(model: nn.Module, mode: str) -> None:
    """Switch the INP gather objective without changing model parameters or checkpoints."""

    if mode not in {"hard", "soft"}:
        raise ValueError(f"Unsupported INP coherence mode: {mode!r}.")
    if not hasattr(model, "gather_loss"):
        raise ValueError("The selected model does not expose INP-Former's gather_loss method.")
    if not hasattr(model, "_inp_original_gather_loss"):
        model._inp_original_gather_loss = model.gather_loss
    if mode == "hard":
        model.gather_loss = model._inp_original_gather_loss
    else:
        def _soft_gather_loss(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
            return soft_inp_coherence_loss(query, keys)

        model.gather_loss = types.MethodType(_soft_gather_loss, model)
    model._inp_coherence_mode = mode


def inpformer_plus_residual(
    encoder_features: Iterable[torch.Tensor],
    decoder_features: Iterable[torch.Tensor],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Build the channel-wise residual used by INP-Former++ Eq. (7)."""

    residuals: list[torch.Tensor] = []
    reference_size: tuple[int, int] | None = None
    for target, prediction in zip(encoder_features, decoder_features):
        target = target.detach()
        if target.shape != prediction.shape:
            raise ValueError(f"Encoder/decoder feature shapes differ: {target.shape}, {prediction.shape}.")
        cosine_distance = (1.0 - F.cosine_similarity(target, prediction, dim=1, eps=eps)).clamp_min(0.0)
        residual = cosine_distance.unsqueeze(1) * (target - prediction).abs()
        if reference_size is None:
            reference_size = residual.shape[-2:]
        elif residual.shape[-2:] != reference_size:
            residual = F.interpolate(residual, size=reference_size, mode="bilinear", align_corners=False)
        residuals.append(residual)
    if not residuals:
        raise ValueError("No encoder/decoder feature pairs were provided.")
    return torch.stack(residuals, dim=0).mean(dim=0)


def inpformer_plus_reconstruction_map(
    encoder_features: Iterable[torch.Tensor],
    decoder_features: Iterable[torch.Tensor],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Cosine plus feature-magnitude reconstruction map from INP-Former++ Eq. (9)."""

    maps: list[torch.Tensor] = []
    reference_size: tuple[int, int] | None = None
    for target, prediction in zip(encoder_features, decoder_features):
        target = target.detach()
        cosine = (1.0 - F.cosine_similarity(target, prediction, dim=1, eps=eps)).clamp_min(0.0)
        magnitude = torch.linalg.vector_norm(target - prediction, ord=2, dim=1)
        score = 0.5 * (cosine + magnitude)
        if reference_size is None:
            reference_size = score.shape[-2:]
        elif score.shape[-2:] != reference_size:
            score = F.interpolate(score.unsqueeze(1), size=reference_size, mode="bilinear", align_corners=False)[:, 0]
        maps.append(score.unsqueeze(1))
    if not maps:
        raise ValueError("No encoder/decoder feature pairs were provided.")
    return torch.stack(maps, dim=0).mean(dim=0)


@dataclass(frozen=True)
class ResidualFlowConfig:
    input_channels: int
    projection_dim: int = 64
    coupling_layers: int = 6
    hidden_dim: int = 128
    scale_limit: float = 1.5
    projection_seed: int = 17

    @classmethod
    def from_dict(cls, payload: dict) -> "ResidualFlowConfig":
        return cls(**{key: payload[key] for key in cls.__dataclass_fields__ if key in payload})

    def to_dict(self) -> dict:
        return asdict(self)


class AffineCoupling(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, mask: torch.Tensor, scale_limit: float) -> None:
        super().__init__()
        self.register_buffer("mask", mask.reshape(1, dim))
        self.scale_limit = float(scale_limit)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fixed = inputs * self.mask
        log_scale, shift = self.net(fixed).chunk(2, dim=-1)
        transformed_mask = 1.0 - self.mask
        log_scale = self.scale_limit * torch.tanh(log_scale) * transformed_mask
        shift = shift * transformed_mask
        outputs = fixed + transformed_mask * (inputs * torch.exp(log_scale) + shift)
        return outputs, log_scale.sum(dim=-1)


class ResidualFlow(nn.Module):
    """Normal-only density estimator over fixed projections of INP residual tokens."""

    def __init__(self, config: ResidualFlowConfig) -> None:
        super().__init__()
        if config.projection_dim < 2:
            raise ValueError("projection_dim must be at least 2.")
        if config.projection_dim > config.input_channels:
            raise ValueError("projection_dim cannot exceed input_channels.")
        self.config = config

        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.projection_seed)
        random_matrix = torch.randn(config.input_channels, config.projection_dim, generator=generator)
        projection, _ = torch.linalg.qr(random_matrix, mode="reduced")
        self.register_buffer("projection", projection.transpose(0, 1).contiguous())
        self.register_buffer("feature_mean", torch.zeros(config.projection_dim))
        self.register_buffer("feature_std", torch.ones(config.projection_dim))
        self.register_buffer("score_mean", torch.tensor(0.0))
        self.register_buffer("score_std", torch.tensor(1.0))

        couplings = []
        indices = torch.arange(config.projection_dim)
        for layer_index in range(config.coupling_layers):
            mask = ((indices + layer_index) % 2 == 0).float()
            couplings.append(AffineCoupling(config.projection_dim, config.hidden_dim, mask, config.scale_limit))
        self.couplings = nn.ModuleList(couplings)

    def raw_project_map(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.ndim != 4 or residual.shape[1] != self.config.input_channels:
            raise ValueError(
                f"Expected residual [B, {self.config.input_channels}, H, W], got {tuple(residual.shape)}."
            )
        return torch.einsum("dc,bchw->bdhw", self.projection.to(residual.dtype), residual)

    def project_map(self, residual: torch.Tensor) -> torch.Tensor:
        projected = self.raw_project_map(residual)
        mean = self.feature_mean.to(projected.dtype).view(1, -1, 1, 1)
        std = self.feature_std.to(projected.dtype).view(1, -1, 1, 1)
        return (projected - mean) / std.clamp_min(1e-6)

    def nll_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        latent = tokens
        log_determinant = torch.zeros(tokens.shape[0], device=tokens.device, dtype=tokens.dtype)
        for coupling in self.couplings:
            latent, layer_logdet = coupling(latent)
            log_determinant = log_determinant + layer_logdet
        base_nll = 0.5 * (latent.square() + math.log(2.0 * math.pi)).sum(dim=-1)
        return (base_nll - log_determinant) / float(self.config.projection_dim)

    def nll_map(self, residual: torch.Tensor) -> torch.Tensor:
        projected = self.project_map(residual)
        batch, channels, height, width = projected.shape
        tokens = projected.permute(0, 2, 3, 1).reshape(-1, channels)
        return self.nll_tokens(tokens).reshape(batch, 1, height, width)

    def anomaly_map(
        self,
        residual: torch.Tensor,
        temperature: float = 1.0,
        normal_offset: float = 3.0,
    ) -> torch.Tensor:
        standardized = (self.nll_map(residual) - self.score_mean) / self.score_std.clamp_min(1e-6)
        return torch.sigmoid(
            (standardized - float(normal_offset)) / max(float(temperature), 1e-6)
        )

    @torch.no_grad()
    def set_feature_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feature_mean.copy_(mean.to(self.feature_mean))
        self.feature_std.copy_(std.to(self.feature_std).clamp_min(1e-6))

    @torch.no_grad()
    def set_score_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.score_mean.copy_(mean.to(self.score_mean))
        self.score_std.copy_(std.to(self.score_std).clamp_min(1e-6))


def residual_flow_payload(flow: ResidualFlow) -> dict:
    return {"config": flow.config.to_dict(), "state_dict": flow.state_dict()}


def residual_flow_from_payload(payload: dict, device: torch.device) -> ResidualFlow:
    if "config" not in payload or "state_dict" not in payload:
        raise ValueError("Residual-flow payload must contain config and state_dict.")
    flow = ResidualFlow(ResidualFlowConfig.from_dict(payload["config"])).to(device)
    flow.load_state_dict(payload["state_dict"], strict=True)
    return flow
