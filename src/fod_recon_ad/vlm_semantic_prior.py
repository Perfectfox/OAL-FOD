from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


VLM_PRIOR_CLASSES = ("flat_background", "texture", "regular_pattern")
TOKEN_PRIOR_ARCHITECTURE = "token_semantic_v1"
PIXEL_PRIOR_ARCHITECTURE = "pixel_hierarchical_v1"


class VLMSemanticPriorHead(nn.Module):
    """Small spatial head distilled from VLM semantic region labels.

    The frozen reconstruction encoder supplies one feature vector per token.  A
    low-rank projection plus a depth-wise 3x3 block adds only local context; the
    head never changes the reconstruction encoder.
    """

    def __init__(self, dim: int, hidden_dim: int = 96) -> None:
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.norm = nn.LayerNorm(self.dim)
        self.project = nn.Linear(self.dim, self.hidden_dim)
        self.spatial = nn.Sequential(
            nn.Conv2d(
                self.hidden_dim,
                self.hidden_dim,
                kernel_size=3,
                padding=1,
                groups=self.hidden_dim,
            ),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(self.hidden_dim, len(VLM_PRIOR_CLASSES), kernel_size=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, count, dim = tokens.shape
        if dim != self.dim:
            raise ValueError(f"Expected token dim {self.dim}, got {dim}.")
        side = int(count**0.5)
        if side * side != count:
            raise ValueError(f"VLM semantic prior requires a square token grid, got {count} tokens.")
        features = self.project(self.norm(tokens.float()))
        features = features.transpose(1, 2).reshape(batch, self.hidden_dim, side, side)
        logits = self.classifier(self.spatial(features))
        return logits.flatten(2).transpose(1, 2).contiguous()


class _PixelConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        groups = max(1, min(8, out_channels // 4))
        while out_channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                groups=out_channels,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block(values)


class PixelHierarchicalPriorHead(nn.Module):
    """Small RGB student with independent object and conditional-texture heads.

    The hierarchy avoids a three-way softmax collapse when object-like pixels are
    sparse.  It predicts ``p(object)`` and ``p(texture | non-object)``; the three
    semantic probabilities are composed by :func:`predict_pixel_prior`.
    """

    def __init__(self, width: int = 24) -> None:
        super().__init__()
        self.width = int(width)
        self.stem = _PixelConvBlock(3, self.width, stride=2)
        self.encoder_2 = _PixelConvBlock(self.width, self.width * 2, stride=2)
        self.encoder_3 = _PixelConvBlock(self.width * 2, self.width * 3, stride=2)
        self.lateral_3 = nn.Conv2d(self.width * 3, self.width * 2, kernel_size=1)
        self.fuse_2 = _PixelConvBlock(self.width * 4, self.width * 2)
        self.lateral_2 = nn.Conv2d(self.width * 2, self.width, kernel_size=1)
        self.fuse_1 = _PixelConvBlock(self.width * 2, self.width)
        self.object_head = nn.Conv2d(self.width, 1, kernel_size=1)
        self.texture_head = nn.Conv2d(self.width, 1, kernel_size=1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_size = images.shape[-2:]
        level_1 = self.stem(images.float())
        level_2 = self.encoder_2(level_1)
        level_3 = self.encoder_3(level_2)
        decoded_2 = F.interpolate(
            self.lateral_3(level_3), size=level_2.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded_2 = self.fuse_2(torch.cat([decoded_2, level_2], dim=1))
        decoded_1 = F.interpolate(
            self.lateral_2(decoded_2), size=level_1.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded_1 = self.fuse_1(torch.cat([decoded_1, level_1], dim=1))
        object_logit = F.interpolate(
            self.object_head(decoded_1), size=input_size, mode="bilinear", align_corners=False
        )
        texture_logit = F.interpolate(
            self.texture_head(decoded_1), size=input_size, mode="bilinear", align_corners=False
        )
        return object_logit, texture_logit


@dataclass(frozen=True)
class VLMPriorCheckpoint:
    head: nn.Module
    temperature: float
    support_centers: torch.Tensor | None
    support_radii: torch.Tensor | None
    metadata: dict
    architecture: str = TOKEN_PRIOR_ARCHITECTURE
    object_pool_mean_weight: float = 0.7
    object_pool_quantile: float = 0.9
    object_logit_bias: float = 0.0
    texture_logit_bias: float = 0.0


def save_vlm_prior_checkpoint(
    path: str | Path,
    head: VLMSemanticPriorHead,
    *,
    temperature: float = 1.0,
    support_centers: torch.Tensor | None = None,
    support_radii: torch.Tensor | None = None,
    metadata: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "architecture": TOKEN_PRIOR_ARCHITECTURE,
            "dim": head.dim,
            "hidden_dim": head.hidden_dim,
            "classes": list(VLM_PRIOR_CLASSES),
            "temperature": float(temperature),
            "support_centers": (
                None if support_centers is None else support_centers.detach().float().cpu()
            ),
            "support_radii": (
                None if support_radii is None else support_radii.detach().float().cpu()
            ),
            "metadata": metadata or {},
        },
        path,
    )


def save_pixel_prior_checkpoint(
    path: str | Path,
    head: PixelHierarchicalPriorHead,
    *,
    temperature: float = 1.0,
    object_pool_mean_weight: float = 0.7,
    object_pool_quantile: float = 0.9,
    object_logit_bias: float = 0.0,
    texture_logit_bias: float = 0.0,
    metadata: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "architecture": PIXEL_PRIOR_ARCHITECTURE,
            "width": head.width,
            "classes": list(VLM_PRIOR_CLASSES),
            "temperature": float(temperature),
            "object_pool_mean_weight": float(object_pool_mean_weight),
            "object_pool_quantile": float(object_pool_quantile),
            "object_logit_bias": float(object_logit_bias),
            "texture_logit_bias": float(texture_logit_bias),
            "metadata": metadata or {},
        },
        path,
    )


def load_vlm_prior_checkpoint(
    path: str | Path,
    device: torch.device | str,
) -> VLMPriorCheckpoint:
    payload = torch.load(path, map_location=device)
    classes = tuple(payload.get("classes", ()))
    if classes != VLM_PRIOR_CLASSES:
        raise ValueError(
            f"VLM prior class order must be {VLM_PRIOR_CLASSES}, got {classes}."
        )
    architecture = str(payload.get("architecture", TOKEN_PRIOR_ARCHITECTURE))
    if architecture == TOKEN_PRIOR_ARCHITECTURE:
        head = VLMSemanticPriorHead(
            dim=int(payload["dim"]),
            hidden_dim=int(payload["hidden_dim"]),
        ).to(device)
    elif architecture == PIXEL_PRIOR_ARCHITECTURE:
        head = PixelHierarchicalPriorHead(width=int(payload["width"])).to(device)
    else:
        raise ValueError(f"Unknown VLM prior architecture: {architecture!r}.")
    head.load_state_dict(payload["state_dict"], strict=True)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return VLMPriorCheckpoint(
        head=head,
        temperature=float(payload.get("temperature", 1.0)),
        support_centers=(
            None
            if payload.get("support_centers") is None
            else payload["support_centers"].to(device=device, dtype=torch.float32)
        ),
        support_radii=(
            None
            if payload.get("support_radii") is None
            else payload["support_radii"].to(device=device, dtype=torch.float32)
        ),
        metadata=dict(payload.get("metadata", {})),
        architecture=architecture,
        object_pool_mean_weight=float(payload.get("object_pool_mean_weight", 0.7)),
        object_pool_quantile=float(payload.get("object_pool_quantile", 0.9)),
        object_logit_bias=float(payload.get("object_logit_bias", 0.0)),
        texture_logit_bias=float(payload.get("texture_logit_bias", 0.0)),
    )


def predict_pixel_prior(
    head: PixelHierarchicalPriorHead,
    images: torch.Tensor,
    *,
    temperature: float = 1.0,
    object_logit_bias: float = 0.0,
    texture_logit_bias: float = 0.0,
) -> torch.Tensor:
    object_logit, texture_logit = head(images)
    temperature = max(float(temperature), 1e-6)
    objectness = torch.sigmoid((object_logit + float(object_logit_bias)) / temperature)
    conditional_texture = torch.sigmoid(
        (texture_logit + float(texture_logit_bias)) / temperature
    )
    background = (1.0 - objectness) * (1.0 - conditional_texture)
    texture = (1.0 - objectness) * conditional_texture
    return torch.cat([background, texture, objectness], dim=1)


def pixel_prior_to_tokens(
    pixel_prior: torch.Tensor,
    token_side: int,
    *,
    object_mean_weight: float = 0.7,
    object_quantile: float = 0.9,
) -> torch.Tensor:
    """Area-pool semantic probabilities and preserve sparse object responses."""

    if pixel_prior.ndim != 4 or pixel_prior.shape[1] != 3:
        raise ValueError(f"Expected [B,3,H,W] pixel prior, got {tuple(pixel_prior.shape)}.")
    token_side = int(token_side)
    if token_side <= 0:
        raise ValueError("token_side must be positive.")
    pooled = F.adaptive_avg_pool2d(pixel_prior.float(), (token_side, token_side))
    object_map = pixel_prior[:, 2:3].float()
    height, width = object_map.shape[-2:]
    if height % token_side == 0 and width % token_side == 0:
        patch_h, patch_w = height // token_side, width // token_side
        values = object_map.view(
            object_map.shape[0], 1, token_side, patch_h, token_side, patch_w
        ).permute(0, 1, 2, 4, 3, 5)
        values = values.reshape(object_map.shape[0], 1, token_side, token_side, -1)
        high = torch.quantile(values, float(object_quantile), dim=-1)
    else:
        high = F.adaptive_max_pool2d(object_map, (token_side, token_side))
    mean_weight = float(object_mean_weight)
    pooled[:, 2:3] = mean_weight * pooled[:, 2:3] + (1.0 - mean_weight) * high
    pooled = pooled.clamp_min(1e-6)
    pooled = pooled / pooled.sum(dim=1, keepdim=True)
    return pooled.permute(0, 2, 3, 1).reshape(pixel_prior.shape[0], token_side**2, 3)


def predict_vlm_prior(
    checkpoint: VLMPriorCheckpoint,
    tokens: torch.Tensor,
    images: torch.Tensor | None = None,
) -> torch.Tensor:
    if checkpoint.architecture == PIXEL_PRIOR_ARCHITECTURE:
        if images is None:
            raise ValueError("Pixel VLM prior requires the input images.")
        token_side = int(tokens.shape[1] ** 0.5)
        if token_side * token_side != tokens.shape[1]:
            raise ValueError("Pixel VLM prior requires a square token grid.")
        pixel_prior = predict_pixel_prior(
            checkpoint.head,
            images,
            temperature=checkpoint.temperature,
            object_logit_bias=checkpoint.object_logit_bias,
            texture_logit_bias=checkpoint.texture_logit_bias,
        )
        return pixel_prior_to_tokens(
            pixel_prior,
            token_side,
            object_mean_weight=checkpoint.object_pool_mean_weight,
            object_quantile=checkpoint.object_pool_quantile,
        ).to(dtype=tokens.dtype)
    logits = checkpoint.head(tokens) / max(float(checkpoint.temperature), 1e-6)
    return torch.softmax(logits, dim=-1).to(dtype=tokens.dtype)


def predict_vlm_objectness(
    checkpoint: VLMPriorCheckpoint,
    tokens: torch.Tensor,
    images: torch.Tensor | None = None,
    transition: float = 0.20,
) -> torch.Tensor:
    """OOD-style objectness: lack of support from every normal semantic mode."""

    if checkpoint.architecture == PIXEL_PRIOR_ARCHITECTURE:
        return predict_vlm_prior(checkpoint, tokens, images)[..., 2]

    if checkpoint.support_centers is None or checkpoint.support_radii is None:
        raise ValueError("VLM prior checkpoint does not contain normal support calibration.")
    query = F.normalize(tokens.detach().float(), dim=-1)
    centers = F.normalize(checkpoint.support_centers.to(query), dim=-1)
    radii = checkpoint.support_radii.to(query).clamp_min(1e-4)
    distance = 1.0 - torch.einsum("bnc,kmc->bnkm", query, centers)
    normalized = distance / radii.view(1, 1, *radii.shape)
    nearest = normalized.amin(dim=(-1, -2))
    return torch.sigmoid((nearest - 1.0) / max(float(transition), 1e-3)).to(tokens.dtype)


def fuse_vlm_objectness(
    learned_novelty: torch.Tensor,
    physical_objectness: torch.Tensor | None,
    physical_weight: float,
) -> torch.Tensor:
    """Retain the semantic OOD score and add only weak physical object evidence."""

    weight = float(physical_weight)
    if physical_objectness is None or weight <= 0.0:
        return learned_novelty
    if not 0.0 <= weight <= 1.0:
        raise ValueError("physical_weight must be in [0, 1].")
    return (
        1.0
        - (1.0 - learned_novelty.float())
        * (1.0 - weight * physical_objectness.float())
    ).clamp(0.0, 1.0).to(learned_novelty.dtype)


def fuse_vlm_and_physical_prior(
    vlm_prior: torch.Tensor,
    physical_prior: torch.Tensor | None,
    physical_weight: float,
    floor: float = 1e-4,
) -> torch.Tensor:
    """Log-probability pooling with the VLM prior as the dominant expert.

    physical_weight=0 is pure learned semantics.  A small positive value keeps
    weak edge/texture evidence without allowing the old hand prior to decide the
    semantic class on its own.
    """

    weight = float(physical_weight)
    if physical_prior is None or weight <= 0.0:
        return vlm_prior
    if not 0.0 <= weight <= 1.0:
        raise ValueError("physical_weight must be in [0, 1].")
    logits = (1.0 - weight) * vlm_prior.float().clamp_min(floor).log()
    logits = logits + weight * physical_prior.float().clamp_min(floor).log()
    return F.softmax(logits, dim=-1).to(dtype=vlm_prior.dtype)


def correct_physical_prior_with_vlm(
    vlm_prior: torch.Tensor,
    physical_prior: torch.Tensor,
    *,
    max_weight: float = 0.5,
    physical_margin_threshold: float = 0.15,
    vlm_margin_threshold: float = 0.15,
    floor: float = 1e-4,
    return_weight: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Use VLM semantics only where the physical prior is locally ambiguous.

    The correction strength is the product of physical-prior uncertainty and
    VLM confidence.  Consequently, a confident physical prior is unchanged,
    and an uncertain VLM prediction cannot overwrite it.
    """

    if vlm_prior.shape != physical_prior.shape:
        raise ValueError(
            "VLM and physical priors must have the same shape, got "
            f"{tuple(vlm_prior.shape)} and {tuple(physical_prior.shape)}."
        )
    maximum = float(max_weight)
    physical_threshold = float(physical_margin_threshold)
    vlm_threshold = float(vlm_margin_threshold)
    if not 0.0 <= maximum <= 1.0:
        raise ValueError("max_weight must be in [0, 1].")
    if physical_threshold <= 0.0 or vlm_threshold <= 0.0:
        raise ValueError("Prior confidence-margin thresholds must be positive.")

    physical_top2 = physical_prior.float().topk(2, dim=-1).values
    vlm_top2 = vlm_prior.float().topk(2, dim=-1).values
    physical_margin = physical_top2[..., 0] - physical_top2[..., 1]
    vlm_margin = vlm_top2[..., 0] - vlm_top2[..., 1]
    physical_uncertainty = (1.0 - physical_margin / physical_threshold).clamp(0.0, 1.0)
    vlm_confidence = (vlm_margin / vlm_threshold).clamp(0.0, 1.0)
    weight = maximum * physical_uncertainty * vlm_confidence

    weight_channel = weight.unsqueeze(-1)
    logits = (1.0 - weight_channel) * physical_prior.float().clamp_min(floor).log()
    logits = logits + weight_channel * vlm_prior.float().clamp_min(floor).log()
    corrected = F.softmax(logits, dim=-1).to(dtype=physical_prior.dtype)
    if return_weight:
        return corrected, weight.to(dtype=physical_prior.dtype)
    return corrected
