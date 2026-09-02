from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class RiskConfig:
    position_weight: float = 0.55
    texture_weight: float = 0.30
    disagreement_weight: float = 0.25
    threshold: float = 0.55
    temperature: float = 0.18


@dataclass
class MapConfig:
    resize_mask: int = 448
    local_kernel: int = 9
    switch_threshold: float = 0.45
    switch_blend: float = 0.70
    near_sigma: float = 4.0
    far_sigma: float = 1.5


@dataclass
class RowStats:
    row_quantile: Dict[str, np.ndarray]
    row_mean: Dict[str, np.ndarray]
    quantile: float

    def to_jsonable(self) -> dict:
        return {
            "quantile": self.quantile,
            "row_quantile": {key: value.tolist() for key, value in self.row_quantile.items()},
            "row_mean": {key: value.tolist() for key, value in self.row_mean.items()},
        }


def reconstruction_components(en: Iterable[torch.Tensor], de: Iterable[torch.Tensor]) -> Dict[str, torch.Tensor]:
    residuals = []
    for en_feat, de_feat in zip(en, de):
        residual = 1.0 - F.cosine_similarity(en_feat.detach(), de_feat, dim=1).unsqueeze(1)
        residuals.append(residual.clamp_min(0.0))
    if not residuals:
        raise ValueError("No reconstruction feature groups were provided.")
    shallow = residuals[0]
    deep = residuals[-1]
    base = torch.stack(residuals, dim=0).mean(dim=0)
    return {
        "base": base,
        "shallow": shallow,
        "deep": deep,
        "disagreement": (shallow - deep).abs(),
    }


def normalize_per_image(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    flat = tensor.flatten(1)
    min_v = flat.min(dim=1)[0].view(-1, 1, 1, 1)
    max_v = flat.max(dim=1)[0].view(-1, 1, 1, 1)
    return (tensor - min_v) / (max_v - min_v + eps)


def feature_texture(en: Iterable[torch.Tensor]) -> torch.Tensor:
    first = list(en)[0].detach()
    feat = F.normalize(first, dim=1)
    gx = torch.zeros((feat.shape[0], 1, feat.shape[2], feat.shape[3]), dtype=feat.dtype, device=feat.device)
    gy = torch.zeros_like(gx)
    gx[:, :, :, 1:] = (feat[:, :, :, 1:] - feat[:, :, :, :-1]).square().mean(dim=1, keepdim=True)
    gy[:, :, 1:, :] = (feat[:, :, 1:, :] - feat[:, :, :-1, :]).square().mean(dim=1, keepdim=True)
    return torch.sqrt(gx + gy + 1e-8)


def topness_map(batch: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    y = (torch.arange(height, dtype=torch.float32, device=device) + 0.5) / float(height)
    top = 1.0 - y
    return top.view(1, 1, height, 1).expand(batch, 1, height, width)


def make_risk_map(components: Mapping[str, torch.Tensor], texture: torch.Tensor, config: RiskConfig) -> torch.Tensor:
    base = components["base"]
    top = topness_map(base.shape[0], base.shape[2], base.shape[3], base.device)
    tex = normalize_per_image(texture)
    disagreement = normalize_per_image(components["disagreement"])
    raw = (
        config.position_weight * top
        + config.texture_weight * tex
        + config.disagreement_weight * disagreement
    )
    risk = torch.sigmoid((raw - config.threshold) / max(config.temperature, 1e-6))
    return risk.clamp(0.0, 1.0)


def local_contrast(score: torch.Tensor, kernel_size: int) -> torch.Tensor:
    pad = kernel_size // 2
    local_mean = F.avg_pool2d(score, kernel_size=kernel_size, stride=1, padding=pad)
    return F.relu(score - local_mean)


def row_calibrate(score: torch.Tensor, stats: RowStats, key: str) -> torch.Tensor:
    quantile = torch.from_numpy(stats.row_quantile[key]).to(score.device, score.dtype)
    scale = quantile.view(1, 1, -1, 1).clamp_min(1e-6)
    return score / scale


def upsample(score: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(score, size=(size, size), mode="bilinear", align_corners=False)


def _gaussian_kernel1d(sigma: float, dtype: torch.dtype) -> torch.Tensor:
    radius = max(int(4.0 * sigma + 0.5), 1)
    x = torch.arange(-radius, radius + 1, dtype=dtype)
    kernel = torch.exp(-(x ** 2) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def smooth_np(maps: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return maps.astype(np.float32)
    tensor = torch.from_numpy(maps.astype(np.float32)).unsqueeze(1)
    kernel = _gaussian_kernel1d(sigma, tensor.dtype)
    pad = kernel.numel() // 2
    kernel_x = kernel.view(1, 1, 1, -1)
    kernel_y = kernel.view(1, 1, -1, 1)
    mode = "reflect" if pad < min(tensor.shape[-2], tensor.shape[-1]) else "replicate"
    tensor = F.pad(tensor, (pad, pad, 0, 0), mode=mode)
    tensor = F.conv2d(tensor, kernel_x)
    tensor = F.pad(tensor, (0, 0, pad, pad), mode=mode)
    tensor = F.conv2d(tensor, kernel_y)
    return tensor[:, 0].numpy().astype(np.float32)


def adaptive_smooth(raw: torch.Tensor, risk: torch.Tensor, near_sigma: float, far_sigma: float) -> np.ndarray:
    raw_np = raw[:, 0].detach().cpu().numpy().astype(np.float32)
    risk_np = risk[:, 0].detach().cpu().numpy().astype(np.float32)
    smooth_near = smooth_np(raw_np, near_sigma)
    smooth_far = smooth_np(raw_np, far_sigma)
    return ((1.0 - risk_np) * smooth_near + risk_np * smooth_far).astype(np.float32)


def fit_row_stats(
    model,
    loader,
    device: torch.device,
    quantile: float = 0.95,
    local_kernel: int = 9,
) -> RowStats:
    model.eval()
    rows: Dict[str, list[np.ndarray]] = {"base": [], "shallow": [], "deep": [], "contrast": []}
    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(device)
            en, de = model(images)[:2]
            components = reconstruction_components(en, de)
            contrast = local_contrast(components["shallow"], local_kernel)
            for key in ("base", "shallow", "deep"):
                rows[key].append(components[key].mean(dim=3).squeeze(1).detach().cpu().numpy())
            rows["contrast"].append(contrast.mean(dim=3).squeeze(1).detach().cpu().numpy())
    row_mean: Dict[str, np.ndarray] = {}
    row_quantile: Dict[str, np.ndarray] = {}
    for key, chunks in rows.items():
        values = np.concatenate(chunks, axis=0)
        row_mean[key] = values.mean(axis=0).astype(np.float32)
        row_quantile[key] = np.quantile(values, quantile, axis=0).astype(np.float32)
    return RowStats(row_quantile=row_quantile, row_mean=row_mean, quantile=quantile)


def fit_row_stats_from_components(
    loader,
    device: torch.device,
    component_fn: Callable[[torch.Tensor], Mapping[str, torch.Tensor]],
    quantile: float = 0.95,
    local_kernel: int = 9,
) -> RowStats:
    rows: Dict[str, list[np.ndarray]] = {"base": [], "shallow": [], "deep": [], "contrast": []}
    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(device)
            components = component_fn(images)
            contrast = local_contrast(components["shallow"], local_kernel)
            for key in ("base", "shallow", "deep"):
                rows[key].append(components[key].mean(dim=3).squeeze(1).detach().cpu().numpy())
            rows["contrast"].append(contrast.mean(dim=3).squeeze(1).detach().cpu().numpy())
    row_mean: Dict[str, np.ndarray] = {}
    row_quantile: Dict[str, np.ndarray] = {}
    for key, chunks in rows.items():
        values = np.concatenate(chunks, axis=0)
        row_mean[key] = values.mean(axis=0).astype(np.float32)
        row_quantile[key] = np.quantile(values, quantile, axis=0).astype(np.float32)
    return RowStats(row_quantile=row_quantile, row_mean=row_mean, quantile=quantile)


def build_variant_maps(
    components: Mapping[str, torch.Tensor],
    risk: torch.Tensor,
    stats: RowStats,
    map_config: MapConfig,
    memory_residual: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    base_raw = upsample(components["base"], map_config.resize_mask)
    base = row_calibrate(components["base"], stats, "base")
    shallow = row_calibrate(components["shallow"], stats, "shallow")
    deep = row_calibrate(components["deep"], stats, "deep")
    contrast = row_calibrate(local_contrast(components["shallow"], map_config.local_kernel), stats, "contrast")

    base_n = normalize_per_image(upsample(base, map_config.resize_mask))
    shallow_n = normalize_per_image(upsample(shallow, map_config.resize_mask))
    deep_n = normalize_per_image(upsample(deep, map_config.resize_mask))
    contrast_n = normalize_per_image(upsample(contrast, map_config.resize_mask))
    detail = torch.maximum(shallow_n, contrast_n)
    if memory_residual is not None:
        memory_n = normalize_per_image(upsample(memory_residual, map_config.resize_mask))
        detail = torch.maximum(detail, memory_n)
    risk_u = upsample(risk, map_config.resize_mask).clamp(0.0, 1.0)
    risk_soft = risk_u.pow(0.75)

    deep_near_detail_far = (1.0 - risk_soft) * (0.65 * base_n + 0.35 * deep_n) + risk_soft * (
        0.35 * base_n + 0.65 * detail
    )
    switched = torch.where(
        risk_u > map_config.switch_threshold,
        (1.0 - map_config.switch_blend) * base_n + map_config.switch_blend * detail,
        base_n,
    )

    return {
        "raw_base": smooth_np(normalize_per_image(base_raw)[:, 0].detach().cpu().numpy(), map_config.near_sigma),
        "row_base": smooth_np(base_n[:, 0].detach().cpu().numpy(), map_config.near_sigma),
        "deep_near_detail_far": adaptive_smooth(
            deep_near_detail_far,
            risk_u,
            near_sigma=map_config.near_sigma,
            far_sigma=map_config.far_sigma,
        ),
        "recon_switch": adaptive_smooth(
            switched,
            risk_u,
            near_sigma=map_config.near_sigma,
            far_sigma=map_config.far_sigma,
        ),
    }
