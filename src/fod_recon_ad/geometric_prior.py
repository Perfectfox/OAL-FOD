from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _robust_unit_interval(
    values: torch.Tensor,
    low_quantile: float,
    high_quantile: float,
) -> torch.Tensor:
    if not 0.0 <= low_quantile < high_quantile <= 1.0:
        raise ValueError("Robust quantiles must satisfy 0 <= low < high <= 1.")
    flat = values.flatten(1)
    low = torch.quantile(flat, float(low_quantile), dim=1).view(-1, 1, 1, 1)
    high = torch.quantile(flat, float(high_quantile), dim=1).view(-1, 1, 1, 1)
    return ((values - low) / (high - low).clamp_min(1e-6)).clamp(0.0, 1.0)


def depth_local_residual_objectness(
    predicted_depth: torch.Tensor,
    output_side: int,
    *,
    local_kernel: int = 5,
    depth_quantiles: tuple[float, float] = (0.02, 0.98),
    residual_quantiles: tuple[float, float] = (0.50, 0.99),
) -> torch.Tensor:
    """Convert relative depth into an interpretable local geometry residual.

    The score is invariant to a global affine rescaling of positive depth up to
    robust clipping.  A locally planar or smoothly sloped surface has a small
    residual, while a compact protrusion or depression has a larger residual.
    """

    if output_side <= 0:
        raise ValueError("output_side must be positive.")
    if local_kernel <= 1 or local_kernel % 2 == 0:
        raise ValueError("local_kernel must be an odd integer greater than one.")
    depth = predicted_depth.float()
    if depth.ndim == 3:
        depth = depth.unsqueeze(1)
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError(
            "predicted_depth must have shape [B,H,W] or [B,1,H,W], got "
            f"{tuple(predicted_depth.shape)}."
        )
    if not torch.isfinite(depth).all():
        raise ValueError("predicted_depth contains non-finite values.")

    depth = F.interpolate(
        depth,
        size=(int(output_side), int(output_side)),
        mode="bilinear",
        align_corners=False,
    )
    depth = _robust_unit_interval(depth, *depth_quantiles)
    padding = local_kernel // 2
    if min(depth.shape[-2:]) < local_kernel:
        raise ValueError(
            f"Depth grid {tuple(depth.shape[-2:])} is smaller than local kernel "
            f"{local_kernel}."
        )
    # Compute only where the full neighborhood exists.  Padding depth itself
    # would turn a smooth perspective slope into a synthetic border residual.
    local_mean = F.avg_pool2d(depth, kernel_size=local_kernel, stride=1)
    center = depth[:, :, padding:-padding, padding:-padding]
    residual = (center - local_mean).abs()
    residual = F.pad(
        residual,
        (padding, padding, padding, padding),
        mode="replicate",
    )
    residual = _robust_unit_interval(residual, *residual_quantiles)
    return residual.flatten(1)


def cross_view_feature_support_gate(
    features: torch.Tensor,
    view_ids: torch.Tensor,
    *,
    reference_size_per_view: int = 2048,
    radius_quantile: float = 0.95,
    temperature: float = 0.10,
    min_weight: float = 0.25,
    query_chunk_size: int = 1024,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Estimate whether each Normal token is supported by other Normal views.

    Matching is performed only in normalized feature space.  No pixel
    coordinates or absolute camera positions are consumed.  For every source
    view, each token is compared with a deterministic reference subset from
    every *other* view.  The median nearest-neighbor distance across those
    views is converted into a soft gate using a source-view Normal-only radius.
    """

    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must have non-empty shape [N,C].")
    if view_ids.ndim != 1 or view_ids.shape[0] != features.shape[0]:
        raise ValueError("view_ids must have shape [N] matching features.")
    if not torch.isfinite(features).all():
        raise ValueError("features contains non-finite values.")
    if reference_size_per_view <= 0 or query_chunk_size <= 0:
        raise ValueError("reference and query chunk sizes must be positive.")
    if not 0.0 < radius_quantile < 1.0:
        raise ValueError("radius_quantile must be in (0,1).")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if not 0.0 <= min_weight <= 1.0:
        raise ValueError("min_weight must be in [0,1].")

    unique_views = torch.unique(view_ids.detach().cpu(), sorted=True)
    if unique_views.numel() < 2:
        raise ValueError("Cross-view support requires at least two Normal views.")
    compute_device = torch.device(device)
    gate = torch.empty(features.shape[0], dtype=torch.float32)
    per_view: dict[str, dict[str, float | int]] = {}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    reference_indices: dict[int, torch.Tensor] = {}
    for raw_view in unique_views.tolist():
        view = int(raw_view)
        indices = torch.nonzero(view_ids.cpu() == view, as_tuple=False).flatten()
        if indices.numel() > reference_size_per_view:
            order = torch.randperm(indices.numel(), generator=generator)
            indices = indices[order[:reference_size_per_view]]
        reference_indices[view] = indices

    for raw_source in unique_views.tolist():
        source = int(raw_source)
        query_indices = torch.nonzero(
            view_ids.cpu() == source, as_tuple=False
        ).flatten()
        distances_to_views = []
        for raw_target in unique_views.tolist():
            target = int(raw_target)
            if target == source:
                continue
            reference = F.normalize(
                features[reference_indices[target]].float(), dim=1
            ).to(compute_device)
            chunks = []
            for start in range(0, query_indices.numel(), query_chunk_size):
                index = query_indices[start : start + query_chunk_size]
                query = F.normalize(features[index].float(), dim=1).to(compute_device)
                nearest_similarity = (query @ reference.T).amax(dim=1)
                chunks.append((1.0 - nearest_similarity).clamp_min(0.0).cpu())
            distances_to_views.append(torch.cat(chunks))
        consensus_distance = torch.stack(distances_to_views, dim=1).median(dim=1).values
        radius = torch.quantile(consensus_distance, float(radius_quantile)).clamp_min(1e-6)
        raw_gate = torch.sigmoid(
            (radius - consensus_distance) / (float(temperature) * radius)
        )
        source_gate = float(min_weight) + (1.0 - float(min_weight)) * raw_gate
        gate[query_indices] = source_gate
        per_view[str(source)] = {
            "query_count": int(query_indices.numel()),
            "other_view_count": int(unique_views.numel() - 1),
            "reference_count_per_other_view": int(
                min(
                    reference_size_per_view,
                    min(
                        reference_indices[int(view)].numel()
                        for view in unique_views.tolist()
                        if int(view) != source
                    ),
                )
            ),
            "normal_radius": float(radius),
            "distance_mean": float(consensus_distance.mean()),
            "distance_q95": float(torch.quantile(consensus_distance, 0.95)),
            "gate_mean": float(source_gate.mean()),
            "gate_min": float(source_gate.min()),
        }

    diagnostics: dict[str, Any] = {
        "enabled": True,
        "version": "normal_cross_view_feature_support_v1",
        "source_view_count": int(unique_views.numel()),
        "reference_size_per_view": int(reference_size_per_view),
        "radius_quantile": float(radius_quantile),
        "temperature": float(temperature),
        "min_weight": float(min_weight),
        "seed": int(seed),
        "matching_space": "l2_normalized_dino_feature_cosine",
        "uses_spatial_coordinates": False,
        "uses_val_test_gt": False,
        "gate_mean": float(gate.mean()),
        "gate_min": float(gate.min()),
        "gate_q05": float(torch.quantile(gate, 0.05)),
        "per_source_view": per_view,
    }
    return gate, diagnostics


def transfer_unsupported_object_mass_to_background(
    semantic_prior: torch.Tensor,
    support_gate: torch.Tensor,
) -> torch.Tensor:
    """Apply a soft object gate while preserving a normalized three-way prior."""

    if semantic_prior.ndim != 2 or semantic_prior.shape[1] != 3:
        raise ValueError("semantic_prior must have shape [N,3].")
    if support_gate.ndim != 1 or support_gate.shape[0] != semantic_prior.shape[0]:
        raise ValueError("support_gate must have shape [N].")
    if not torch.isfinite(semantic_prior).all() or not torch.isfinite(support_gate).all():
        raise ValueError("semantic prior and support gate must be finite.")
    gate = support_gate.to(
        device=semantic_prior.device, dtype=semantic_prior.dtype
    ).clamp(0.0, 1.0)
    output = semantic_prior.clone()
    suppressed = output[:, 2] * (1.0 - gate)
    output[:, 2] = output[:, 2] * gate
    output[:, 0] = output[:, 0] + suppressed
    return output / output.sum(dim=1, keepdim=True).clamp_min(1e-6)


@dataclass
class FrozenDepthResidualPrior:
    processor: object
    model: torch.nn.Module
    device: torch.device
    model_id: str
    revision: str
    local_kernel: int = 5
    prediction_count: int = 0
    score_sum: float = 0.0
    score_count: int = 0
    score_max: float = 0.0

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        device: torch.device | str,
        cache_dir: str | Path | None = None,
        local_kernel: int = 5,
    ) -> "FrozenDepthResidualPrior":
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        resolved_device = torch.device(device)
        processor = AutoImageProcessor.from_pretrained(
            model_id,
            cache_dir=None if cache_dir is None else str(cache_dir),
        )
        model = AutoModelForDepthEstimation.from_pretrained(
            model_id,
            cache_dir=None if cache_dir is None else str(cache_dir),
        ).to(resolved_device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        revision = str(getattr(model.config, "_commit_hash", "") or "unresolved")
        return cls(
            processor=processor,
            model=model,
            device=resolved_device,
            model_id=str(model_id),
            revision=revision,
            local_kernel=int(local_kernel),
        )

    @torch.inference_mode()
    def predict(self, image, output_side: int) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }
        predicted_depth = self.model(**inputs).predicted_depth
        score = depth_local_residual_objectness(
            predicted_depth,
            output_side,
            local_kernel=self.local_kernel,
        )
        self.prediction_count += int(score.shape[0])
        self.score_sum += float(score.double().sum().cpu())
        self.score_count += int(score.numel())
        self.score_max = max(self.score_max, float(score.max().cpu()))
        return score

    def diagnostics(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "local_kernel": int(self.local_kernel),
            "prediction_count": int(self.prediction_count),
            "score_mean": (
                0.0 if self.score_count == 0 else self.score_sum / self.score_count
            ),
            "score_max": float(self.score_max),
            "normal_only": True,
            "runtime_dependency": False,
        }
