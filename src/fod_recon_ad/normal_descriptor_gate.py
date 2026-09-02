from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .context_familiarity_transport import (
    ContextFamiliarityTransportMemory,
    _balanced_context_entries,
    _temporary_eval,
)
from .context_normal_prototype import context_ring_descriptor, extract_inpformer_tokens


@dataclass(frozen=True)
class NormalDescriptorEvidence:
    """Normal-only evidence consumed by descriptor write/read gates."""

    expected_normal: torch.Tensor
    appearance_novelty: torch.Tensor
    context_surprise: torch.Tensor
    confidence: torch.Tensor
    local_objectness: torch.Tensor
    write_risk: torch.Tensor
    read_risk: torch.Tensor


class NormalDescriptorMemory(ContextFamiliarityTransportMemory):
    """Cross-source normal descriptor memory for semantic-free B/C/D gates.

    The inherited context memory predicts a normal center token from center-excluded
    context.  This class adds leave-source-out appearance calibration and exposes
    continuous evidence instead of assigning fixed background/texture/object labels.
    """

    def __init__(
        self,
        dim: int,
        radii: Sequence[int],
        size: int,
        topk: int,
        temperature: float,
        query_chunk_size: int,
        key_dim: int = 64,
        mode_signature: Sequence[float] = (),
    ) -> None:
        super().__init__(
            dim=dim,
            radii=radii,
            size=size,
            topk=topk,
            temperature=temperature,
            query_chunk_size=query_chunk_size,
            key_dim=key_dim,
            mode_signature=mode_signature,
        )
        self.register_buffer(
            "calibration_appearance_distances",
            torch.full((self.size,), 2.0),
            persistent=True,
        )
        self.register_buffer(
            "appearance_calibration_count",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "calibration_read_risks",
            torch.full((self.size,), 1.0),
            persistent=True,
        )
        self.register_buffer(
            "read_risk_calibration_count",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )

    @property
    def ready(self) -> bool:
        return (
            super().ready
            and int(self.appearance_calibration_count.item()) > 0
            and int(self.read_risk_calibration_count.item()) > 0
        )

    @torch.no_grad()
    def set_entries(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> None:
        super().set_entries(keys, values, group_ids)
        self._calibrate_appearance_leave_source_out()
        self._calibrate_read_risk_leave_source_out(
            keys[:, : int(self.count.item())].detach().to(self.values),
            values[: int(self.count.item())].detach().to(self.values),
            group_ids[: int(self.count.item())].detach().to(self.group_ids),
        )

    @torch.no_grad()
    def _calibrate_appearance_leave_source_out(self) -> None:
        count = int(self.count.item())
        values = F.normalize(self.values[:count].float(), dim=-1)
        groups = self.group_ids[:count]
        distances = []
        for start in range(0, count, self.query_chunk_size):
            stop = min(start + self.query_chunk_size, count)
            similarity = values[start:stop] @ values.T
            valid = (groups.unsqueeze(0) >= 0) & (
                groups.unsqueeze(0) != groups[start:stop].unsqueeze(1)
            )
            valid_rows = valid.any(dim=1)
            best = similarity.masked_fill(~valid, -1e4).max(dim=1).values
            distances.append((1.0 - best[valid_rows]).clamp(0.0, 2.0))
        calibrated = torch.cat(distances) if distances else values.new_empty(0)
        if calibrated.numel() == 0:
            raise RuntimeError(
                "Normal descriptor appearance calibration requires at least two source groups."
            )
        take = min(self.size, calibrated.numel())
        self.calibration_appearance_distances.fill_(2.0)
        self.calibration_appearance_distances[:take].copy_(
            calibrated.sort().values[:take]
        )
        self.appearance_calibration_count.fill_(take)

    @torch.no_grad()
    def _calibrate_read_risk_leave_source_out(
        self,
        raw_keys: torch.Tensor,
        values: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> None:
        confidence_scales = []
        surprise_scales = []
        for scale_index in range(len(self.radii)):
            expected, key_distance, agreement, diversity, _ = self._retrieve_scale(
                scale_index,
                raw_keys[scale_index].unsqueeze(1),
                group_ids,
            )
            calibration_count = int(self.calibration_counts[scale_index].item())
            residual_reference = self.calibration_residuals[
                scale_index, :calibration_count
            ]
            key_reference = self.calibration_key_distances[
                scale_index, :calibration_count
            ]
            residual = (
                1.0
                - F.cosine_similarity(values.float(), expected[:, 0].float(), dim=-1)
            ).clamp(0.0, 2.0)
            surprise_scales.append(self._empirical_cdf(residual, residual_reference))
            key_familiarity = 1.0 - self._empirical_cdf(
                key_distance[:, 0], key_reference
            )
            confidence_scales.append(
                (key_familiarity * agreement[:, 0] * diversity[:, 0]).clamp(0.0, 1.0)
            )
        confidence = torch.stack(confidence_scales)
        surprise = torch.stack(surprise_scales)
        scale_score = confidence * self.scale_reliability.to(confidence)[:, None]
        denominator = scale_score.sum(dim=0, keepdim=True)
        uniform = torch.full_like(scale_score, 1.0 / float(len(self.radii)))
        weights = torch.where(
            denominator > 1e-6,
            scale_score / denominator.clamp_min(1e-6),
            uniform,
        )
        raw_risk = (
            (weights * confidence).sum(dim=0)
            * (weights * surprise).sum(dim=0)
        ).clamp(0.0, 1.0)
        take = min(self.size, raw_risk.numel())
        self.calibration_read_risks.fill_(1.0)
        self.calibration_read_risks[:take].copy_(raw_risk.sort().values[:take])
        self.read_risk_calibration_count.fill_(take)

    def _appearance_distance(
        self,
        target_tokens: torch.Tensor,
        source_group_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, int]:
        count = int(self.count.item())
        memory = F.normalize(self.values[:count].float(), dim=-1)
        query = F.normalize(target_tokens.detach().float(), dim=-1)
        batch, token_count, channels = query.shape
        flat = query.reshape(-1, channels)
        flat_batch = torch.arange(batch, device=query.device).repeat_interleave(token_count)
        distances = []
        fallback_rows = 0
        memory_groups = self.group_ids[:count]
        for start in range(0, flat.shape[0], self.query_chunk_size):
            stop = min(start + self.query_chunk_size, flat.shape[0])
            similarity = flat[start:stop] @ memory.T
            if source_group_ids is not None:
                query_groups = source_group_ids[flat_batch[start:stop]].to(memory_groups.device)
                valid = (memory_groups.unsqueeze(0) >= 0) & (
                    memory_groups.unsqueeze(0) != query_groups.unsqueeze(1)
                )
                has_valid = valid.any(dim=1)
                fallback_rows += int((~has_valid).sum().item())
                similarity = torch.where(
                    has_valid.unsqueeze(1),
                    similarity.masked_fill(~valid, -1e4),
                    similarity,
                )
            distances.append((1.0 - similarity.max(dim=1).values).clamp(0.0, 2.0))
        return torch.cat(distances).reshape(batch, token_count), fallback_rows

    @torch.no_grad()
    def describe(
        self,
        target_tokens: torch.Tensor,
        source_group_ids: torch.Tensor | None = None,
    ) -> tuple[NormalDescriptorEvidence, dict[str, float]]:
        if not self.ready:
            raise RuntimeError("Normal descriptor memory is empty or uncalibrated.")
        if target_tokens.ndim != 3:
            raise ValueError("Normal descriptor tokens must have shape [B,N,C].")
        token_side = int(target_tokens.shape[1] ** 0.5)
        if token_side * token_side != target_tokens.shape[1]:
            raise ValueError("Normal descriptor context requires a square token grid.")
        if source_group_ids is not None and source_group_ids.shape != (target_tokens.shape[0],):
            raise ValueError("Normal descriptor source_group_ids must have one id per image.")

        expected_scales = []
        confidence_scales = []
        surprise_scales = []
        objectness_scales = []
        diversity_scales = []
        fallback_rows = 0
        for scale_index, radius in enumerate(self.radii):
            context = context_ring_descriptor(target_tokens, radius)
            expected, key_distance, agreement, diversity, fallback = self._retrieve_scale(
                scale_index,
                context,
                source_group_ids,
            )
            calibration_count = int(self.calibration_counts[scale_index].item())
            residual_reference = self.calibration_residuals[
                scale_index, :calibration_count
            ]
            key_reference = self.calibration_key_distances[
                scale_index, :calibration_count
            ]
            objectness_reference = self.calibration_objectness[
                scale_index, :calibration_count
            ]
            residual = (
                1.0
                - F.cosine_similarity(
                    target_tokens.detach().float(), expected.float(), dim=-1
                )
            ).clamp(0.0, 2.0)
            surprise = self._empirical_cdf(residual, residual_reference)
            key_familiarity = 1.0 - self._empirical_cdf(key_distance, key_reference)
            confidence = (key_familiarity * agreement * diversity).clamp(0.0, 1.0)
            local_contrast = (
                1.0
                - F.cosine_similarity(
                    target_tokens.detach().float(), context.detach().float(), dim=-1
                )
            ).clamp(0.0, 2.0)
            local_objectness = self._empirical_cdf(
                local_contrast, objectness_reference
            )
            expected_scales.append(expected)
            confidence_scales.append(confidence)
            surprise_scales.append(surprise)
            objectness_scales.append(local_objectness)
            diversity_scales.append(diversity)
            fallback_rows += fallback

        expected_stack = torch.stack(expected_scales)
        confidence_stack = torch.stack(confidence_scales)
        surprise_stack = torch.stack(surprise_scales)
        objectness_stack = torch.stack(objectness_scales)
        diversity_stack = torch.stack(diversity_scales)
        scale_score = confidence_stack * self.scale_reliability.to(confidence_stack)[
            :, None, None
        ]
        denominator = scale_score.sum(dim=0, keepdim=True)
        uniform = torch.full_like(scale_score, 1.0 / float(len(self.radii)))
        scale_weights = torch.where(
            denominator > 1e-6,
            scale_score / denominator.clamp_min(1e-6),
            uniform,
        )

        expected_normal = (scale_weights.unsqueeze(-1) * expected_stack).sum(dim=0)
        confidence = (scale_weights * confidence_stack).sum(dim=0).clamp(0.0, 1.0)
        context_surprise = (scale_weights * surprise_stack).sum(dim=0).clamp(0.0, 1.0)
        local_objectness = (scale_weights * objectness_stack).sum(dim=0).clamp(0.0, 1.0)
        source_diversity = (scale_weights * diversity_stack).sum(dim=0).clamp(0.0, 1.0)

        appearance_distance, appearance_fallback = self._appearance_distance(
            target_tokens,
            source_group_ids,
        )
        appearance_count = int(self.appearance_calibration_count.item())
        appearance_novelty = self._empirical_cdf(
            appearance_distance,
            self.calibration_appearance_distances[:appearance_count],
        ).clamp(0.0, 1.0)
        joint_novelty = 1.0 - (1.0 - appearance_novelty) * (1.0 - context_surprise)
        write_risk = (confidence * joint_novelty).clamp(0.0, 1.0)
        raw_read_risk = (confidence * context_surprise).clamp(0.0, 1.0)
        read_count = int(self.read_risk_calibration_count.item())
        read_risk = self._empirical_cdf(
            raw_read_risk,
            self.calibration_read_risks[:read_count],
        ).clamp(0.0, 1.0)
        evidence = NormalDescriptorEvidence(
            expected_normal=expected_normal.to(target_tokens.dtype),
            appearance_novelty=appearance_novelty.to(target_tokens.dtype),
            context_surprise=context_surprise.to(target_tokens.dtype),
            confidence=confidence.to(target_tokens.dtype),
            local_objectness=local_objectness.to(target_tokens.dtype),
            write_risk=write_risk.to(target_tokens.dtype),
            read_risk=read_risk.to(target_tokens.dtype),
        )
        query_count = max(1, target_tokens.shape[0] * target_tokens.shape[1])
        diagnostics = {
            "guided_descriptor_memory_count": float(self.count.item()),
            "guided_descriptor_appearance_novelty": float(appearance_novelty.mean().cpu()),
            "guided_descriptor_context_surprise": float(context_surprise.mean().cpu()),
            "guided_descriptor_confidence": float(confidence.mean().cpu()),
            "guided_descriptor_local_objectness": float(local_objectness.mean().cpu()),
            "guided_descriptor_source_diversity": float(source_diversity.mean().cpu()),
            "guided_descriptor_write_risk": float(write_risk.mean().cpu()),
            "guided_descriptor_read_risk": float(read_risk.mean().cpu()),
            "guided_descriptor_read_risk_raw": float(raw_read_risk.mean().cpu()),
            "guided_descriptor_context_fallback_ratio": float(
                fallback_rows / max(1, len(self.radii) * query_count)
            ),
            "guided_descriptor_appearance_fallback_ratio": float(
                appearance_fallback / query_count
            ),
        }
        for index, radius in enumerate(self.radii):
            diagnostics[f"guided_descriptor_scale_r{radius}"] = float(
                scale_weights[index].mean().cpu()
            )
            diagnostics[f"guided_descriptor_reliability_r{radius}"] = float(
                self.scale_reliability[index].cpu()
            )
        return evidence, diagnostics


@torch.no_grad()
def fit_normal_descriptor_memory(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    source_group_resolver,
    candidates_per_image: int,
    candidates_per_group: int,
    memory_build_batches: int = 0,
) -> dict[str, float]:
    memory = getattr(model, "guided_normal_descriptor_memory", None)
    if not isinstance(memory, NormalDescriptorMemory):
        return {}
    if memory.ready:
        return {
            "guided_descriptor_memory_count": float(memory.count.item()),
            "guided_descriptor_memory_groups": float(
                torch.unique(memory.group_ids[: int(memory.count.item())]).numel()
            ),
        }

    buckets: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    generator = torch.Generator(device="cpu").manual_seed(20260722)
    batches = 0
    with _temporary_eval(model):
        for images, _, paths in loader:
            images = images.to(device, non_blocking=True)
            _, target_tokens, _, _ = extract_inpformer_tokens(model, images)
            contexts = [
                context_ring_descriptor(target_tokens, radius)
                for radius in memory.radii
            ]
            group_ids = source_group_resolver.ids(paths).tolist()
            token_count = target_tokens.shape[1]
            take = min(int(candidates_per_image), token_count)
            for image_index, group_id in enumerate(group_ids):
                bucket = buckets.setdefault(int(group_id), [])
                remaining = int(candidates_per_group) - len(bucket)
                if remaining <= 0:
                    continue
                indices = torch.randperm(token_count, generator=generator)[
                    : min(take, remaining)
                ]
                device_indices = indices.to(target_tokens.device)
                values = target_tokens[
                    image_index, device_indices
                ].detach().float().cpu()
                scale_keys = torch.stack(
                    [
                        context[image_index, device_indices].detach().float().cpu()
                        for context in contexts
                    ]
                )
                bucket.extend(
                    (scale_keys[:, index], values[index])
                    for index in range(values.shape[0])
                )
            batches += 1
            if memory_build_batches > 0 and batches >= int(memory_build_batches):
                break

        keys, values, group_ids = _balanced_context_entries(buckets, memory.size)
        memory.set_entries(keys.to(device), values.to(device), group_ids.to(device))
    return {
        "guided_descriptor_memory_count": float(memory.count.item()),
        "guided_descriptor_memory_groups": float(torch.unique(group_ids).numel()),
        "guided_descriptor_memory_batches": float(batches),
        **{
            f"guided_descriptor_reliability_r{radius}": float(
                memory.scale_reliability[index].cpu()
            )
            for index, radius in enumerate(memory.radii)
        },
    }


def normal_descriptor_memory_ready(model: nn.Module) -> bool:
    memory = getattr(model, "guided_normal_descriptor_memory", None)
    return isinstance(memory, NormalDescriptorMemory) and memory.ready
