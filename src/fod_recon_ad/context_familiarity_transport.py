from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .context_normal_prototype import context_ring_descriptor, extract_inpformer_tokens


def _retrieval_source_diversity(selected_groups: torch.Tensor) -> torch.Tensor:
    """Fraction of top-k neighbors contributed by distinct valid source groups."""

    if selected_groups.ndim != 2:
        raise ValueError("Selected context-memory groups must have shape [Q,K].")
    first_occurrence = selected_groups >= 0
    for index in range(1, selected_groups.shape[1]):
        first_occurrence[:, index] &= ~(
            selected_groups[:, index : index + 1] == selected_groups[:, :index]
        ).any(dim=1)
    return first_occurrence.float().sum(dim=1) / float(selected_groups.shape[1])


class ContextFamiliarityTransportMemory(nn.Module):
    """Normal context-to-center memory with leave-source-out calibration."""

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
        super().__init__()
        self.radii = tuple(int(radius) for radius in radii)
        self.size = int(size)
        self.topk = int(topk)
        self.temperature = float(temperature)
        self.query_chunk_size = int(query_chunk_size)
        self.value_dim = int(dim)
        self.key_dim = min(int(key_dim), int(dim))
        scale_count = len(self.radii)
        self.register_buffer(
            "keys", torch.zeros(scale_count, self.size, self.key_dim), persistent=True
        )
        self.register_buffer("values", torch.zeros(self.size, dim), persistent=True)
        self.register_buffer("key_means", torch.zeros(scale_count, dim), persistent=True)
        self.register_buffer(
            "key_projections", torch.zeros(scale_count, dim, self.key_dim), persistent=True
        )
        self.register_buffer("group_ids", torch.full((self.size,), -1, dtype=torch.long), persistent=True)
        self.register_buffer("count", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer(
            "calibration_residuals", torch.full((scale_count, self.size), 2.0), persistent=True
        )
        self.register_buffer(
            "calibration_key_distances", torch.full((scale_count, self.size), 2.0), persistent=True
        )
        self.register_buffer(
            "calibration_objectness", torch.full((scale_count, self.size), 2.0), persistent=True
        )
        self.register_buffer("calibration_counts", torch.zeros(scale_count, dtype=torch.long), persistent=True)
        self.register_buffer("scale_reliability", torch.ones(scale_count), persistent=True)
        self.register_buffer(
            "radii_state", torch.tensor(self.radii, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "retrieval_config_state",
            torch.tensor([float(self.topk), self.temperature, float(self.key_dim)]),
            persistent=True,
        )
        self.register_buffer(
            "mode_signature_state",
            torch.tensor(tuple(float(value) for value in mode_signature), dtype=torch.float32),
            persistent=True,
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        expected = {
            "radii_state": self.radii_state,
            "retrieval_config_state": self.retrieval_config_state,
            "mode_signature_state": self.mode_signature_state,
        }
        for name, current in expected.items():
            key = f"{prefix}{name}"
            incoming = state_dict.get(key)
            if incoming is None:
                continue
            incoming = incoming.to(device=current.device, dtype=current.dtype)
            if incoming.shape != current.shape or not torch.allclose(incoming, current, atol=1e-7, rtol=0.0):
                error_msgs.append(
                    f"Context transport checkpoint config mismatch for {key}: "
                    f"checkpoint={incoming.detach().cpu().tolist()} current={current.detach().cpu().tolist()}"
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @property
    def ready(self) -> bool:
        return int(self.count.item()) > 0 and bool((self.calibration_counts > 0).all())

    @torch.no_grad()
    def set_entries(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> None:
        if keys.ndim != 3 or keys.shape[0] != len(self.radii):
            raise ValueError(f"Expected context keys [S,M,C], got {tuple(keys.shape)}.")
        if values.ndim != 2 or keys.shape[1:] != values.shape:
            raise ValueError("Context keys and center values must agree on [M,C].")
        if group_ids.shape != (values.shape[0],):
            raise ValueError("Context memory group_ids must have shape [M].")
        take = min(self.size, values.shape[0])
        self.keys.zero_()
        self.values.zero_()
        self.key_means.zero_()
        self.key_projections.zero_()
        self.group_ids.fill_(-1)
        raw_keys = keys[:, :take].detach().float().to(self.keys.device)
        raw_values = values[:take].detach().float().to(self.values.device)
        self.values[:take].copy_(raw_values)
        self.group_ids[:take].copy_(group_ids[:take].detach().long().to(self.group_ids))
        for scale_index in range(len(self.radii)):
            mean = raw_keys[scale_index].mean(dim=0)
            centered = raw_keys[scale_index] - mean
            rank = min(self.key_dim, centered.shape[0], centered.shape[1])
            with torch.random.fork_rng(devices=[centered.device] if centered.is_cuda else []):
                torch.manual_seed(20260715 + scale_index)
                if centered.is_cuda:
                    torch.cuda.manual_seed_all(20260715 + scale_index)
                _, _, projection = torch.pca_lowrank(
                    centered,
                    q=rank,
                    center=False,
                    niter=2,
                )
            self.key_means[scale_index].copy_(mean.to(self.key_means))
            self.key_projections[scale_index, :, :rank].copy_(
                projection[:, :rank].to(self.key_projections)
            )
            projected = centered @ projection[:, :rank]
            if rank < self.key_dim:
                projected = F.pad(projected, (0, self.key_dim - rank))
            self.keys[scale_index, :take].copy_(F.normalize(projected, dim=-1).to(self.keys))
            normal_objectness = (
                1.0
                - F.cosine_similarity(raw_values, raw_keys[scale_index], dim=-1)
            ).clamp(0.0, 2.0)
            self.calibration_objectness[scale_index].fill_(2.0)
            self.calibration_objectness[scale_index, :take].copy_(
                normal_objectness.sort().values
            )
        self.count.fill_(take)
        self._calibrate_leave_source_out()

    def _project_context(self, scale_index: int, context: torch.Tensor) -> torch.Tensor:
        centered = context.detach().float() - self.key_means[scale_index]
        projected = centered @ self.key_projections[scale_index]
        return F.normalize(projected, dim=-1)

    def _retrieve_scale(
        self,
        scale_index: int,
        query_context: torch.Tensor,
        source_group_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        count = int(self.count.item())
        keys = self.keys[scale_index, :count]
        values = self.values[:count]
        memory_groups = self.group_ids[:count]
        query = self._project_context(scale_index, query_context)
        batch, token_count, key_channels = query.shape
        value_channels = values.shape[-1]
        flat_query = query.reshape(-1, key_channels)
        flat_batch = torch.arange(batch, device=query.device).repeat_interleave(token_count)
        expected_parts = []
        key_distance_parts = []
        agreement_parts = []
        diversity_parts = []
        fallback_rows = 0
        for start in range(0, flat_query.shape[0], self.query_chunk_size):
            stop = min(start + self.query_chunk_size, flat_query.shape[0])
            similarity = flat_query[start:stop] @ keys.T
            if source_group_ids is not None:
                query_groups = source_group_ids[flat_batch[start:stop]].to(memory_groups.device)
                valid = (memory_groups.unsqueeze(0) >= 0) & (
                    memory_groups.unsqueeze(0) != query_groups.unsqueeze(1)
                )
                has_valid = valid.any(dim=1)
                fallback_rows += int((~has_valid).sum().item())
                similarity = torch.where(
                    has_valid.unsqueeze(1), similarity.masked_fill(~valid, -1e4), similarity
                )
            k = min(self.topk, count)
            top_similarity, top_indices = similarity.topk(k, dim=1)
            weights = F.softmax(top_similarity / self.temperature, dim=1)
            valid_neighbors = top_similarity > -1e3
            weights = weights * valid_neighbors.to(weights.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            selected_values = values[top_indices]
            selected_groups = memory_groups[top_indices].masked_fill(~valid_neighbors, -1)
            expected = (weights.unsqueeze(-1) * selected_values).sum(dim=1)
            normalized_expected = F.normalize(expected, dim=-1)
            agreement = (
                weights
                * F.cosine_similarity(selected_values, normalized_expected.unsqueeze(1), dim=-1).clamp(0.0, 1.0)
            ).sum(dim=1)
            expected_parts.append(expected)
            key_distance_parts.append((1.0 - top_similarity[:, 0]).clamp(0.0, 2.0))
            agreement_parts.append(agreement)
            diversity_parts.append(_retrieval_source_diversity(selected_groups))
        expected = torch.cat(expected_parts).reshape(batch, token_count, value_channels)
        key_distance = torch.cat(key_distance_parts).reshape(batch, token_count)
        agreement = torch.cat(agreement_parts).reshape(batch, token_count)
        diversity = torch.cat(diversity_parts).reshape(batch, token_count)
        return expected, key_distance, agreement, diversity, fallback_rows

    @staticmethod
    def _empirical_cdf(values: torch.Tensor, sorted_reference: torch.Tensor) -> torch.Tensor:
        flat = values.detach().float().reshape(-1).contiguous()
        positions = torch.searchsorted(sorted_reference.contiguous(), flat, right=True)
        return positions.to(values.dtype).reshape_as(values) / float(sorted_reference.numel())

    @torch.no_grad()
    def _calibrate_leave_source_out(self) -> None:
        count = int(self.count.item())
        groups = self.group_ids[:count]
        if int(torch.unique(groups[groups >= 0]).numel()) < 2:
            raise RuntimeError(
                "Context transport calibration requires at least two normal source images/groups."
            )
        residual_medians = []
        self.calibration_counts.zero_()
        for scale_index in range(len(self.radii)):
            expected_parts = []
            distance_parts = []
            valid_parts = []
            for start in range(0, count, self.query_chunk_size):
                stop = min(start + self.query_chunk_size, count)
                query = self.keys[scale_index, start:stop]
                similarity = query @ self.keys[scale_index, :count].T
                valid = groups.unsqueeze(0) != groups[start:stop].unsqueeze(1)
                valid = valid & (groups.unsqueeze(0) >= 0) & (groups[start:stop].unsqueeze(1) >= 0)
                valid_rows = valid.any(dim=1)
                similarity = similarity.masked_fill(~valid, -1e4)
                k = min(self.topk, count)
                top_similarity, top_indices = similarity.topk(k, dim=1)
                weights = F.softmax(top_similarity / self.temperature, dim=1)
                valid_neighbors = top_similarity > -1e3
                weights = weights * valid_neighbors.to(weights.dtype)
                weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
                predicted = (weights.unsqueeze(-1) * self.values[top_indices]).sum(dim=1)
                expected_parts.append(predicted)
                distance_parts.append((1.0 - top_similarity[:, 0]).clamp(0.0, 2.0))
                valid_parts.append(valid_rows)
            predicted = torch.cat(expected_parts)
            key_distance = torch.cat(distance_parts)
            valid_rows = torch.cat(valid_parts)
            residual = (
                1.0 - F.cosine_similarity(self.values[:count].float(), predicted.float(), dim=-1)
            ).clamp(0.0, 2.0)
            residual = residual[valid_rows]
            key_distance = key_distance[valid_rows]
            if residual.numel() == 0:
                raise RuntimeError("No cross-source normal samples were available for context calibration.")
            take = min(self.size, residual.numel())
            self.calibration_residuals[scale_index].fill_(2.0)
            self.calibration_key_distances[scale_index].fill_(2.0)
            self.calibration_residuals[scale_index, :take].copy_(residual.sort().values[:take])
            self.calibration_key_distances[scale_index, :take].copy_(key_distance.sort().values[:take])
            self.calibration_counts[scale_index] = take
            residual_medians.append(residual.median().clamp_min(1e-3))
        median_error = torch.stack(residual_medians)
        pooled_median = median_error.median().clamp_min(1e-3)
        bounded_reliability = pooled_median / (pooled_median + median_error)
        self.scale_reliability.copy_(bounded_reliability)

    @torch.no_grad()
    def transport(
        self,
        target_tokens: torch.Tensor,
        objectness: torch.Tensor,
        fixed_scale_weights: Sequence[float],
        adaptive_scale: bool,
        source_group_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        if not self.ready:
            raise RuntimeError("Context transport memory is empty or uncalibrated.")
        if objectness.shape != target_tokens.shape[:2]:
            raise ValueError("Objectness and target token grids do not match.")
        contexts = [context_ring_descriptor(target_tokens, radius) for radius in self.radii]
        expected_scales = []
        confidence_scales = []
        surprise_scales = []
        objectness_scales = []
        diversity_scales = []
        fallback_rows = 0
        for scale_index, context in enumerate(contexts):
            expected, key_distance, agreement, diversity, fallback = self._retrieve_scale(
                scale_index, context, source_group_ids
            )
            count = int(self.calibration_counts[scale_index].item())
            residual_reference = self.calibration_residuals[scale_index, :count]
            key_reference = self.calibration_key_distances[scale_index, :count]
            objectness_reference = self.calibration_objectness[scale_index, :count]
            residual = (
                1.0 - F.cosine_similarity(target_tokens.detach().float(), expected.float(), dim=-1)
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
            calibrated_objectness = self._empirical_cdf(local_contrast, objectness_reference)
            expected_scales.append(expected)
            confidence_scales.append(confidence)
            surprise_scales.append(surprise)
            objectness_scales.append(calibrated_objectness)
            diversity_scales.append(diversity)
            fallback_rows += fallback

        expected = torch.stack(expected_scales)
        confidence = torch.stack(confidence_scales)
        surprise = torch.stack(surprise_scales)
        calibrated_objectness = torch.stack(objectness_scales)
        source_diversity = torch.stack(diversity_scales)
        fixed = target_tokens.new_tensor(tuple(float(value) for value in fixed_scale_weights))
        fixed = fixed / fixed.sum().clamp_min(1e-6)
        if adaptive_scale:
            scores = confidence * self.scale_reliability.to(confidence)[:, None, None]
            denominator = scores.sum(dim=0, keepdim=True)
            fallback = fixed[:, None, None].expand_as(scores)
            scale_weights = torch.where(
                denominator > 1e-6,
                scores / denominator.clamp_min(1e-6),
                fallback,
            )
        else:
            scale_weights = fixed[:, None, None].expand_as(confidence)

        if adaptive_scale:
            scale_objectness = calibrated_objectness
        else:
            scale_objectness = objectness.detach().float().unsqueeze(0).expand_as(confidence)
        scale_gate = scale_objectness * confidence * surprise
        weighted_gate = scale_weights * scale_gate
        gate = weighted_gate.sum(dim=0).clamp(0.0, 1.0)
        effective_objectness = (scale_weights * scale_objectness).sum(dim=0).clamp(0.0, 1.0)
        target_gate = gate.to(target_tokens.dtype)
        target_weighted_gate = weighted_gate.to(target_tokens.dtype)
        transported = (1.0 - target_gate).unsqueeze(-1) * target_tokens
        transported = transported + (target_weighted_gate.unsqueeze(-1) * expected.to(target_tokens)).sum(dim=0)
        entropy = -(scale_weights.clamp_min(1e-8) * scale_weights.clamp_min(1e-8).log()).sum(dim=0)
        diagnostics = {
            "guided_transport_gate": float(gate.mean().cpu()),
            "guided_transport_confidence": float((scale_weights * confidence).sum(dim=0).mean().cpu()),
            "guided_transport_surprise": float((scale_weights * surprise).sum(dim=0).mean().cpu()),
            "guided_transport_objectness": float(
                effective_objectness.mean().cpu()
            ),
            "guided_transport_source_diversity": float(
                (scale_weights * source_diversity).sum(dim=0).mean().cpu()
            ),
            "guided_transport_shift": float(
                (1.0 - F.cosine_similarity(target_tokens.float(), transported.float(), dim=-1)).mean().cpu()
            ),
            "guided_transport_scale_entropy": float(entropy.mean().cpu()),
            "guided_transport_fallback_ratio": float(
                fallback_rows / max(1, len(self.radii) * target_tokens.shape[0] * target_tokens.shape[1])
            ),
            "guided_transport_memory_count": float(self.count.item()),
        }
        for scale_index, radius in enumerate(self.radii):
            diagnostics[f"guided_transport_scale_r{radius}"] = float(scale_weights[scale_index].mean().cpu())
            diagnostics[f"guided_transport_reliability_r{radius}"] = float(
                self.scale_reliability[scale_index].cpu()
            )
        return transported, effective_objectness, diagnostics


@contextmanager
def _temporary_eval(model: nn.Module):
    states = [(module, bool(module.training)) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, training in states:
            module.training = training


def _balanced_context_entries(
    buckets: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
    limit: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected_keys = []
    selected_values = []
    selected_groups = []
    offsets = {group_id: 0 for group_id in buckets}
    active = sorted(buckets)
    while active and len(selected_values) < limit:
        next_active = []
        for group_id in active:
            offset = offsets[group_id]
            entries = buckets[group_id]
            if offset < len(entries) and len(selected_values) < limit:
                keys, value = entries[offset]
                selected_keys.append(keys)
                selected_values.append(value)
                selected_groups.append(group_id)
                offsets[group_id] += 1
            if offsets[group_id] < len(entries):
                next_active.append(group_id)
        active = next_active
    if not selected_values:
        raise RuntimeError("No normal context candidates were collected.")
    return (
        torch.stack(selected_keys, dim=1),
        torch.stack(selected_values),
        torch.tensor(selected_groups, dtype=torch.long),
    )


@torch.no_grad()
def fit_context_familiarity_transport_memory(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    source_group_resolver,
    candidates_per_image: int,
    candidates_per_group: int,
    memory_build_batches: int = 0,
) -> dict[str, float]:
    memory = getattr(model, "guided_context_transport_memory", None)
    if not isinstance(memory, ContextFamiliarityTransportMemory):
        return {}
    if memory.ready:
        return {
            "guided_transport_memory_count": float(memory.count.item()),
            "guided_transport_memory_groups": float(
                torch.unique(memory.group_ids[: int(memory.count.item())]).numel()
            ),
        }

    buckets: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    generator = torch.Generator(device="cpu").manual_seed(20260715)
    batches = 0
    with _temporary_eval(model):
        for images, _, paths in loader:
            images = images.to(device, non_blocking=True)
            _, target_tokens, _, _ = extract_inpformer_tokens(model, images)
            contexts = [context_ring_descriptor(target_tokens, radius) for radius in memory.radii]
            group_ids = source_group_resolver.ids(paths).tolist()
            token_count = target_tokens.shape[1]
            take = min(int(candidates_per_image), token_count)
            for image_index, group_id in enumerate(group_ids):
                bucket = buckets.setdefault(int(group_id), [])
                remaining = int(candidates_per_group) - len(bucket)
                if remaining <= 0:
                    continue
                indices = torch.randperm(token_count, generator=generator)[: min(take, remaining)]
                device_indices = indices.to(target_tokens.device)
                values = target_tokens[image_index, device_indices].detach().float().cpu()
                scale_keys = torch.stack(
                    [context[image_index, device_indices].detach().float().cpu() for context in contexts]
                )
                bucket.extend((scale_keys[:, index], values[index]) for index in range(values.shape[0]))
            batches += 1
            if memory_build_batches > 0 and batches >= int(memory_build_batches):
                break

        keys, values, group_ids = _balanced_context_entries(buckets, memory.size)
        memory.set_entries(keys.to(device), values.to(device), group_ids.to(device))
    return {
        "guided_transport_memory_count": float(memory.count.item()),
        "guided_transport_memory_groups": float(torch.unique(group_ids).numel()),
        "guided_transport_memory_batches": float(batches),
        **{
            f"guided_transport_reliability_r{radius}": float(memory.scale_reliability[index].cpu())
            for index, radius in enumerate(memory.radii)
        },
    }


def context_familiarity_transport_memory_ready(model: nn.Module) -> bool:
    memory = getattr(model, "guided_context_transport_memory", None)
    return isinstance(memory, ContextFamiliarityTransportMemory) and memory.ready
