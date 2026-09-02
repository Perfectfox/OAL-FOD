from __future__ import annotations

import itertools

import numpy as np
import torch
import torch.nn.functional as F


def balanced_group_quotas(total: int, group_count: int) -> tuple[int, ...]:
    if total <= 0 or group_count <= 0:
        raise ValueError("Memory size and group count must be positive.")
    base, remainder = divmod(int(total), int(group_count))
    return tuple(base + int(index < remainder) for index in range(group_count))


def _kcenter_indices(features: torch.Tensor, count: int) -> torch.Tensor:
    """Deterministic cosine k-center selection on already normalized features."""

    if features.ndim != 2 or not features.shape[0]:
        raise ValueError("Expected a non-empty [tokens, channels] feature tensor.")
    count = min(int(count), int(features.shape[0]))
    centroid = F.normalize(features.mean(dim=0, keepdim=True), dim=-1)
    first = (features @ centroid.T).squeeze(1).argmin()
    selected = [first]
    chosen = torch.zeros(features.shape[0], dtype=torch.bool, device=features.device)
    chosen[first] = True
    best_similarity = features @ features[first]
    for _ in range(1, count):
        next_index = best_similarity.masked_fill(chosen, float("inf")).argmin()
        selected.append(next_index)
        chosen[next_index] = True
        best_similarity = torch.maximum(
            best_similarity,
            features @ features[next_index],
        )
    return torch.stack(selected)


def _kcenter_indices_with_initial(
    features: torch.Tensor,
    count: int,
    initial_indices: torch.Tensor,
) -> torch.Tensor:
    """Deterministic cosine k-center seeded by hierarchy-aware leaves."""

    if features.ndim != 2 or not features.shape[0]:
        raise ValueError("Expected a non-empty [tokens, channels] feature tensor.")
    count = min(int(count), int(features.shape[0]))
    initial = initial_indices.detach().long().flatten().to(features.device)
    if initial.numel() == 0 or initial.numel() > count:
        raise ValueError("Initial k-center indices must be non-empty and fit the quota.")
    if bool((initial < 0).any()) or bool((initial >= features.shape[0]).any()):
        raise ValueError("Initial k-center index is outside the candidate tensor.")
    if torch.unique(initial).numel() != initial.numel():
        raise ValueError("Initial k-center indices must be unique.")

    selected = [value for value in initial]
    chosen = torch.zeros(features.shape[0], dtype=torch.bool, device=features.device)
    chosen[initial] = True
    best_similarity = (features @ features[initial].T).max(dim=1).values
    for _ in range(int(initial.numel()), count):
        next_index = best_similarity.masked_fill(chosen, float("inf")).argmin()
        selected.append(next_index)
        chosen[next_index] = True
        best_similarity = torch.maximum(best_similarity, features @ features[next_index])
    return torch.stack(selected)


def _weighted_integer_quotas(
    weights: torch.Tensor,
    total: int,
    minimums: torch.Tensor,
) -> torch.Tensor:
    """Round proportional quotas while respecting deterministic minima."""

    if weights.ndim != 1 or minimums.shape != weights.shape:
        raise ValueError("Quota weights and minima must be equal one-dimensional tensors.")
    if total <= 0 or bool((weights <= 0.0).any()) or bool((minimums < 0).any()):
        raise ValueError("Quota weights must be positive and minima non-negative.")
    if int(minimums.sum()) > int(total):
        raise ValueError("Quota minima exceed the available capacity.")
    raw = weights.double() / weights.double().sum() * int(total)
    quotas = raw.floor().long()
    remainder = int(total) - int(quotas.sum())
    if remainder:
        fractions = raw - quotas.double()
        order = sorted(range(weights.numel()), key=lambda i: (-float(fractions[i]), i))
        for index in order[:remainder]:
            quotas[index] += 1
    for receiver in range(weights.numel()):
        while quotas[receiver] < minimums[receiver]:
            donors = [
                index
                for index in range(weights.numel())
                if quotas[index] > minimums[index]
            ]
            if not donors:
                raise RuntimeError("Could not satisfy quota minima.")
            donor = max(donors, key=lambda i: (int(quotas[i] - minimums[i]), -i))
            quotas[donor] -= 1
            quotas[receiver] += 1
    return quotas


def boundary_soft_semantic_weights(
    priors: torch.Tensor,
    confidence_margin: float,
) -> torch.Tensor:
    """Keep confident semantic assignments hard and split ambiguous top-2 mass.

    This removes the discontinuous group-label flip at a prior boundary without
    allowing the low prior floor of the third group to pollute every group.
    """

    if priors.ndim != 2 or priors.shape[1] != 3:
        raise ValueError("Semantic priors must have shape [tokens, 3].")
    if not 0.0 <= float(confidence_margin) <= 1.0:
        raise ValueError("Semantic confidence margin must be in [0,1].")
    if not bool(torch.isfinite(priors).all()) or bool((priors < 0.0).any()):
        raise ValueError("Semantic priors must be finite and non-negative.")
    normalized = priors.float() / priors.float().sum(dim=1, keepdim=True).clamp_min(1e-12)
    top = normalized.topk(2, dim=1)
    confidence = top.values[:, 0] - top.values[:, 1]
    hard = confidence >= float(confidence_margin)
    hard_output = torch.zeros_like(normalized).scatter(
        1, top.indices[:, :1], torch.ones_like(top.values[:, :1])
    )
    soft_values = top.values / top.values.sum(dim=1, keepdim=True).clamp_min(1e-12)
    soft_output = torch.zeros_like(normalized).scatter(1, top.indices, soft_values)
    return torch.where(hard[:, None], hard_output, soft_output)


def stratified_kcenter_memory(
    features: torch.Tensor,
    group_ids: torch.Tensor,
    size: int,
    *,
    max_candidates_per_group: int = 4096,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build an equal-capacity background/texture/object memory with k-center coverage."""

    if features.ndim != 2 or group_ids.shape != (features.shape[0],):
        raise ValueError("Features and group IDs have incompatible shapes.")
    if max_candidates_per_group <= 0:
        raise ValueError("max_candidates_per_group must be positive.")
    unique_groups = torch.unique(group_ids.cpu(), sorted=True)
    if unique_groups.numel() == 0:
        raise ValueError("No memory groups were provided.")
    quotas = balanced_group_quotas(size, int(unique_groups.numel()))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    output_features = []
    output_groups = []
    compute_device = torch.device(device) if device is not None else features.device
    normalized = F.normalize(features.float(), dim=-1).cpu()
    for group, quota in zip(unique_groups.tolist(), quotas):
        indices = torch.nonzero(group_ids.cpu() == int(group), as_tuple=False).flatten()
        if indices.numel() < quota:
            raise ValueError(
                f"Group {group} has {indices.numel()} candidates, fewer than quota {quota}."
            )
        if indices.numel() > max_candidates_per_group:
            order = torch.randperm(indices.numel(), generator=generator)[:max_candidates_per_group]
            indices = indices[order]
        candidates = normalized[indices].to(compute_device)
        selected_local = _kcenter_indices(candidates, quota).cpu()
        output_features.append(candidates[selected_local.to(compute_device)].cpu())
        output_groups.append(torch.full((quota,), int(group), dtype=torch.long))
    memory = torch.cat(output_features, dim=0)
    memory_groups = torch.cat(output_groups, dim=0)
    if memory.shape[0] != size:
        raise RuntimeError(f"Built {memory.shape[0]} memory tokens, expected {size}.")
    return memory, memory_groups


def view_group_balanced_kcenter_memory(
    features: torch.Tensor,
    view_ids: torch.Tensor,
    group_ids: torch.Tensor,
    size: int,
    *,
    max_candidates_per_stratum: int = 4096,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a view-balanced, de-duplicated memory with semantic-group coverage.

    Capacity is split equally between source views, then equally between the
    semantic groups present in each view.  K-center selection removes the heavy
    redundancy introduced by overlapping Clean-ROI crops.
    """

    if features.ndim != 2:
        raise ValueError("Expected features with shape [tokens, channels].")
    expected = (features.shape[0],)
    if view_ids.shape != expected or group_ids.shape != expected:
        raise ValueError("Features, view IDs, and group IDs have incompatible shapes.")
    if size <= 0 or max_candidates_per_stratum <= 0:
        raise ValueError("Memory size and candidate limit must be positive.")
    views = torch.unique(view_ids.cpu(), sorted=True)
    if views.numel() < 2:
        raise ValueError("View-balanced memory requires at least two source views.")
    view_quotas = balanced_group_quotas(size, int(views.numel()))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    compute_device = torch.device(device) if device is not None else features.device
    normalized = F.normalize(features.detach().float(), dim=-1).cpu()
    output_features: list[torch.Tensor] = []
    output_views: list[torch.Tensor] = []
    output_groups: list[torch.Tensor] = []
    for view, view_quota in zip(views.tolist(), view_quotas):
        view_mask = view_ids.cpu() == int(view)
        groups = torch.unique(group_ids.cpu()[view_mask], sorted=True)
        if groups.numel() == 0:
            raise ValueError(f"View {view} has no candidates.")
        group_quotas = balanced_group_quotas(view_quota, int(groups.numel()))
        for group, quota in zip(groups.tolist(), group_quotas):
            indices = torch.nonzero(
                view_mask & (group_ids.cpu() == int(group)), as_tuple=False
            ).flatten()
            if indices.numel() < quota:
                raise ValueError(
                    f"View {view}, group {group} has {indices.numel()} candidates, "
                    f"fewer than quota {quota}."
                )
            if indices.numel() > max_candidates_per_stratum:
                order = torch.randperm(indices.numel(), generator=generator)
                indices = indices[order[:max_candidates_per_stratum]]
            candidates = normalized[indices].to(compute_device)
            selected = _kcenter_indices(candidates, quota)
            output_features.append(candidates[selected].cpu())
            output_views.append(torch.full((quota,), int(view), dtype=torch.long))
            output_groups.append(torch.full((quota,), int(group), dtype=torch.long))
    memory = torch.cat(output_features, dim=0)
    memory_views = torch.cat(output_views, dim=0)
    memory_groups = torch.cat(output_groups, dim=0)
    if memory.shape[0] != size:
        raise RuntimeError(f"Built {memory.shape[0]} memory tokens, expected {size}.")
    return memory, memory_views, memory_groups


def view_balanced_kcenter_memory(
    features: torch.Tensor,
    view_ids: torch.Tensor,
    size: int,
    *,
    semantic_group_ids: torch.Tensor | None = None,
    max_candidates_per_view: int = 4096,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build an equal-capacity coreset per source view.

    Semantic groups are returned for diagnostics but do not receive artificial
    equal quotas. This keeps the memory representative of the normal data while
    k-center still favors rare modes that increase coverage.
    """

    if features.ndim != 2 or view_ids.shape != (features.shape[0],):
        raise ValueError("Features and view IDs have incompatible shapes.")
    if semantic_group_ids is None:
        semantic_group_ids = torch.full_like(view_ids, -1)
    if semantic_group_ids.shape != (features.shape[0],):
        raise ValueError("Semantic group IDs have an incompatible shape.")
    views = torch.unique(view_ids.cpu(), sorted=True)
    if views.numel() < 2:
        raise ValueError("View-balanced memory requires at least two source views.")
    if size <= 0 or max_candidates_per_view <= 0:
        raise ValueError("Memory size and candidate limit must be positive.")
    view_quotas = balanced_group_quotas(size, int(views.numel()))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    compute_device = torch.device(device) if device is not None else features.device
    normalized = F.normalize(features.detach().float(), dim=-1).cpu()
    output_features: list[torch.Tensor] = []
    output_views: list[torch.Tensor] = []
    output_groups: list[torch.Tensor] = []
    for view, quota in zip(views.tolist(), view_quotas):
        indices = torch.nonzero(view_ids.cpu() == int(view), as_tuple=False).flatten()
        if indices.numel() < quota:
            raise ValueError(f"View {view} has {indices.numel()} candidates, fewer than quota {quota}.")
        if indices.numel() > max_candidates_per_view:
            order = torch.randperm(indices.numel(), generator=generator)
            indices = indices[order[:max_candidates_per_view]]
        candidates = normalized[indices].to(compute_device)
        selected_local = _kcenter_indices(candidates, quota).cpu()
        selected_global = indices[selected_local]
        output_features.append(normalized[selected_global])
        output_views.append(torch.full((quota,), int(view), dtype=torch.long))
        output_groups.append(semantic_group_ids.cpu()[selected_global].long())
    memory = torch.cat(output_features, dim=0)
    memory_views = torch.cat(output_views, dim=0)
    memory_groups = torch.cat(output_groups, dim=0)
    if memory.shape[0] != size:
        raise RuntimeError(f"Built {memory.shape[0]} memory tokens, expected {size}.")
    return memory, memory_views, memory_groups


def _weighted_spherical_modes(
    features: torch.Tensor,
    weights: torch.Tensor,
    count: int,
    *,
    max_iterations: int = 25,
    initial_centers: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fit deterministic weighted spherical k-means centers.

    The farthest-first initialization intentionally matches the existing P2
    memory-mode construction.  Per-token weights are only used by the centroid
    updates, which lets every source view contribute equal total mass even when
    its semantic-group candidate count differs.
    """

    if features.ndim != 2 or weights.shape != (features.shape[0],):
        raise ValueError("Mode features and weights have incompatible shapes.")
    if count <= 0 or features.shape[0] < count:
        raise ValueError(f"Need at least {count} mode candidates, got {features.shape[0]}.")
    if max_iterations <= 0 or not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
        raise ValueError("Mode weights and iteration count must be positive and finite.")

    # The final full-pool refinement can contain millions of 768-D tokens.  It
    # already has initialized centers, so stream normalized chunks and weighted
    # centroid accumulators instead of materializing feature[assignment] tensors
    # that are several times larger than the source pool.
    stream_full_refine = initial_centers is not None and features.shape[0] > 65536
    if stream_full_refine:
        compute_device = initial_centers.device
        centers = F.normalize(initial_centers.detach().float(), dim=-1).to(compute_device)
        weight_total = weights.detach().double().sum().clamp_min(1e-12)
        chunk_size = 16384
        for _ in range(int(max_iterations)):
            sums = torch.zeros_like(centers)
            masses = torch.zeros(int(count), dtype=torch.float64, device=compute_device)
            for start in range(0, int(features.shape[0]), chunk_size):
                stop = min(start + chunk_size, int(features.shape[0]))
                chunk = F.normalize(
                    features[start:stop].detach().float().to(compute_device), dim=-1
                )
                chunk_weights = (
                    weights[start:stop].detach().double().to(compute_device) / weight_total
                )
                assignment = (chunk @ centers.T).argmax(dim=1)
                sums.index_add_(
                    0,
                    assignment,
                    chunk * chunk_weights.to(chunk.dtype)[:, None],
                )
                masses.index_add_(0, assignment, chunk_weights)
            updated = centers.clone()
            occupied = masses > 0.0
            updated[occupied] = F.normalize(sums[occupied], dim=-1)
            if torch.allclose(updated, centers, atol=1e-7, rtol=0.0):
                break
            centers = updated
        return F.normalize(centers, dim=-1)

    normalized = F.normalize(features.detach().float(), dim=-1)
    normalized_weights = weights.to(normalized).div(weights.sum().clamp_min(1e-12))
    if initial_centers is not None:
        if initial_centers.shape != (int(count), normalized.shape[1]):
            raise ValueError(
                "Initial mode centers do not match the requested mode count and feature dimension."
            )
        centers = F.normalize(initial_centers.detach().float().to(normalized), dim=-1)
    else:
        mean = F.normalize(
            (normalized * normalized_weights[:, None]).sum(dim=0, keepdim=True),
            dim=-1,
        )
        first = (normalized @ mean.T).squeeze(1).argmin()
        selected = [first]
        chosen = torch.zeros(normalized.shape[0], dtype=torch.bool, device=normalized.device)
        chosen[first] = True
        best_similarity = normalized @ normalized[first]
        for _ in range(1, int(count)):
            next_index = best_similarity.masked_fill(chosen, float("inf")).argmin()
            selected.append(next_index)
            chosen[next_index] = True
            best_similarity = torch.maximum(best_similarity, normalized @ normalized[next_index])
        centers = normalized[torch.stack(selected)].clone()
    for _ in range(int(max_iterations)):
        assignment = (normalized @ centers.T).argmax(dim=1)
        updated = centers.clone()
        for index in range(int(count)):
            members = assignment == index
            if bool(members.any()):
                member_weights = normalized_weights[members]
                updated[index] = F.normalize(
                    (normalized[members] * member_weights[:, None]).sum(dim=0),
                    dim=0,
                )
        if torch.allclose(updated, centers, atol=1e-7, rtol=0.0):
            break
        centers = updated
    return F.normalize(centers, dim=-1)


def _align_mode_centers(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """Globally align a small set of permutation-invariant mode centers."""

    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("Reference and candidate mode centers must have equal [modes, channels] shapes.")
    count = int(reference.shape[0])
    if count > 8:
        raise ValueError("Exact mode alignment is limited to at most eight centers per group.")
    similarity = F.normalize(reference.float(), dim=-1) @ F.normalize(candidate.float(), dim=-1).T
    best_permutation = max(
        itertools.permutations(range(count)),
        key=lambda permutation: float(
            sum(similarity[index, permutation[index]] for index in range(count))
        ),
    )
    return candidate[list(best_permutation)]


def stable_view_balanced_mode_teacher(
    features: torch.Tensor,
    view_ids: torch.Tensor,
    semantic_group_ids: torch.Tensor,
    groups: tuple[int, int, int],
    *,
    bootstrap_repeats: int = 12,
    bootstrap_fraction: float = 0.80,
    max_candidates_per_view_group: int = 2048,
    min_mode_fraction: float = 0.10,
    min_assignment_stability: float = 0.75,
    min_separation_ratio: float = 1.0,
    margin_quantile: float = 0.10,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    """Construct a self-constrained normal-mode teacher from all source views.

    This is deliberately independent of any older teacher.  Within each
    semantic group, source views receive equal total weight.  Stratified
    bootstrap fits are permutation-aligned and averaged into consensus centers.
    A group is trusted only when both modes have sufficient mass, assignments
    are stable under resampling, and inter-center distance exceeds normal
    within-mode spread.
    """

    if features.ndim != 2:
        raise ValueError("Expected teacher candidates with shape [tokens, channels].")
    expected = (features.shape[0],)
    if view_ids.shape != expected or semantic_group_ids.shape != expected:
        raise ValueError("Teacher features, view IDs, and semantic group IDs are incompatible.")
    if len(groups) != 3 or any(int(value) <= 0 for value in groups):
        raise ValueError(f"Expected three positive mode counts, got {groups}.")
    if bootstrap_repeats <= 0 or not 0.0 < bootstrap_fraction <= 1.0:
        raise ValueError("Bootstrap repeats and fraction must be positive.")
    if max_candidates_per_view_group <= 0:
        raise ValueError("Mode candidate limit must be positive.")
    if not 0.0 < min_mode_fraction < 1.0:
        raise ValueError("Minimum mode fraction must be in (0,1).")
    if not 0.0 <= min_assignment_stability <= 1.0 or min_separation_ratio < 0.0:
        raise ValueError("Mode stability thresholds are outside their valid ranges.")
    if not 0.0 <= margin_quantile < 0.5:
        raise ValueError("Mode margin quantile must be in [0,0.5).")

    compute_device = torch.device(device) if device is not None else features.device
    normalized = F.normalize(features.detach().float(), dim=-1).cpu()
    cpu_views = view_ids.detach().long().cpu()
    cpu_groups = semantic_group_ids.detach().long().cpu()
    all_views = torch.unique(cpu_views, sorted=True)
    if all_views.numel() < 2:
        raise ValueError("Stable mode construction requires at least two source views.")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))

    output_centers: list[torch.Tensor] = []
    group_reliability: list[float] = []
    margin_floors: list[float] = []
    margin_scales: list[float] = []
    group_diagnostics: dict[str, object] = {}

    for group_index, mode_count in enumerate(groups):
        source_strata: list[torch.Tensor] = []
        strata: list[torch.Tensor] = []
        stratum_sizes: dict[str, int] = {}
        missing_views: list[int] = []
        for view in all_views.tolist():
            indices = torch.nonzero(
                (cpu_groups == int(group_index)) & (cpu_views == int(view)),
                as_tuple=False,
            ).flatten()
            stratum_sizes[str(int(view))] = int(indices.numel())
            if indices.numel() < int(mode_count):
                missing_views.append(int(view))
                continue
            source_strata.append(indices)
            if indices.numel() > max_candidates_per_view_group:
                order = torch.randperm(indices.numel(), generator=generator)
                indices = indices[order[:max_candidates_per_view_group]]
            strata.append(indices)
        if missing_views:
            raise ValueError(
                f"Semantic group {group_index} has fewer than {mode_count} candidates "
                f"in source views {missing_views}."
            )

        selected_indices = torch.cat(strata)
        selected_features = normalized[selected_indices].to(compute_device)
        weights = torch.empty(selected_indices.numel(), dtype=torch.float32)
        offset = 0
        for indices in strata:
            count = int(indices.numel())
            weights[offset : offset + count] = 1.0 / (len(strata) * count)
            offset += count
        weights = weights.to(compute_device)
        base = _weighted_spherical_modes(selected_features, weights, int(mode_count))
        aligned_fits = [base]

        for _ in range(int(bootstrap_repeats)):
            sample_features = []
            sample_weights = []
            for source_indices in source_strata:
                capacity = min(int(source_indices.numel()), int(max_candidates_per_view_group))
                sample_count = max(int(mode_count), int(round(capacity * bootstrap_fraction)))
                sampled = torch.randint(
                    int(source_indices.numel()), (sample_count,), generator=generator
                )
                sample_features.append(normalized[source_indices[sampled]])
                sample_weights.append(
                    torch.full((sample_count,), 1.0 / (len(strata) * sample_count))
                )
            bootstrap_features = torch.cat(sample_features).to(compute_device)
            bootstrap_weights = torch.cat(sample_weights).to(compute_device)
            fitted = _weighted_spherical_modes(
                bootstrap_features,
                bootstrap_weights,
                int(mode_count),
            )
            aligned_fits.append(_align_mode_centers(base, fitted))

        fit_stack = torch.stack(aligned_fits)
        consensus = F.normalize(fit_stack.mean(dim=0), dim=-1)
        consensus = _align_mode_centers(base, consensus)
        full_indices = torch.cat(source_strata)
        full_features = normalized[full_indices]
        full_weights = torch.empty(full_indices.numel(), dtype=torch.float32)
        offset = 0
        for source_indices in source_strata:
            count = int(source_indices.numel())
            full_weights[offset : offset + count] = 1.0 / (len(source_strata) * count)
            offset += count
        consensus = _weighted_spherical_modes(
            full_features,
            full_weights,
            int(mode_count),
            initial_centers=consensus,
        )
        # Full-pool refinement intentionally runs on CPU so a large normal-token
        # pool does not exhaust GPU memory.  Move the refined centers back to the
        # bootstrap compute device before alignment and downstream assignment.
        consensus = consensus.to(base)
        consensus = _align_mode_centers(base, consensus)
        logits = selected_features @ consensus.T
        assignment = logits.argmax(dim=1)
        repeat_assignments = torch.stack(
            [(selected_features @ fitted.T).argmax(dim=1) for fitted in aligned_fits]
        )
        assignment_agreement = (repeat_assignments == assignment.unsqueeze(0)).float().mean(dim=0)
        weighted_assignment_stability = float((assignment_agreement * weights).sum().item())

        mode_fractions = []
        for mode_index in range(int(mode_count)):
            mode_fractions.append(float(weights[assignment == mode_index].sum().item()))
        min_fraction = min(mode_fractions)

        normalized_consensus = F.normalize(consensus, dim=-1)
        center_similarity = normalized_consensus @ normalized_consensus.T
        off_diagonal = ~torch.eye(int(mode_count), dtype=torch.bool, device=compute_device)
        min_center_distance = float((1.0 - center_similarity[off_diagonal]).min().item())
        assigned_similarity = logits.gather(1, assignment[:, None]).squeeze(1)
        within_distance_median = float(torch.quantile(1.0 - assigned_similarity, 0.50).item())
        within_distance_q90 = float(torch.quantile(1.0 - assigned_similarity, 0.90).item())
        # The hard separation constraint is measured against the typical
        # within-mode radius.  The q90 radius is retained as a tail diagnostic;
        # using it as the constraint would reject a stable mode because of the
        # widest ten percent of otherwise normal tokens.
        separation_ratio = min_center_distance / max(within_distance_median, 1e-6)

        top_two = logits.topk(min(2, int(mode_count)), dim=1).values
        if int(mode_count) == 1:
            margins = torch.ones_like(top_two[:, 0])
        else:
            margins = top_two[:, 0] - top_two[:, 1]
        margin_floor = float(torch.quantile(margins, float(margin_quantile)).item())
        margin_scale = float(torch.quantile(margins, 0.50).item())
        margin_scale = max(margin_scale, margin_floor + 1e-6)

        chance = 1.0 / int(mode_count)
        stability_score = max(
            0.0,
            min(1.0, (weighted_assignment_stability - chance) / max(1.0 - chance, 1e-6)),
        )
        occupancy_score = min(1.0, min_fraction / float(min_mode_fraction))
        separation_score = min(1.0, separation_ratio / max(float(min_separation_ratio), 1e-6))
        constraints_passed = bool(
            min_fraction >= float(min_mode_fraction)
            and weighted_assignment_stability >= float(min_assignment_stability)
            and separation_ratio >= float(min_separation_ratio)
        )
        reliability = stability_score * occupancy_score * separation_score if constraints_passed else 0.0
        center_stability = (fit_stack @ consensus.unsqueeze(0).transpose(1, 2)).diagonal(
            dim1=1, dim2=2
        )

        output_centers.append(consensus.detach().cpu())
        group_reliability.append(float(reliability))
        margin_floors.append(margin_floor)
        margin_scales.append(margin_scale)
        group_diagnostics[str(group_index)] = {
            "candidate_counts_by_view": stratum_sizes,
            "selected_candidates": int(selected_indices.numel()),
            "mode_fractions": mode_fractions,
            "minimum_mode_fraction": min_fraction,
            "assignment_stability": weighted_assignment_stability,
            "center_stability_by_mode": [
                float(value) for value in center_stability.mean(dim=0).cpu().tolist()
            ],
            "minimum_center_cosine_distance": min_center_distance,
            "within_mode_cosine_distance_median": within_distance_median,
            "within_mode_cosine_distance_q90": within_distance_q90,
            "separation_ratio": separation_ratio,
            "margin_floor": margin_floor,
            "margin_median": margin_scale,
            "constraints_passed": constraints_passed,
            "reliability": float(reliability),
        }

    diagnostics: dict[str, object] = {
        "version": "stable_view_bootstrap_full_refine_v3",
        "normal_only": True,
        "source_view_count": int(all_views.numel()),
        "semantic_groups": [int(value) for value in groups],
        "bootstrap_repeats": int(bootstrap_repeats),
        "bootstrap_fraction": float(bootstrap_fraction),
        "max_candidates_per_view_group": int(max_candidates_per_view_group),
        "constraints": {
            "minimum_mode_fraction": float(min_mode_fraction),
            "minimum_assignment_stability": float(min_assignment_stability),
            "minimum_separation_ratio": float(min_separation_ratio),
            "margin_quantile": float(margin_quantile),
        },
        "groups": group_diagnostics,
    }
    return (
        torch.cat(output_centers, dim=0),
        torch.tensor(group_reliability, dtype=torch.float32),
        torch.tensor(margin_floors, dtype=torch.float32),
        torch.tensor(margin_scales, dtype=torch.float32),
        diagnostics,
    )


def stable_view_balanced_soft_semantic_teacher(
    features: torch.Tensor,
    view_ids: torch.Tensor,
    semantic_weights: torch.Tensor,
    groups: tuple[int, int, int],
    *,
    bootstrap_repeats: int = 12,
    bootstrap_fraction: float = 0.80,
    max_candidates_per_view_group: int = 2048,
    min_mode_fraction: float = 0.10,
    min_assignment_stability: float = 0.75,
    min_separation_ratio: float = 1.0,
    margin_quantile: float = 0.10,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    """Construct stable modes while sharing ambiguous tokens across top-2 groups."""

    if features.ndim != 2:
        raise ValueError("Expected teacher candidates with shape [tokens, channels].")
    expected = (features.shape[0],)
    if view_ids.shape != expected or semantic_weights.shape != (features.shape[0], 3):
        raise ValueError("Teacher features, view IDs, and semantic weights are incompatible.")
    if len(groups) != 3 or any(int(value) <= 0 for value in groups):
        raise ValueError(f"Expected three positive mode counts, got {groups}.")
    if bootstrap_repeats <= 0 or not 0.0 < bootstrap_fraction <= 1.0:
        raise ValueError("Bootstrap repeats and fraction must be positive.")
    if max_candidates_per_view_group <= 0:
        raise ValueError("Mode candidate limit must be positive.")
    if not 0.0 < min_mode_fraction < 1.0:
        raise ValueError("Minimum mode fraction must be in (0,1).")
    if not 0.0 <= min_assignment_stability <= 1.0 or min_separation_ratio < 0.0:
        raise ValueError("Mode stability thresholds are outside their valid ranges.")
    if not 0.0 <= margin_quantile < 0.5:
        raise ValueError("Mode margin quantile must be in [0,0.5).")
    if (
        not bool(torch.isfinite(semantic_weights).all())
        or bool((semantic_weights < 0.0).any())
        or not bool(torch.allclose(
            semantic_weights.sum(dim=1),
            torch.ones(features.shape[0], dtype=semantic_weights.dtype),
            atol=1e-5,
            rtol=0.0,
        ))
    ):
        raise ValueError("Semantic weights must be finite non-negative rows summing to one.")

    compute_device = torch.device(device) if device is not None else features.device
    normalized = F.normalize(features.detach().float(), dim=-1).cpu()
    cpu_views = view_ids.detach().long().cpu()
    cpu_semantic = semantic_weights.detach().float().cpu()
    all_views = torch.unique(cpu_views, sorted=True)
    if all_views.numel() < 2:
        raise ValueError("Stable mode construction requires at least two source views.")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))

    output_centers: list[torch.Tensor] = []
    group_reliability: list[float] = []
    margin_floors: list[float] = []
    margin_scales: list[float] = []
    group_diagnostics: dict[str, object] = {}

    for group_index, mode_count in enumerate(groups):
        source_indices_by_view: list[torch.Tensor] = []
        source_mass_by_view: list[torch.Tensor] = []
        selected_indices_by_view: list[torch.Tensor] = []
        selected_mass_by_view: list[torch.Tensor] = []
        stratum_sizes: dict[str, int] = {}
        stratum_masses: dict[str, float] = {}
        for view in all_views.tolist():
            mask = (cpu_views == int(view)) & (cpu_semantic[:, group_index] > 0.0)
            indices = torch.nonzero(mask, as_tuple=False).flatten()
            mass = cpu_semantic[indices, group_index]
            stratum_sizes[str(int(view))] = int(indices.numel())
            stratum_masses[str(int(view))] = float(mass.sum())
            if indices.numel() < int(mode_count) or not bool((mass.sum() > 0).item()):
                raise ValueError(
                    f"Semantic group {group_index} has insufficient weighted candidates "
                    f"in source view {view}."
                )
            source_indices_by_view.append(indices)
            source_mass_by_view.append(mass)
            if indices.numel() > max_candidates_per_view_group:
                sampled = torch.multinomial(
                    mass / mass.sum(),
                    int(max_candidates_per_view_group),
                    replacement=False,
                    generator=generator,
                )
                indices = indices[sampled]
                mass = mass[sampled]
            selected_indices_by_view.append(indices)
            selected_mass_by_view.append(mass)

        selected_indices = torch.cat(selected_indices_by_view)
        selected_features = normalized[selected_indices].to(compute_device)
        selected_weights = torch.cat(
            [mass / (len(all_views) * mass.sum()) for mass in selected_mass_by_view]
        ).to(compute_device)
        base = _weighted_spherical_modes(
            selected_features, selected_weights, int(mode_count)
        )
        aligned_fits = [base]

        for _ in range(int(bootstrap_repeats)):
            sample_features = []
            sample_weights = []
            for source_indices, source_mass in zip(
                source_indices_by_view, source_mass_by_view
            ):
                capacity = min(
                    int(source_indices.numel()), int(max_candidates_per_view_group)
                )
                sample_count = max(
                    int(mode_count), int(round(capacity * bootstrap_fraction))
                )
                sampled = torch.multinomial(
                    source_mass / source_mass.sum(),
                    sample_count,
                    replacement=True,
                    generator=generator,
                )
                sample_features.append(normalized[source_indices[sampled]])
                sample_weights.append(
                    torch.full((sample_count,), 1.0 / (len(all_views) * sample_count))
                )
            fitted = _weighted_spherical_modes(
                torch.cat(sample_features).to(compute_device),
                torch.cat(sample_weights).to(compute_device),
                int(mode_count),
            )
            aligned_fits.append(_align_mode_centers(base, fitted))

        fit_stack = torch.stack(aligned_fits)
        consensus = F.normalize(fit_stack.mean(dim=0), dim=-1)
        consensus = _align_mode_centers(base, consensus)
        full_indices = torch.cat(source_indices_by_view)
        full_features = normalized[full_indices]
        full_weights = torch.cat(
            [mass / (len(all_views) * mass.sum()) for mass in source_mass_by_view]
        ).to(compute_device)
        consensus = _weighted_spherical_modes(
            full_features,
            full_weights.cpu(),
            int(mode_count),
            initial_centers=consensus,
        )
        # Full-pool refinement intentionally runs on CPU.  Restore the
        # bootstrap compute device before permutation alignment and assignment.
        consensus = consensus.to(base)
        consensus = _align_mode_centers(base, consensus)

        logits = selected_features @ consensus.T
        assignment = logits.argmax(dim=1)
        repeat_assignments = torch.stack(
            [(selected_features @ fitted.T).argmax(dim=1) for fitted in aligned_fits]
        )
        agreement = (repeat_assignments == assignment.unsqueeze(0)).float().mean(dim=0)
        assignment_stability = float((agreement * selected_weights).sum().item())
        mode_fractions = [
            float(selected_weights[assignment == index].sum().item())
            for index in range(int(mode_count))
        ]
        min_fraction = min(mode_fractions)

        normalized_consensus = F.normalize(consensus, dim=-1)
        center_similarity = normalized_consensus @ normalized_consensus.T
        off_diagonal = ~torch.eye(
            int(mode_count), dtype=torch.bool, device=compute_device
        )
        min_center_distance = float((1.0 - center_similarity[off_diagonal]).min())
        assigned_similarity = logits.gather(1, assignment[:, None]).squeeze(1)
        within_median = float(torch.quantile(1.0 - assigned_similarity, 0.50))
        within_q90 = float(torch.quantile(1.0 - assigned_similarity, 0.90))
        separation_ratio = min_center_distance / max(within_median, 1e-6)

        top_two = logits.topk(min(2, int(mode_count)), dim=1).values
        margins = (
            torch.ones_like(top_two[:, 0])
            if int(mode_count) == 1
            else top_two[:, 0] - top_two[:, 1]
        )
        margin_floor = float(torch.quantile(margins, float(margin_quantile)))
        margin_scale = max(
            float(torch.quantile(margins, 0.50)), margin_floor + 1e-6
        )
        chance = 1.0 / int(mode_count)
        stability_score = max(
            0.0,
            min(1.0, (assignment_stability - chance) / max(1.0 - chance, 1e-6)),
        )
        occupancy_score = min(1.0, min_fraction / float(min_mode_fraction))
        separation_score = min(
            1.0, separation_ratio / max(float(min_separation_ratio), 1e-6)
        )
        constraints_passed = bool(
            min_fraction >= float(min_mode_fraction)
            and assignment_stability >= float(min_assignment_stability)
            and separation_ratio >= float(min_separation_ratio)
        )
        reliability = (
            stability_score * occupancy_score * separation_score
            if constraints_passed
            else 0.0
        )
        center_stability = (
            fit_stack @ consensus.unsqueeze(0).transpose(1, 2)
        ).diagonal(dim1=1, dim2=2)

        output_centers.append(consensus.detach().cpu())
        group_reliability.append(float(reliability))
        margin_floors.append(margin_floor)
        margin_scales.append(margin_scale)
        group_diagnostics[str(group_index)] = {
            "candidate_counts_by_view": stratum_sizes,
            "semantic_mass_by_view": stratum_masses,
            "selected_candidates": int(selected_indices.numel()),
            "mode_fractions": mode_fractions,
            "minimum_mode_fraction": min_fraction,
            "assignment_stability": assignment_stability,
            "center_stability_by_mode": [
                float(value) for value in center_stability.mean(dim=0).cpu().tolist()
            ],
            "minimum_center_cosine_distance": min_center_distance,
            "within_mode_cosine_distance_median": within_median,
            "within_mode_cosine_distance_q90": within_q90,
            "separation_ratio": separation_ratio,
            "margin_floor": margin_floor,
            "margin_median": margin_scale,
            "constraints_passed": constraints_passed,
            "reliability": float(reliability),
        }

    diagnostics: dict[str, object] = {
        "version": "stable_soft_semantic_full_refine_v1",
        "normal_only": True,
        "source_view_count": int(all_views.numel()),
        "semantic_groups": [int(value) for value in groups],
        "bootstrap_repeats": int(bootstrap_repeats),
        "bootstrap_fraction": float(bootstrap_fraction),
        "max_candidates_per_view_group": int(max_candidates_per_view_group),
        "constraints": {
            "minimum_mode_fraction": float(min_mode_fraction),
            "minimum_assignment_stability": float(min_assignment_stability),
            "minimum_separation_ratio": float(min_separation_ratio),
            "margin_quantile": float(margin_quantile),
        },
        "groups": group_diagnostics,
    }
    return (
        torch.cat(output_centers, dim=0),
        torch.tensor(group_reliability, dtype=torch.float32),
        torch.tensor(margin_floors, dtype=torch.float32),
        torch.tensor(margin_scales, dtype=torch.float32),
        diagnostics,
    )


def adaptive_modes_from_stable_teacher(
    mode_centers: torch.Tensor,
    groups: tuple[int, int, int],
    mode_diagnostics: dict[str, object],
    bank: torch.Tensor,
    bank_view_ids: torch.Tensor,
    bank_group_ids: torch.Tensor,
    *,
    min_memory_members_per_mode: int = 2,
    min_memory_views_per_mode: int = 2,
) -> tuple[
    torch.Tensor,
    tuple[int, int, int],
    torch.Tensor,
    torch.Tensor,
    dict[str, object],
]:
    """Collapse unsupported fixed modes while preserving all prototype slots.

    The stable teacher is fitted on a view-balanced sample of the full Normal
    candidate pool.  A requested split is retained only when that fit passed
    its stability/occupancy/separation constraints *and* every resulting mode
    is represented by enough leaves and source views in the final memory
    coreset.  Otherwise the group's centers are merged into one weighted parent
    mode.  Prototype slots are never deleted: all slots of a collapsed group
    map to the same parent mode and are supervised through their summed
    probability downstream.

    The first implementation intentionally supports the existing two slots per
    semantic group.  This keeps ``inp_num=6`` and the familiarity gate fully
    checkpoint-compatible while making the effective mode count adaptive.
    """

    slot_groups = tuple(int(value) for value in groups)
    if len(slot_groups) != 3 or any(value != 2 for value in slot_groups):
        raise ValueError(
            "Adaptive Center6 currently requires exactly two prototype slots per group."
        )
    if mode_centers.ndim != 2 or mode_centers.shape[0] != sum(slot_groups):
        raise ValueError("Stable mode centers do not match the prototype slot groups.")
    expected = (bank.shape[0],)
    if bank.ndim != 2 or bank_view_ids.shape != expected or bank_group_ids.shape != expected:
        raise ValueError("Adaptive-mode memory tensors have incompatible shapes.")
    if bank.shape[1] != mode_centers.shape[1]:
        raise ValueError("Adaptive-mode centers and memory use different feature dimensions.")
    if min_memory_members_per_mode <= 0 or min_memory_views_per_mode <= 0:
        raise ValueError("Adaptive-mode memory support thresholds must be positive.")
    group_diagnostics = mode_diagnostics.get("groups")
    if not isinstance(group_diagnostics, dict):
        raise ValueError("Stable-mode diagnostics are missing per-group evidence.")

    normalized_bank = F.normalize(bank.detach().float().cpu(), dim=-1)
    normalized_centers = F.normalize(mode_centers.detach().float().cpu(), dim=-1)
    cpu_views = bank_view_ids.detach().long().cpu()
    cpu_groups = bank_group_ids.detach().long().cpu()
    total_views = int(torch.unique(cpu_views).numel())
    required_views = min(int(min_memory_views_per_mode), total_views)

    adaptive_centers: list[torch.Tensor] = []
    adaptive_groups: list[int] = []
    slot_to_mode: list[int] = []
    mode_group_ids: list[int] = []
    selection: dict[str, object] = {}
    source_start = 0
    mode_start = 0
    for group_index, slot_count in enumerate(slot_groups):
        source_stop = source_start + slot_count
        candidate = normalized_centers[source_start:source_stop]
        evidence = group_diagnostics.get(str(group_index))
        if not isinstance(evidence, dict):
            raise ValueError(f"Stable-mode diagnostics are missing group {group_index}.")
        memory_indices = torch.nonzero(
            cpu_groups == int(group_index), as_tuple=False
        ).flatten()
        if memory_indices.numel() == 0:
            raise ValueError(f"Adaptive-mode group {group_index} has no memory leaves.")
        distances = 1.0 - normalized_bank[memory_indices] @ candidate.T
        assignments = distances.argmin(dim=1)
        member_counts = [
            int((assignments == mode_index).sum()) for mode_index in range(slot_count)
        ]
        view_counts = [
            int(torch.unique(cpu_views[memory_indices][assignments == mode_index]).numel())
            for mode_index in range(slot_count)
        ]
        raw_constraints_passed = bool(evidence.get("constraints_passed", False))
        memory_support_passed = bool(
            min(member_counts) >= int(min_memory_members_per_mode)
            and min(view_counts) >= required_views
        )
        keep_split = raw_constraints_passed and memory_support_passed

        if keep_split:
            selected = candidate
            selected_count = slot_count
            local_slot_to_mode = list(range(mode_start, mode_start + slot_count))
            reason = "stable_candidate_split_and_memory_supported"
        else:
            fractions = evidence.get("mode_fractions")
            if not isinstance(fractions, list) or len(fractions) != slot_count:
                fractions = [1.0 / slot_count] * slot_count
            weights = torch.as_tensor(fractions, dtype=candidate.dtype)
            weights = weights / weights.sum().clamp_min(1e-12)
            parent = F.normalize((candidate * weights[:, None]).sum(dim=0), dim=0)
            selected = parent.unsqueeze(0)
            selected_count = 1
            local_slot_to_mode = [mode_start] * slot_count
            reason = (
                "candidate_split_failed"
                if not raw_constraints_passed
                else "final_memory_support_failed"
            )

        adaptive_centers.append(selected)
        adaptive_groups.append(selected_count)
        slot_to_mode.extend(local_slot_to_mode)
        mode_group_ids.extend([group_index] * selected_count)
        selection[str(group_index)] = {
            "requested_modes": slot_count,
            "selected_modes": selected_count,
            "reason": reason,
            "raw_constraints_passed": raw_constraints_passed,
            "memory_member_counts_for_requested_split": member_counts,
            "memory_view_counts_for_requested_split": view_counts,
            "required_memory_members_per_mode": int(min_memory_members_per_mode),
            "required_memory_views_per_mode": required_views,
        }
        source_start = source_stop
        mode_start += selected_count

    adaptive_group_tuple = tuple(adaptive_groups)
    diagnostics: dict[str, object] = {
        "version": "adaptive_stable_modes_parent_child_v1",
        "normal_only": True,
        "slot_groups": list(slot_groups),
        "selected_mode_groups": list(adaptive_group_tuple),
        "effective_mode_count": int(sum(adaptive_group_tuple)),
        "prototype_slot_count": int(sum(slot_groups)),
        "slot_to_mode": slot_to_mode,
        "mode_group_ids": mode_group_ids,
        "selection": selection,
    }
    return (
        torch.cat(adaptive_centers, dim=0),
        adaptive_group_tuple,
        torch.tensor(slot_to_mode, dtype=torch.long),
        torch.tensor(mode_group_ids, dtype=torch.long),
        diagnostics,
    )


def hierarchical_reliability_modes_from_stable_teacher(
    mode_centers: torch.Tensor,
    groups: tuple[int, int, int],
    mode_diagnostics: dict[str, object],
    bank: torch.Tensor,
    bank_view_ids: torch.Tensor,
    bank_group_ids: torch.Tensor,
    *,
    min_memory_members_per_mode: int = 2,
    min_memory_views_per_mode: int = 2,
) -> tuple[
    torch.Tensor,
    tuple[int, int, int],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, object],
]:
    """Keep candidate child modes but soften their supervision by reliability.

    The hard adaptive teacher is useful for validating memory support, but its
    all-or-nothing merge can force two physical prototype slots toward the same
    parent center.  This v2 representation instead keeps all candidate centers.
    Group-level supervision is always active downstream, while within-group
    supervision is weighted by continuous Normal-only evidence.  The score is
    exactly the pre-threshold product used by the stable teacher: occupancy,
    chance-corrected bootstrap stability, and separation.  Missing final-memory
    support still fails closed with zero child reliability.
    """

    slot_groups = tuple(int(value) for value in groups)
    if len(slot_groups) != 3 or any(value != 2 for value in slot_groups):
        raise ValueError(
            "Hierarchical adaptive Center6 currently requires two candidate modes per group."
        )
    # Reuse the hard selector's shape, evidence and final-memory support checks.
    *_, hard_diagnostics = adaptive_modes_from_stable_teacher(
        mode_centers,
        slot_groups,
        mode_diagnostics,
        bank,
        bank_view_ids,
        bank_group_ids,
        min_memory_members_per_mode=min_memory_members_per_mode,
        min_memory_views_per_mode=min_memory_views_per_mode,
    )
    constraints = mode_diagnostics.get("constraints")
    group_diagnostics = mode_diagnostics.get("groups")
    if not isinstance(constraints, dict) or not isinstance(group_diagnostics, dict):
        raise ValueError("Stable-mode diagnostics are missing constraints or groups.")
    min_fraction_threshold = float(constraints.get("minimum_mode_fraction", 0.10))
    min_separation_threshold = float(constraints.get("minimum_separation_ratio", 1.0))
    if min_fraction_threshold <= 0.0 or min_separation_threshold <= 0.0:
        raise ValueError("Stable-mode reliability thresholds must be positive.")

    reliabilities: list[float] = []
    reliability_diagnostics: dict[str, object] = {}
    for group_index, mode_count in enumerate(slot_groups):
        evidence = group_diagnostics.get(str(group_index))
        hard_evidence = hard_diagnostics["selection"][str(group_index)]
        if not isinstance(evidence, dict) or not isinstance(hard_evidence, dict):
            raise ValueError(f"Stable-mode diagnostics are missing group {group_index}.")
        chance = 1.0 / float(mode_count)
        minimum_fraction = float(evidence.get("minimum_mode_fraction", 0.0))
        assignment_stability = float(evidence.get("assignment_stability", 0.0))
        separation_ratio = float(evidence.get("separation_ratio", 0.0))
        occupancy_score = max(
            0.0, min(1.0, minimum_fraction / min_fraction_threshold)
        )
        stability_score = max(
            0.0,
            min(
                1.0,
                (assignment_stability - chance) / max(1.0 - chance, 1e-12),
            ),
        )
        separation_score = max(
            0.0, min(1.0, separation_ratio / min_separation_threshold)
        )
        member_counts = hard_evidence["memory_member_counts_for_requested_split"]
        view_counts = hard_evidence["memory_view_counts_for_requested_split"]
        required_members = int(hard_evidence["required_memory_members_per_mode"])
        required_views = int(hard_evidence["required_memory_views_per_mode"])
        memory_support_passed = bool(
            min(member_counts) >= required_members and min(view_counts) >= required_views
        )
        raw_reliability = occupancy_score * stability_score * separation_score
        reliability = raw_reliability if memory_support_passed else 0.0
        reliabilities.append(float(reliability))
        reliability_diagnostics[str(group_index)] = {
            "reliability": float(reliability),
            "raw_reliability": float(raw_reliability),
            "occupancy_score": float(occupancy_score),
            "stability_score": float(stability_score),
            "separation_score": float(separation_score),
            "memory_support_passed": memory_support_passed,
            "memory_member_counts": member_counts,
            "memory_view_counts": view_counts,
        }

    mode_group_ids = torch.repeat_interleave(
        torch.arange(len(slot_groups), dtype=torch.long),
        torch.as_tensor(slot_groups, dtype=torch.long),
    )
    diagnostics: dict[str, object] = {
        "version": "adaptive_hierarchical_reliability_v2",
        "normal_only": True,
        "slot_groups": list(slot_groups),
        "selected_mode_groups": list(slot_groups),
        "effective_mode_count": int(sum(slot_groups)),
        "prototype_slot_count": int(sum(slot_groups)),
        "slot_to_mode": list(range(sum(slot_groups))),
        "mode_group_ids": mode_group_ids.tolist(),
        "group_reliability": reliabilities,
        "reliability": reliability_diagnostics,
        "hard_v1_selection": hard_diagnostics["selection"],
    }
    return (
        F.normalize(mode_centers.detach().float().cpu(), dim=-1),
        slot_groups,
        torch.arange(sum(slot_groups), dtype=torch.long),
        mode_group_ids,
        torch.tensor(reliabilities, dtype=torch.float32),
        diagnostics,
    )


def hierarchical_mode_conditioned_memory(
    features: torch.Tensor,
    view_ids: torch.Tensor,
    semantic_weights: torch.Tensor,
    mode_centers: torch.Tensor,
    groups: tuple[int, int, int],
    size: int,
    *,
    mode_temperature: float = 0.10,
    group_reliability: torch.Tensor | None = None,
    group_quota_power: float = 0.70,
    fixed_group_quotas: tuple[int, int, int] | None = None,
    device: torch.device | str | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, object],
]:
    """Select full-pool memory leaves under stable Normal-mode centers."""

    if features.ndim != 2 or view_ids.shape != (features.shape[0],):
        raise ValueError("Memory features and view IDs are incompatible.")
    if semantic_weights.shape != (features.shape[0], 3):
        raise ValueError("Memory semantic weights must have shape [tokens,3].")
    if mode_centers.shape != (sum(groups), features.shape[1]):
        raise ValueError("Mode centers do not match memory features and groups.")
    if mode_temperature <= 0.0 or size < len(groups):
        raise ValueError("Memory size and mode temperature are invalid.")
    if not 0.0 < float(group_quota_power) <= 1.0:
        raise ValueError("Memory semantic-group quota power must be in (0,1].")
    if fixed_group_quotas is not None:
        if len(fixed_group_quotas) != len(groups) or any(value <= 0 for value in fixed_group_quotas):
            raise ValueError("Fixed memory group quotas must contain three positive integers.")
        if sum(fixed_group_quotas) != size:
            raise ValueError("Fixed memory group quotas must sum to the requested memory size.")
    if group_reliability is None:
        group_reliability = torch.ones(len(groups), dtype=torch.float32)
    if group_reliability.shape != (len(groups),):
        raise ValueError("Memory group reliability does not match semantic groups.")
    if not bool(torch.isfinite(group_reliability).all()) or bool((group_reliability < 0).any()):
        raise ValueError("Memory group reliability must be finite and non-negative.")
    views = torch.unique(view_ids.detach().long().cpu(), sorted=True)
    if views.numel() < 2:
        raise ValueError("Hierarchical memory requires at least two source views.")
    quotas = balanced_group_quotas(size, int(views.numel()))
    reliable = group_reliability.detach().float().cpu() > 0.0
    group_minimums = torch.tensor(
        [int(count) if bool(reliable[index]) else 1 for index, count in enumerate(groups)],
        dtype=torch.long,
    )
    if any(quota < int(group_minimums.sum()) for quota in quotas):
        raise ValueError("Every source view needs capacity for reliable Normal modes.")
    fixed_by_view: list[torch.Tensor] | None = None
    if fixed_group_quotas is not None:
        remaining = torch.tensor(fixed_group_quotas, dtype=torch.long)
        fixed_by_view = []
        for view_index, quota in enumerate(quotas):
            views_left = len(quotas) - view_index
            if views_left == 1:
                allocation = remaining.clone()
            else:
                allocation = _weighted_integer_quotas(
                    remaining.float(),
                    int(quota),
                    group_minimums,
                )
                maximum = remaining - group_minimums * (views_left - 1)
                allocation = torch.minimum(allocation, maximum)
                deficit = int(quota) - int(allocation.sum())
                while deficit > 0:
                    capacity = maximum - allocation
                    index = int(capacity.argmax())
                    if int(capacity[index]) <= 0:
                        raise ValueError("Cannot distribute fixed semantic quotas across views.")
                    allocation[index] += 1
                    deficit -= 1
            if int(allocation.sum()) != int(quota):
                raise ValueError("Fixed semantic view quotas do not match the view capacity.")
            fixed_by_view.append(allocation)
            remaining -= allocation
        if bool((remaining != 0).any()):
            raise ValueError("Fixed semantic quotas were not exhausted across source views.")

    compute_device = torch.device(device) if device is not None else features.device
    normalized = F.normalize(features.detach().float(), dim=-1).cpu()
    centers = F.normalize(mode_centers.detach().float(), dim=-1).to(compute_device)
    semantic = semantic_weights.detach().float().cpu()
    conditional = torch.zeros(features.shape[0], sum(groups), dtype=torch.float32)
    start = 0
    # Keep the full candidate pool on CPU.  UAVVASTE can exceed seven million
    # tokens, so materializing the complete float32 pool on a 24 GB GPU leaves
    # no room for the per-view/group k-center working set.  Conditional mode
    # probabilities are row-independent and can be computed in bounded chunks
    # without changing the result or the selected-memory algorithm.
    conditional_chunk_size = 65536
    for group_index, count in enumerate(groups):
        stop = start + int(count)
        for offset in range(0, normalized.shape[0], conditional_chunk_size):
            chunk_stop = min(offset + conditional_chunk_size, normalized.shape[0])
            normalized_chunk = normalized[offset:chunk_stop].to(compute_device)
            conditional[offset:chunk_stop, start:stop] = F.softmax(
                (normalized_chunk @ centers[start:stop].T) / float(mode_temperature),
                dim=1,
            ).cpu()
        start = stop
    semantic_assignments = semantic.argmax(dim=1)
    mode_to_group = torch.cat(
        [torch.full((int(count),), index, dtype=torch.long) for index, count in enumerate(groups)]
    )
    within_mode = torch.cat(
        [torch.arange(int(count), dtype=torch.long) for count in groups]
    )

    output_indices: list[torch.Tensor] = []
    counts_by_view_mode: dict[str, dict[str, int]] = {}
    quotas_by_view_group: dict[str, dict[str, int]] = {}
    for view_index, (view, quota) in enumerate(zip(views.tolist(), quotas)):
        view_mask = view_ids.cpu() == int(view)
        semantic_mass = semantic[view_mask].sum(dim=0).clamp_min(1e-12)
        group_quotas = (
            fixed_by_view[view_index]
            if fixed_by_view is not None
            else _weighted_integer_quotas(
                semantic_mass.pow(float(group_quota_power)),
                int(quota),
                group_minimums,
            )
        )
        quotas_by_view_group[str(int(view))] = {
            str(group): int(group_quotas[group]) for group in range(len(groups))
        }
        selected_by_view: list[torch.Tensor] = []
        for group_index, group_quota in enumerate(group_quotas.tolist()):
            indices = torch.nonzero(
                view_mask & (semantic_assignments == int(group_index)),
                as_tuple=False,
            ).flatten()
            if indices.numel() < int(group_quota):
                raise ValueError(
                    f"View {view}, semantic group {group_index} has {indices.numel()} "
                    f"candidates, fewer than quota {group_quota}."
                )
            candidates = normalized[indices].to(compute_device)
            start = sum(int(value) for value in groups[:group_index])
            stop = start + int(groups[group_index])
            if bool(reliable[group_index]):
                initial_local = []
                for mode_index in range(start, stop):
                    best = conditional[indices, mode_index].argmax()
                    initial_local.append(best)
                selected_local = _kcenter_indices_with_initial(
                    candidates,
                    int(group_quota),
                    torch.stack(initial_local).to(compute_device),
                ).cpu()
            else:
                selected_local = _kcenter_indices(candidates, int(group_quota)).cpu()
            selected_by_view.append(indices[selected_local])
        selected_global = torch.cat(selected_by_view)
        output_indices.append(selected_global)
        selected_groups = semantic_assignments[selected_global]
        selected_modes = torch.empty_like(selected_groups)
        for group_index, count in enumerate(groups):
            members = selected_groups == int(group_index)
            start = sum(int(value) for value in groups[:group_index])
            stop = start + int(count)
            selected_modes[members] = start + conditional[
                selected_global[members], start:stop
            ].argmax(dim=1)
        counts_by_view_mode[str(int(view))] = {
            str(mode): int((selected_modes == mode).sum())
            for mode in range(sum(groups))
        }

    selected_indices = torch.cat(output_indices)
    bank = normalized[selected_indices]
    bank_views = view_ids.cpu()[selected_indices].long()
    bank_groups = semantic_assignments[selected_indices].long()
    bank_modes = torch.empty_like(bank_groups)
    for group_index, count in enumerate(groups):
        members = bank_groups == int(group_index)
        start = sum(int(value) for value in groups[:group_index])
        stop = start + int(count)
        bank_modes[members] = start + conditional[
            selected_indices[members], start:stop
        ].argmax(dim=1)
    bank_within_modes = within_mode[bank_modes]
    bank_mode_probability = conditional[selected_indices, bank_modes]
    bank_mode_distance = 1.0 - (
        bank.to(compute_device) * centers[bank_modes.to(compute_device)]
    ).sum(dim=1).cpu()
    if bank.shape[0] != size:
        raise RuntimeError(f"Built {bank.shape[0]} memory leaves, expected {size}.")

    diagnostics: dict[str, object] = {
        "version": "hierarchical_semantic_quota_reliable_mode_kcenter_v3",
        "normal_only": True,
        "candidate_count": int(features.shape[0]),
        "source_view_count": int(views.numel()),
        "view_quotas": {str(int(view)): int(quota) for view, quota in zip(views.tolist(), quotas)},
        "semantic_group_quota_power": float(group_quota_power),
        "fixed_semantic_group_quotas": (
            None if fixed_group_quotas is None else list(fixed_group_quotas)
        ),
        "reliable_mode_seeded_groups": [
            int(index) for index in range(len(groups)) if bool(reliable[index])
        ],
        "memory_quotas_by_view_group": quotas_by_view_group,
        "memory_counts_by_view_mode": counts_by_view_mode,
        "memory_counts_by_group": {
            str(group): int((bank_groups == group).sum()) for group in range(3)
        },
        "memory_counts_by_mode": {
            str(mode): int((bank_modes == mode).sum()) for mode in range(sum(groups))
        },
        "mode_probability_mean": float(bank_mode_probability.mean()),
        "mode_probability_min": float(bank_mode_probability.min()),
        "mode_distance_mean": float(bank_mode_distance.mean()),
        "mode_distance_q95": float(torch.quantile(bank_mode_distance, 0.95)),
    }
    return (
        bank,
        bank_views,
        bank_groups,
        bank_within_modes,
        bank_mode_probability,
        bank_mode_distance,
        diagnostics,
    )


def _weighted_quantile(
    values: torch.Tensor,
    weights: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    """Return a deterministic weighted quantile for one-dimensional CPU tensors."""

    if values.ndim != 1 or weights.shape != values.shape or values.numel() == 0:
        raise ValueError("Weighted quantile expects non-empty matching vectors.")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("Weighted quantile must be in (0,1].")
    if (
        not bool(torch.isfinite(values).all())
        or not bool(torch.isfinite(weights).all())
        or bool((weights < 0.0).any())
        or not bool((weights.sum() > 0.0).item())
    ):
        raise ValueError("Weighted quantile inputs must be finite with positive mass.")
    order = values.argsort()
    sorted_values = values[order]
    cumulative = weights[order].cumsum(dim=0)
    target = cumulative[-1] * float(quantile)
    index = torch.searchsorted(cumulative, target, right=False).clamp_max(
        sorted_values.numel() - 1
    )
    return sorted_values[index]


def calibrate_view_balanced_semantic_mode_radii(
    features: torch.Tensor,
    view_ids: torch.Tensor,
    semantic_weights: torch.Tensor,
    mode_centers: torch.Tensor,
    mode_groups: tuple[int, int, int],
    *,
    quantile: float = 0.95,
    min_radius: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Calibrate effective-mode radii from the full soft-semantic Normal pool.

    Each semantic group is treated as an independent conditional distribution.
    Ambiguous boundary tokens retain their soft group mass.  Inside a group,
    tokens are assigned to the closest effective mode and a weighted radius is
    estimated separately for every source view.  The final radius is the median
    of view radii, preventing a large or diffuse camera view from dominating the
    cross-semantic distance scale.
    """

    if features.ndim != 2 or view_ids.shape != (features.shape[0],):
        raise ValueError("Mode-radius features and view IDs are incompatible.")
    if semantic_weights.shape != (features.shape[0], 3):
        raise ValueError("Mode-radius semantic weights must have shape [tokens,3].")
    if len(mode_groups) != 3 or any(int(count) <= 0 for count in mode_groups):
        raise ValueError("Mode-radius calibration requires three positive group counts.")
    if mode_centers.shape != (sum(mode_groups), features.shape[1]):
        raise ValueError("Mode-radius centers do not match features and groups.")
    if not 0.0 < quantile <= 1.0 or min_radius <= 0.0:
        raise ValueError("Mode-radius quantile and floor must be positive.")

    normalized = F.normalize(features.detach().float().cpu(), dim=-1)
    centers = F.normalize(mode_centers.detach().float().cpu(), dim=-1)
    views = view_ids.detach().long().cpu()
    semantic = semantic_weights.detach().float().cpu()
    unique_views = torch.unique(views, sorted=True)
    if unique_views.numel() < 2:
        raise ValueError("View-balanced mode radii require at least two source views.")

    output_radii: list[torch.Tensor] = []
    output_counts: list[int] = []
    group_diagnostics: dict[str, object] = {}
    mode_start = 0
    for group_index, mode_count in enumerate(mode_groups):
        mode_stop = mode_start + int(mode_count)
        group_centers = centers[mode_start:mode_stop]
        radius_by_mode: list[list[torch.Tensor]] = [
            [] for _ in range(int(mode_count))
        ]
        count_by_mode = [0 for _ in range(int(mode_count))]
        per_view: dict[str, object] = {}
        for view in unique_views.tolist():
            selected = (views == int(view)) & (semantic[:, group_index] > 0.0)
            indices = torch.nonzero(selected, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            weights = semantic[indices, group_index]
            distances = (1.0 - normalized[indices] @ group_centers.T).clamp(0.0, 2.0)
            assignments = distances.argmin(dim=1)
            view_summary: dict[str, object] = {}
            for local_mode in range(int(mode_count)):
                members = assignments == local_mode
                member_count = int(members.sum())
                if member_count == 0:
                    continue
                radius = _weighted_quantile(
                    distances[members, local_mode],
                    weights[members],
                    float(quantile),
                ).clamp_min(float(min_radius))
                radius_by_mode[local_mode].append(radius)
                count_by_mode[local_mode] += member_count
                view_summary[str(local_mode)] = {
                    "members": member_count,
                    "semantic_mass": float(weights[members].sum()),
                    "radius": float(radius),
                }
            per_view[str(int(view))] = view_summary

        group_radii = []
        for local_mode, view_radii in enumerate(radius_by_mode):
            if not view_radii or count_by_mode[local_mode] <= 0:
                raise ValueError(
                    f"Semantic group {group_index} mode {local_mode} has no radius support."
                )
            radius = torch.stack(view_radii).median().clamp_min(float(min_radius))
            group_radii.append(radius)
            output_radii.append(radius)
            output_counts.append(count_by_mode[local_mode])
        group_diagnostics[str(group_index)] = {
            "mode_radii": [float(value) for value in group_radii],
            "mode_member_counts": count_by_mode,
            "views": per_view,
        }
        mode_start = mode_stop

    radii = torch.stack(output_radii).float()
    member_counts = torch.tensor(output_counts, dtype=torch.long)
    diagnostics: dict[str, object] = {
        "version": "full_pool_soft_semantic_view_balanced_q_v1",
        "normal_only": True,
        "source_view_count": int(unique_views.numel()),
        "semantic_groups": [int(value) for value in mode_groups],
        "quantile": float(quantile),
        "radius_floor": float(min_radius),
        "radius_aggregation": "median_of_per_view_weighted_quantiles",
        "groups": group_diagnostics,
    }
    return radii, member_counts, diagnostics


def local_radius_adjusted_distance(
    queries: torch.Tensor,
    memory: torch.Tensor,
    local_radii: torch.Tensor,
    *,
    reference_radius: float | torch.Tensor | None = None,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return density-normalized, raw cosine distance, and nearest memory ID.

    ``raw * reference_radius / local_radius`` keeps the familiar distance scale
    while allowing broader normal modes a wider acceptance radius.
    """

    if queries.ndim < 2 or memory.ndim != 2:
        raise ValueError("Queries and memory must end in the same feature dimension.")
    if queries.shape[-1] != memory.shape[-1] or local_radii.shape != (memory.shape[0],):
        raise ValueError("Queries, memory, and local radii have incompatible shapes.")
    if chunk_size <= 0 or bool((local_radii <= 0).any()):
        raise ValueError("Chunk size and every local radius must be positive.")
    query = F.normalize(queries.detach().float(), dim=-1)
    bank = F.normalize(memory.detach().float(), dim=-1).to(query)
    flat = query.reshape(-1, query.shape[-1])
    best_similarity = flat.new_full((flat.shape[0],), -1.0)
    nearest = torch.zeros(flat.shape[0], dtype=torch.long, device=flat.device)
    for start in range(0, bank.shape[0], chunk_size):
        similarity = flat @ bank[start : start + chunk_size].T
        values, indices = similarity.max(dim=-1)
        update = values > best_similarity
        best_similarity[update] = values[update]
        nearest[update] = indices[update] + start
    raw = (1.0 - best_similarity).clamp(0.0, 2.0)
    radii = local_radii.detach().float().to(raw).clamp_min(1e-6)
    if reference_radius is None:
        reference = radii.median()
    else:
        reference = torch.as_tensor(reference_radius, dtype=raw.dtype, device=raw.device)
    adjusted = raw * reference.clamp_min(1e-6) / radii[nearest]
    shape = queries.shape[:-1]
    return adjusted.reshape(shape), raw.reshape(shape), nearest.reshape(shape)


def estimate_local_radii(
    candidates: torch.Tensor,
    memory: torch.Tensor,
    *,
    quantile: float = 0.95,
    min_radius: float = 1e-3,
    chunk_size: int = 8192,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Estimate one robust normal-coverage radius for every memory token.

    Candidate tokens are assigned to their nearest selected center. Sparse
    centers fall back to the global assignment-distance quantile.
    """

    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0,1).")
    if min_radius <= 0 or chunk_size <= 0:
        raise ValueError("min_radius and chunk_size must be positive.")
    bank = F.normalize(memory.detach().float(), dim=-1)
    assignments: list[torch.Tensor] = []
    distances: list[torch.Tensor] = []
    normalized = F.normalize(candidates.detach().float(), dim=-1)
    for start in range(0, normalized.shape[0], chunk_size):
        similarity = normalized[start : start + chunk_size] @ bank.T
        values, indices = similarity.max(dim=-1)
        assignments.append(indices.cpu())
        distances.append((1.0 - values).clamp(0.0, 2.0).cpu())
    assigned = torch.cat(assignments)
    distance = torch.cat(distances)
    fallback = torch.quantile(distance, quantile).clamp_min(min_radius)
    radii = torch.full((memory.shape[0],), float(fallback), dtype=torch.float32)
    populated = 0
    for index in range(memory.shape[0]):
        values = distance[assigned == index]
        if values.numel() >= 2:
            radii[index] = torch.quantile(values, quantile).clamp_min(min_radius)
            populated += 1
    diagnostics: dict[str, float | int] = {
        "candidate_count": int(candidates.shape[0]),
        "populated_centers": int(populated),
        "fallback_centers": int(memory.shape[0] - populated),
        "fallback_radius": float(fallback),
        "radius_min": float(radii.min()),
        "radius_median": float(radii.median()),
        "radius_max": float(radii.max()),
    }
    return radii, diagnostics


def calibrate_cross_view_local_radii(
    candidates: torch.Tensor,
    candidate_view_ids: torch.Tensor,
    memory: torch.Tensor,
    memory_view_ids: torch.Tensor,
    *,
    radius_quantile: float = 0.95,
    threshold_quantile: float = 0.95,
    min_radius: float = 1e-3,
    chunk_size: int = 8192,
) -> tuple[torch.Tensor, float, float, dict[str, float | int]]:
    """Calibrate local radii and threshold without looking at Val labels.

    Each normal token may query only memory centers originating from another
    source view.  This leave-one-view-out rule prevents overlapping crops from
    producing artificially tiny self/near-duplicate distances.
    """

    if candidates.ndim != 2 or memory.ndim != 2 or candidates.shape[1] != memory.shape[1]:
        raise ValueError("Candidates and memory must be compatible 2-D tensors.")
    if candidate_view_ids.shape != (candidates.shape[0],):
        raise ValueError("Candidate view IDs have an incompatible shape.")
    if memory_view_ids.shape != (memory.shape[0],):
        raise ValueError("Memory view IDs have an incompatible shape.")
    if not 0.0 < radius_quantile < 1.0 or not 0.0 < threshold_quantile < 1.0:
        raise ValueError("Radius and threshold quantiles must be in (0,1).")
    if min_radius <= 0 or chunk_size <= 0:
        raise ValueError("min_radius and chunk_size must be positive.")
    compute_device = candidates.device
    bank = F.normalize(memory.detach().float(), dim=-1).to(compute_device)
    candidate_views = candidate_view_ids.detach().long().to(compute_device)
    memory_views = memory_view_ids.detach().long().to(compute_device)
    all_raw: list[torch.Tensor] = []
    all_nearest: list[torch.Tensor] = []
    all_candidate_indices: list[torch.Tensor] = []
    for view in torch.unique(candidate_views, sorted=True).tolist():
        candidate_indices = torch.nonzero(candidate_views == int(view), as_tuple=False).flatten()
        allowed_indices = torch.nonzero(memory_views != int(view), as_tuple=False).flatten()
        if allowed_indices.numel() == 0:
            raise ValueError(f"View {view} has no cross-view memory centers.")
        allowed_bank = bank[allowed_indices]
        for start in range(0, candidate_indices.numel(), chunk_size):
            indices = candidate_indices[start : start + chunk_size]
            # Normalize only the active query chunk.  A full UAVVASTE normal
            # pool can require more than 20 GB in float32, while the nearest
            # cross-view center calculation is independently chunkable.
            query = F.normalize(candidates[indices].detach().float(), dim=-1)
            similarity = query @ allowed_bank.T
            values, local_indices = similarity.max(dim=-1)
            all_raw.append((1.0 - values).clamp(0.0, 2.0).cpu())
            all_nearest.append(allowed_indices[local_indices].cpu())
            all_candidate_indices.append(indices.cpu())
    raw_unordered = torch.cat(all_raw)
    nearest_unordered = torch.cat(all_nearest)
    candidate_indices = torch.cat(all_candidate_indices)
    raw = torch.empty(candidates.shape[0], dtype=torch.float32)
    nearest = torch.empty(candidates.shape[0], dtype=torch.long)
    raw[candidate_indices] = raw_unordered
    nearest[candidate_indices] = nearest_unordered
    fallback = torch.quantile(raw, radius_quantile).clamp_min(min_radius)
    radii = torch.full((memory.shape[0],), float(fallback), dtype=torch.float32)
    populated = 0
    for index in range(memory.shape[0]):
        values = raw[nearest == index]
        if values.numel() >= 2:
            radii[index] = torch.quantile(values, radius_quantile).clamp_min(min_radius)
            populated += 1
    reference = float(radii.median())
    adjusted = raw * reference / radii[nearest]
    threshold = float(torch.quantile(adjusted, threshold_quantile))
    diagnostics: dict[str, float | int] = {
        "candidate_count": int(candidates.shape[0]),
        "source_view_count": int(torch.unique(candidate_views).numel()),
        "populated_centers": int(populated),
        "fallback_centers": int(memory.shape[0] - populated),
        "fallback_radius": float(fallback),
        "radius_min": float(radii.min()),
        "radius_median": reference,
        "radius_max": float(radii.max()),
        "cross_view_distance_mean": float(raw.mean()),
        "cross_view_distance_q95": float(torch.quantile(raw, 0.95)),
        "adjusted_distance_mean": float(adjusted.mean()),
        "adjusted_distance_q95": float(torch.quantile(adjusted, 0.95)),
        "threshold_quantile": float(threshold_quantile),
        "normal_only_novelty_threshold": threshold,
    }
    return radii, reference, threshold, diagnostics


def threshold_for_target_fpr(
    normal_distances: np.ndarray | torch.Tensor,
    target_fpr: float,
    *,
    floor: float = 0.0,
) -> float:
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must be in (0,1).")
    values = (
        normal_distances.detach().float().cpu().numpy()
        if isinstance(normal_distances, torch.Tensor)
        else np.asarray(normal_distances, dtype=np.float32)
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite normal distances were provided.")
    return max(float(floor), float(np.quantile(values, 1.0 - float(target_fpr))))
