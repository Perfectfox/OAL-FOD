import numpy as np
import torch

from fod_recon_ad.familiarity_memory_rebuild import (
    _weighted_spherical_modes,
    adaptive_modes_from_stable_teacher,
    balanced_group_quotas,
    boundary_soft_semantic_weights,
    calibrate_view_balanced_semantic_mode_radii,
    hierarchical_mode_conditioned_memory,
    hierarchical_reliability_modes_from_stable_teacher,
    stable_view_balanced_soft_semantic_teacher,
    stable_view_balanced_mode_teacher,
    stratified_kcenter_memory,
    threshold_for_target_fpr,
)


def test_adaptive_modes_collapse_unseparated_group_and_keep_supported_split() -> None:
    centers = torch.eye(6)
    bank = torch.cat(
        [
            centers[0].repeat(4, 1),
            centers[1].repeat(4, 1),
            centers[2].repeat(4, 1),
            centers[3].repeat(4, 1),
            centers[4].repeat(4, 1),
            centers[5].repeat(4, 1),
        ]
    )
    groups = torch.arange(3).repeat_interleave(8)
    views = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1] * 3)
    diagnostics = {
        "groups": {
            "0": {"constraints_passed": False, "mode_fractions": [0.75, 0.25]},
            "1": {"constraints_passed": True, "mode_fractions": [0.5, 0.5]},
            "2": {"constraints_passed": True, "mode_fractions": [0.5, 0.5]},
        }
    }

    adaptive = adaptive_modes_from_stable_teacher(
        centers,
        (2, 2, 2),
        diagnostics,
        bank,
        views,
        groups,
    )
    selected, mode_groups, slot_to_mode, mode_group_ids, audit = adaptive

    assert selected.shape == (5, 6)
    assert mode_groups == (1, 2, 2)
    assert slot_to_mode.tolist() == [0, 0, 1, 2, 3, 4]
    assert mode_group_ids.tolist() == [0, 1, 1, 2, 2]
    assert audit["effective_mode_count"] == 5
    assert audit["selection"]["0"]["reason"] == "candidate_split_failed"
    assert audit["selection"]["1"]["selected_modes"] == 2


def test_hierarchical_reliability_keeps_children_and_softens_failed_split() -> None:
    centers = torch.eye(6)
    bank = torch.cat([centers[index].repeat(4, 1) for index in range(6)])
    groups = torch.arange(3).repeat_interleave(8)
    views = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1] * 3)
    diagnostics = {
        "constraints": {
            "minimum_mode_fraction": 0.10,
            "minimum_assignment_stability": 0.75,
            "minimum_separation_ratio": 1.0,
        },
        "groups": {
            "0": {
                "constraints_passed": False,
                "mode_fractions": [0.75, 0.25],
                "minimum_mode_fraction": 0.25,
                "assignment_stability": 0.90,
                "separation_ratio": 0.50,
            },
            "1": {
                "constraints_passed": True,
                "mode_fractions": [0.5, 0.5],
                "minimum_mode_fraction": 0.50,
                "assignment_stability": 1.0,
                "separation_ratio": 1.20,
            },
            "2": {
                "constraints_passed": True,
                "mode_fractions": [0.5, 0.5],
                "minimum_mode_fraction": 0.50,
                "assignment_stability": 0.75,
                "separation_ratio": 1.0,
            },
        },
    }

    selected, mode_groups, slot_to_mode, mode_group_ids, reliability, audit = (
        hierarchical_reliability_modes_from_stable_teacher(
            centers, (2, 2, 2), diagnostics, bank, views, groups
        )
    )

    assert selected.shape == (6, 6)
    assert mode_groups == (2, 2, 2)
    assert slot_to_mode.tolist() == list(range(6))
    assert mode_group_ids.tolist() == [0, 0, 1, 1, 2, 2]
    assert torch.allclose(reliability, torch.tensor([0.40, 1.0, 0.50]))
    assert audit["version"] == "adaptive_hierarchical_reliability_v2"
    assert audit["hard_v1_selection"]["0"]["selected_modes"] == 1


def test_weighted_spherical_modes_streams_large_full_refine() -> None:
    left = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(35000, 1)
    right = torch.tensor([0.0, 1.0, 0.0, 0.0]).repeat(35000, 1)
    features = torch.cat([left, right])
    weights = torch.ones(features.shape[0])
    initial = torch.stack([left[0], right[0]])

    centers = _weighted_spherical_modes(
        features,
        weights,
        2,
        initial_centers=initial,
    )

    assert torch.allclose(centers, initial, atol=1e-6)


def test_boundary_soft_semantic_weights_only_share_ambiguous_top2() -> None:
    priors = torch.tensor([[0.80, 0.10, 0.10], [0.45, 0.40, 0.15]])
    weights = boundary_soft_semantic_weights(priors, confidence_margin=0.15)

    assert torch.equal(weights[0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(weights[1], torch.tensor([0.45 / 0.85, 0.40 / 0.85, 0.0]))
    assert torch.allclose(weights.sum(dim=1), torch.ones(2))


def test_view_balanced_mode_radii_use_full_soft_semantic_pool() -> None:
    centers = torch.eye(3)
    features = []
    views = []
    semantic = []
    for view, scale in ((0, 0.05), (1, 0.20)):
        for group in range(3):
            neighbor = centers[(group + 1) % 3]
            for _ in range(4):
                features.append(
                    torch.nn.functional.normalize(
                        centers[group] + scale * neighbor, dim=0
                    )
                )
                views.append(view)
                weights = torch.zeros(3)
                weights[group] = 1.0
                semantic.append(weights)
    features.append(torch.nn.functional.normalize(centers[0] + centers[1], dim=0))
    views.append(1)
    semantic.append(torch.tensor([0.6, 0.4, 0.0]))

    first = calibrate_view_balanced_semantic_mode_radii(
        torch.stack(features),
        torch.tensor(views),
        torch.stack(semantic),
        centers,
        (1, 1, 1),
        quantile=0.95,
    )
    second = calibrate_view_balanced_semantic_mode_radii(
        torch.stack(features),
        torch.tensor(views),
        torch.stack(semantic),
        centers,
        (1, 1, 1),
        quantile=0.95,
    )

    radii, counts, diagnostics = first
    assert radii.shape == counts.shape == (3,)
    assert torch.equal(radii, second[0])
    assert torch.equal(counts, second[1])
    assert counts.tolist() == [9, 9, 8]
    assert diagnostics["source_view_count"] == 2
    assert diagnostics["version"] == "full_pool_soft_semantic_view_balanced_q_v1"


def test_balanced_group_quotas_preserve_total() -> None:
    assert balanced_group_quotas(256, 3) == (86, 85, 85)
    assert sum(balanced_group_quotas(10, 3)) == 10


def test_stratified_kcenter_memory_uses_every_group_quota() -> None:
    angles = torch.linspace(0.0, 1.0, 30)
    features = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    groups = torch.arange(3).repeat_interleave(10)

    memory, memory_groups = stratified_kcenter_memory(
        features,
        groups,
        9,
        max_candidates_per_group=10,
        seed=7,
    )

    assert memory.shape == (9, 2)
    assert torch.bincount(memory_groups, minlength=3).tolist() == [3, 3, 3]
    assert torch.allclose(memory.norm(dim=1), torch.ones(9), atol=1e-6)


def test_threshold_for_target_fpr_matches_requested_quantile() -> None:
    distances = np.arange(100, dtype=np.float32)
    threshold = threshold_for_target_fpr(distances, 0.10)

    assert np.isclose(threshold, np.quantile(distances, 0.90))
    assert np.mean(distances > threshold) == 0.10


def test_stable_mode_teacher_is_view_balanced_and_deterministic() -> None:
    generator = torch.Generator().manual_seed(17)
    features = []
    views = []
    groups = []
    for group in range(3):
        for view in range(3):
            for mode in range(2):
                center = torch.zeros(6)
                center[2 * group + mode] = 1.0
                noise = 0.03 * torch.randn(24, 6, generator=generator)
                features.append(torch.nn.functional.normalize(center + noise, dim=-1))
                views.append(torch.full((24,), view, dtype=torch.long))
                groups.append(torch.full((24,), group, dtype=torch.long))
    candidates = torch.cat(features)
    view_ids = torch.cat(views)
    group_ids = torch.cat(groups)

    first = stable_view_balanced_mode_teacher(
        candidates,
        view_ids,
        group_ids,
        (2, 2, 2),
        bootstrap_repeats=4,
        max_candidates_per_view_group=48,
        seed=9,
    )
    second = stable_view_balanced_mode_teacher(
        candidates,
        view_ids,
        group_ids,
        (2, 2, 2),
        bootstrap_repeats=4,
        max_candidates_per_view_group=48,
        seed=9,
    )

    centers, reliability, floor, scale, diagnostics = first
    assert centers.shape == (6, 6)
    assert torch.allclose(centers.norm(dim=1), torch.ones(6), atol=1e-6)
    assert torch.equal(centers, second[0])
    assert torch.all(reliability > 0.9)
    assert torch.all(scale > floor)
    assert all(
        group["constraints_passed"]
        for group in diagnostics["groups"].values()
    )


def test_soft_semantic_teacher_and_hierarchical_memory_cover_every_mode() -> None:
    generator = torch.Generator().manual_seed(23)
    features = []
    views = []
    semantic = []
    for view in range(3):
        for group in range(3):
            for mode in range(2):
                center = torch.zeros(6)
                center[2 * group + mode] = 1.0
                noise = 0.02 * torch.randn(16, 6, generator=generator)
                features.append(torch.nn.functional.normalize(center + noise, dim=-1))
                views.append(torch.full((16,), view, dtype=torch.long))
                weights = torch.zeros(16, 3)
                weights[:, group] = 1.0
                semantic.append(weights)
    candidates = torch.cat(features)
    view_ids = torch.cat(views)
    semantic_weights = torch.cat(semantic)

    teacher = stable_view_balanced_soft_semantic_teacher(
        candidates,
        view_ids,
        semantic_weights,
        (2, 2, 2),
        bootstrap_repeats=3,
        max_candidates_per_view_group=64,
        seed=11,
    )
    memory = hierarchical_mode_conditioned_memory(
        candidates,
        view_ids,
        semantic_weights,
        teacher[0],
        (2, 2, 2),
        36,
    )
    repeated = hierarchical_mode_conditioned_memory(
        candidates,
        view_ids,
        semantic_weights,
        teacher[0],
        (2, 2, 2),
        36,
    )

    bank, bank_views, bank_groups, bank_modes, probabilities, distances, diagnostics = memory
    assert bank.shape == (36, 6)
    assert torch.equal(bank, repeated[0])
    assert torch.bincount(bank_views, minlength=3).tolist() == [12, 12, 12]
    global_modes = bank_groups * 2 + bank_modes
    for view in range(3):
        assert torch.bincount(global_modes[bank_views == view], minlength=6).min() >= 1
    assert torch.all(probabilities > 0.0)
    assert torch.all(distances >= 0.0)
    assert diagnostics["candidate_count"] == candidates.shape[0]
