import argparse

import torch
import pytest
import torch.nn.functional as F
from pathlib import Path
from tempfile import TemporaryDirectory

import fod_recon_ad.prototype_guidance as prototype_guidance
from fod_recon_ad.normal_calibration import SourceGroupResolver, select_safe_alpha
from fod_recon_ad.prototype_guidance import (
    GuidedPrototypeConfig,
    FrozenCenter6Teacher,
    FrozenMemoryModeTeacher,
    FamiliarityRiskCalibrator,
    PerModeTargetGate,
    TargetRejectGate,
    NormalObjectTokenMemory,
    PrototypePriorHead,
    RobustPriorNormalizer,
    _apply_decoder_read_gate,
    _aggregate_guided_prototypes,
    _cap_removed_decoder_update,
    _decoder_read_layer_enabled,
    _aggregation_attention_with_gate,
    _build_priors,
    _combine_guided_gather_losses,
    _build_decoder_read_risk,
    _build_trainable_priors,
    _center6_balanced_loss,
    _center6_group_priors,
    _center6_teacher_distance_probability,
    center6_mode_normalized_features,
    _group_contrastive_loss,
    _guided_gather_loss,
    _guided_aggregation_alpha,
    _image_edge_texture,
    _intra_group_balance_loss,
    _intra_group_repulsion_loss,
    _memory_mode_supervision_loss,
    _mode_specific_routing_weights,
    _semantic_prototype_coverage_loss,
    _token_roi_coverage,
    _transition_token_roi_prior,
    add_guided_prototype_args,
    guided_target_gate_normal_anchor_loss,
    guided_target_gate_supervision_loss,
    guided_config_from_args,
    load_frozen_center6_teacher,
    load_frozen_memory_mode_teacher,
    resolve_spatial_prior_parameters,
)


def test_familiarity_risk_calibrator_is_identity_initialized_and_monotonic() -> None:
    calibrator = FamiliarityRiskCalibrator()
    risk = torch.tensor([[0.01, 0.2, 0.5, 0.8, 0.99]])

    calibrated, logits = calibrator(risk)

    assert torch.allclose(calibrated, risk, atol=1e-6)
    assert torch.all(logits[:, 1:] > logits[:, :-1])


def test_target_gate_supervision_is_balanced_and_backpropagates() -> None:
    model = torch.nn.Module()
    model.guided_target_gate_calibrator = FamiliarityRiskCalibrator()
    raw = torch.tensor([[0.2, 0.3, 0.7, 0.8]])
    calibrated, logits = model.guided_target_gate_calibrator(raw)
    model._guided_target_gate_raw_risk = raw
    model._guided_target_gate_logits = logits
    mask = torch.zeros(1, 1, 8, 8)
    mask[:, :, :4, :4] = 1.0

    supervised, diag = guided_target_gate_supervision_loss(model, mask)
    anchor, anchor_diag = guided_target_gate_normal_anchor_loss(model)
    (supervised + anchor).backward()

    assert supervised.item() > 0.0
    assert anchor.item() == 0.0
    assert diag["target_gate_positive_fraction"] == 0.25
    assert anchor_diag["target_gate_normal_raw_mean"] > 0.0
    assert model.guided_target_gate_calibrator.bias.grad is not None


def test_center6_mode_normalized_features_respect_mode_radii() -> None:
    teacher = FrozenCenter6Teacher(
        centers=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        radii=torch.tensor([0.2, 0.8]),
        member_counts=torch.tensor([4, 4]),
        groups=(1, 1, 0),
    )
    tokens = torch.tensor([[[0.8, 0.6], [0.6, 0.8]]])
    novelty, ratio, uncertainty = center6_mode_normalized_features(
        tokens, teacher, temperature=0.1
    )

    assert ratio[0, 0] > ratio[0, 1]
    assert novelty[0, 0] > novelty[0, 1]
    assert torch.all((uncertainty >= 0.0) & (uncertainty <= 1.0))


def test_target_reject_gate_is_monotonic_and_backpropagates() -> None:
    gate = TargetRejectGate(initial_residual_weight=0.05)
    familiarity = torch.full((1, 3), 0.3)
    mode = torch.tensor([[0.1, 0.5, 0.9]])
    objectness = torch.full_like(mode, 0.6)
    uncertainty = torch.full_like(mode, 0.4)

    risk, logits = gate(familiarity, mode, objectness, uncertainty)
    assert torch.all(risk[:, 1:] > risk[:, :-1])
    assert torch.all(gate.weights() > 0.0)
    logits.sum().backward()
    assert gate.raw_weights.grad is not None
    assert gate.bias.grad is not None


def test_per_mode_target_gate_is_monotonic_and_uses_sparse_fallback() -> None:
    gate = PerModeTargetGate(
        2, initial_temperature=0.1, minimum_positive_support=2
    )
    ratios = torch.tensor([[0.8, 1.0, 1.2, 1.4]])
    assignments = torch.tensor([[0, 0, 1, 1]])

    risk, logits = gate(ratios, assignments)
    assert torch.all(risk[:, 1:] > risk[:, :-1])
    assert torch.allclose(gate.slopes(), torch.full((2,), 10.0), atol=1e-5)
    gate.record_support(
        assignments,
        positive=torch.tensor([[True, True, False, False]]),
        negative=torch.tensor([[False, False, True, True]]),
    )
    assert gate.positive_support.tolist() == [2, 0]
    _, logits = gate(ratios, assignments)
    logits.sum().backward()
    assert gate.global_boundary.grad is not None
    assert gate.mode_boundary_delta.grad[0] is not None
    assert gate.mode_boundary_delta.grad[0].abs() > 0
    assert gate.mode_boundary_delta.grad[1] == 0


def test_per_mode_target_gate_supervision_uses_hard_negatives_and_records_support() -> None:
    model = torch.nn.Module()
    model.guided_target_mode_gate = PerModeTargetGate(
        2, initial_temperature=0.1, minimum_positive_support=1
    )
    ratios = torch.tensor([[1.3, 0.8, 1.2, 0.4]])
    assignments = torch.tensor([[0, 0, 1, 1]])
    _, logits = model.guided_target_mode_gate(ratios, assignments)
    model._guided_target_gate_logits = logits
    model._guided_target_gate_mode_assignment = assignments
    mask = torch.zeros(1, 1, 8, 8)
    mask[:, :, :4, :4] = 1.0

    loss, diag = guided_target_gate_supervision_loss(model, mask)
    loss.backward()

    assert loss.item() > 0.0
    assert diag["target_gate_hard_negative"] == 1.0
    assert model.guided_target_mode_gate.positive_support.sum() == 1
    assert model.guided_target_mode_gate.negative_support.sum() == 3


def test_explicit_aggregation_alpha_decouples_gate_from_native_gather_alpha() -> None:
    model = torch.nn.Module()
    config = GuidedPrototypeConfig(native_anchor_alpha=0.0, aggregation_alpha=1.0)

    assert _guided_aggregation_alpha(model, config) == 1.0


def test_aggregation_alpha_defaults_to_guided_gather_alpha() -> None:
    model = torch.nn.Module()
    config = GuidedPrototypeConfig(native_anchor_alpha=0.25)

    assert _guided_aggregation_alpha(model, config) == 0.25


def test_guided_distillation_adds_to_full_native_coherence() -> None:
    native = torch.tensor(0.8, requires_grad=True)
    guided = torch.tensor(0.1, requires_grad=True)

    combined = _combine_guided_gather_losses(
        native,
        guided,
        alpha=0.0,
        distill_weight=0.1,
    )
    combined.backward()

    assert torch.allclose(combined, torch.tensor(0.81))
    assert torch.allclose(native.grad, torch.tensor(1.0))
    assert torch.allclose(guided.grad, torch.tensor(0.1))


def test_semantic_coverage_matches_supported_roles_without_fixed_slots() -> None:
    tokens = torch.tensor(
        [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]]
    )
    priors = torch.tensor(
        [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]]
    )
    prototypes = torch.tensor(
        [[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0],
          [0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]],
        requires_grad=True,
    )

    loss, diag = _semantic_prototype_coverage_loss(
        tokens,
        prototypes,
        priors,
        torch.zeros(1, 6),
        min_role_mass=0.05,
    )

    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)
    assert diag["active_roles"] == 3.0
    assert diag["matched_similarity_mean"] == pytest.approx(1.0)
    assert diag["matched_slot_fraction"] == pytest.approx(0.5)


def test_semantic_coverage_uses_risk_to_exclude_unsafe_anchor_content() -> None:
    tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    priors = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    prototypes = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]], requires_grad=True
    )

    loss, diag = _semantic_prototype_coverage_loss(
        tokens,
        prototypes,
        priors,
        torch.tensor([[0.0, 1.0]]),
        min_role_mass=0.05,
    )

    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)
    assert diag["active_roles"] == 1.0
    assert diag["role_active_1"] == 0.0
    assert diag["role_active_2"] == 0.0


def test_semantic_coverage_backpropagates_only_through_selected_prototypes() -> None:
    tokens = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]],
        requires_grad=True,
    )
    priors = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]],
        requires_grad=True,
    )
    prototypes = torch.tensor(
        [[[0.8, 0.2], [0.2, 0.8], [-1.0, 0.0]]], requires_grad=True
    )

    loss, _ = _semantic_prototype_coverage_loss(
        tokens,
        prototypes,
        priors,
        torch.zeros(1, 4),
        min_role_mass=0.05,
    )
    loss.backward()

    assert prototypes.grad is not None
    assert prototypes.grad[:2].abs().sum() > 0.0
    assert prototypes.grad[0, 2].abs().sum() == 0.0
    assert tokens.grad is None
    assert priors.grad is None


def test_selective_semantic_coverage_excludes_object_role_and_stops_at_margin() -> None:
    tokens = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
    )
    priors = torch.eye(3).unsqueeze(0)
    prototypes = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]]],
        requires_grad=True,
    )

    loss, diag = _semantic_prototype_coverage_loss(
        tokens,
        prototypes,
        priors,
        torch.zeros(1, 3),
        min_role_mass=0.05,
        variant="selective_hinge",
        selected_roles=(0, 1),
        margin=0.75,
        min_confidence=0.5,
        max_risk=0.25,
    )

    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)
    assert diag["active_roles"] == 2.0
    assert diag["role_active_2"] == 0.0


def test_selective_semantic_coverage_filters_high_risk_tokens() -> None:
    tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    priors = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    prototypes = torch.tensor(
        [[[0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]], requires_grad=True
    )

    loss, diag = _semantic_prototype_coverage_loss(
        tokens,
        prototypes,
        priors,
        torch.tensor([[0.0, 0.9]]),
        min_role_mass=0.05,
        variant="selective_hinge",
        selected_roles=(0,),
        margin=0.8,
        min_confidence=0.5,
        max_risk=0.25,
    )

    assert loss.item() > 0.0
    assert diag["active_roles"] == 1.0


def test_familiarity_read_only_still_updates_normal_memory() -> None:
    config = GuidedPrototypeConfig(
        aggregation_familiarity_gate=True,
        familiarity_write_enabled=False,
    )

    assert prototype_guidance._familiarity_memory_update_enabled(
        config,
        training=True,
        update_prior_stats=True,
    )
    assert not prototype_guidance._familiarity_memory_update_enabled(
        config,
        training=False,
        update_prior_stats=True,
    )
    assert not prototype_guidance._familiarity_memory_update_enabled(
        config,
        training=True,
        update_prior_stats=False,
    )


def test_decoder_object_read_gate_only_suppresses_object_prototypes() -> None:
    attention = torch.ones(1, 2, 3, 6)
    risk = torch.tensor([[0.0, 0.5, 1.0]])

    gated = _apply_decoder_read_gate(
        attention,
        risk,
        groups=(2, 2, 2),
        strength=1.0,
        scope="object",
    )

    assert torch.equal(gated[..., :4], attention[..., :4])
    assert torch.allclose(gated[:, :, :, 4:], torch.tensor([1.0, 0.5, 0.0])[None, None, :, None])


def test_decoder_prototypewise_risk_suppresses_object_modes_independently() -> None:
    attention = torch.ones(1, 1, 2, 6)
    risk = torch.zeros(1, 2, 6)
    risk[0, 0, 4:] = torch.tensor([1.0, 0.0])
    risk[0, 1, 4:] = torch.tensor([0.25, 0.75])

    gated = _apply_decoder_read_gate(
        attention,
        risk,
        groups=(2, 2, 2),
        strength=1.0,
        scope="object",
    )

    assert torch.equal(gated[..., :4], attention[..., :4])
    assert torch.allclose(gated[0, 0, 0, 4:], torch.tensor([0.0, 1.0]))
    assert torch.allclose(gated[0, 0, 1, 4:], torch.tensor([0.75, 0.25]))


def test_decoder_adaptive_strength_interpolates_mild_to_full_by_risk() -> None:
    attention = torch.ones(1, 1, 3, 6)
    risk = torch.tensor([[0.0, 0.5, 1.0]])

    gated = _apply_decoder_read_gate(
        attention,
        risk,
        groups=(2, 2, 2),
        strength=0.5,
        scope="object",
        adaptive_strength_power=1.0,
    )

    assert torch.equal(gated[..., :4], attention[..., :4])
    assert torch.allclose(
        gated[0, 0, :, 4:],
        torch.tensor([[1.0, 1.0], [0.625, 0.625], [0.0, 0.0]]),
    )


def test_center6_decoder_risk_uses_the_selected_source_and_prototype_shape() -> None:
    model = torch.nn.Module()
    centers = torch.eye(6)
    model.guided_center6_teacher = FrozenCenter6Teacher(
        centers,
        torch.ones(6),
        torch.ones(6, dtype=torch.long),
        groups=(2, 2, 2),
    )
    query = centers[4].reshape(1, 1, 6).repeat(1, 4, 1)
    novelty = torch.full((1, 4), 0.5)
    aggregation_config = GuidedPrototypeConfig(
        center6_balanced=True,
        decoder_read_risk_source="aggregation",
    )

    scalar_risk, objectness = _build_decoder_read_risk(
        model,
        query,
        None,
        aggregation_config,
        novelty,
    )

    assert scalar_risk.shape == (1, 4)
    assert torch.allclose(scalar_risk, 0.5 * objectness)

    hybrid_config = GuidedPrototypeConfig(
        center6_balanced=True,
        decoder_read_risk_source="center6_hybrid",
        decoder_read_prototypewise=True,
    )
    prototype_risk, physical_objectness = _build_decoder_read_risk(
        model,
        query,
        torch.zeros(1, 3, 14, 14),
        hybrid_config,
        novelty,
    )

    assert prototype_risk.shape == (1, 4, 6)
    assert torch.equal(prototype_risk[..., :4], torch.zeros_like(prototype_risk[..., :4]))
    assert bool(torch.isfinite(prototype_risk).all())
    assert physical_objectness.shape == (1, 4)

    radius_config = GuidedPrototypeConfig(
        center6_balanced=True,
        decoder_read_risk_source="center6_radius",
        decoder_read_prototypewise=True,
    )
    radius_risk, radius_objectness = _build_decoder_read_risk(
        model,
        query,
        None,
        radius_config,
        novelty,
    )

    expected_violation = torch.sigmoid(
        (_center6_teacher_distance_probability(query, model.guided_center6_teacher, temperature=0.1)[0] - 1.0)
        / 0.1
    )
    assert radius_risk.shape == (1, 4, 6)
    assert torch.allclose(radius_risk, 0.5 * expected_violation)
    assert radius_objectness.shape == (1, 4)

    global_config = GuidedPrototypeConfig(
        center6_balanced=True,
        decoder_read_risk_source="center6_global",
        decoder_read_center6_support_alpha=0.25,
    )
    global_risk, global_objectness = _build_decoder_read_risk(
        model,
        query,
        None,
        global_config,
        novelty,
    )
    expected_nearest = expected_violation.min(dim=-1).values
    assert global_risk.shape == (1, 4)
    assert torch.allclose(global_risk, 0.5 * (0.25 + 0.75 * expected_nearest))
    assert torch.allclose(global_objectness, radius_objectness)

    conditioned_config = GuidedPrototypeConfig(
        center6_balanced=True,
        decoder_read_risk_source="center6_mode_novelty",
        decoder_read_center6_mode_tail_thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        decoder_read_center6_mode_tail_uppers=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
    )
    conditioned_risk, conditioned_objectness = _build_decoder_read_risk(
        model,
        query,
        None,
        conditioned_config,
        novelty,
    )
    assert conditioned_risk.shape == (1, 4)
    assert torch.equal(conditioned_risk, torch.zeros_like(conditioned_risk))
    assert torch.allclose(conditioned_objectness, radius_objectness)


def test_decoder_prototypewise_tail_gate_can_suppress_all_six_modes() -> None:
    attention = torch.ones(1, 1, 1, 6)
    risk = torch.tensor([[[0.1, 0.2, 0.3, 0.4, 0.5, 0.9]]])

    gated = _apply_decoder_read_gate(
        attention,
        risk,
        groups=(2, 2, 2),
        strength=1.0,
        scope="all",
        mode="tail_suppress",
        tail_threshold=0.2,
        tail_upper=0.8,
    )

    assert torch.allclose(
        gated[0, 0, 0],
        torch.tensor([1.0, 1.0, 5.0 / 6.0, 2.0 / 3.0, 0.5, 0.0]),
    )


def test_decoder_all_read_gate_is_an_unrenormalized_token_scale() -> None:
    attention = torch.arange(1, 13, dtype=torch.float32).reshape(1, 1, 2, 6)
    risk = torch.tensor([[0.0, 1.0]])

    gated = _apply_decoder_read_gate(
        attention,
        risk,
        groups=(2, 2, 2),
        strength=0.5,
        scope="all",
    )

    assert torch.equal(gated[:, :, 0], attention[:, :, 0])
    assert torch.allclose(gated[:, :, 1], 0.5 * attention[:, :, 1])


def test_selective_read_route_only_acts_above_normal_tail_and_preserves_mass() -> None:
    attention = torch.tensor(
        [[[[1.0, 3.0, 2.0, 2.0, 4.0, 2.0], [2.0, 1.0, 1.0, 2.0, 3.0, 1.0]]]]
    )
    risk = torch.tensor([[0.19, 0.80]])

    routed = _apply_decoder_read_gate(
        attention,
        risk,
        groups=(2, 2, 2),
        strength=1.0,
        scope="object",
        mode="selective_route",
        tail_threshold=0.20,
        tail_upper=0.80,
    )

    assert torch.equal(routed[:, :, 0], attention[:, :, 0])
    assert torch.equal(routed[:, :, 1, 4:], torch.zeros_like(routed[:, :, 1, 4:]))
    assert torch.allclose(
        routed[:, :, 1, :4],
        torch.tensor([10.0 / 3.0, 5.0 / 3.0, 5.0 / 3.0, 10.0 / 3.0]),
    )
    assert torch.allclose(routed.sum(dim=-1), attention.sum(dim=-1))


def test_selective_read_route_uses_uniform_fallback_for_zero_non_object_mass() -> None:
    attention = torch.tensor([[[[0.0, 0.0, 0.0, 0.0, 3.0, 1.0]]]])

    routed = _apply_decoder_read_gate(
        attention,
        torch.ones(1, 1),
        groups=(2, 2, 2),
        strength=0.5,
        scope="object",
        mode="selective_route",
        tail_threshold=0.5,
        tail_upper=1.0,
    )

    assert torch.allclose(routed[..., :4], torch.full_like(routed[..., :4], 0.5))
    assert torch.allclose(routed[..., 4:], torch.tensor([1.5, 0.5]))
    assert torch.allclose(routed.sum(dim=-1), attention.sum(dim=-1))


def test_tail_suppress_uses_normal_tail_without_redistributing_mass() -> None:
    attention = torch.ones(1, 1, 3, 6)
    risk = torch.tensor([[0.19, 0.50, 0.80]])

    gated = _apply_decoder_read_gate(
        attention,
        risk,
        groups=(2, 2, 2),
        strength=0.5,
        scope="object",
        mode="tail_suppress",
        tail_threshold=0.20,
        tail_upper=0.80,
    )

    assert torch.equal(gated[..., :4], attention[..., :4])
    assert torch.allclose(
        gated[..., 4:],
        torch.tensor([1.0, 0.75, 0.5])[None, None, :, None],
    )
    assert torch.all(gated.sum(dim=-1) <= attention.sum(dim=-1))


def test_attention_aware_read_gate_preserves_heads_with_low_object_share() -> None:
    attention = torch.tensor(
        [[[[1.0, 1.0, 1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]]]
    )
    gated = _apply_decoder_read_gate(
        attention,
        torch.ones(1, 2),
        groups=(2, 2, 2),
        strength=1.0,
        scope="object",
        attention_aware=True,
    )

    assert torch.equal(gated[:, :, 0], attention[:, :, 0])
    assert torch.equal(gated[:, :, 1, :4], attention[:, :, 1, :4])
    assert torch.equal(gated[:, :, 1, 4:], torch.zeros_like(gated[:, :, 1, 4:]))


def test_attention_aware_read_gate_normalizes_share_by_group_size() -> None:
    attention = torch.tensor([[[[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]]]])
    gated = _apply_decoder_read_gate(
        attention,
        torch.ones(1, 1),
        groups=(2, 2, 2),
        strength=1.0,
        scope="object",
        attention_aware=True,
    )

    # Object share is 0.2 versus the uniform group prior 1/3, hence the
    # effective activation is 0.6 and 40% of the object read remains.
    assert torch.allclose(gated[..., 4:], torch.full_like(gated[..., 4:], 0.2))


def test_responsibility_aware_read_gate_suppresses_dominant_object_slot_more() -> None:
    attention = torch.tensor([[[[1.0, 1.0, 1.0, 1.0, 3.0, 1.0]]]])

    gated = _apply_decoder_read_gate(
        attention,
        torch.ones(1, 1),
        groups=(2, 2, 2),
        strength=1.0,
        scope="object",
        responsibility_aware=True,
    )

    assert torch.equal(gated[..., :4], attention[..., :4])
    assert torch.allclose(gated[..., 4:], torch.tensor([[[[0.75, 0.75]]]]))


def test_responsibility_aware_read_gate_rejects_prototypewise_risk() -> None:
    with pytest.raises(ValueError, match="scalar token risk"):
        _apply_decoder_read_gate(
            torch.ones(1, 1, 1, 6),
            torch.ones(1, 1, 6),
            groups=(2, 2, 2),
            strength=1.0,
            scope="object",
            responsibility_aware=True,
        )


def test_decoder_removed_update_cap_limits_vector_norm() -> None:
    ungated = torch.tensor([[[3.0, 4.0]]])
    gated = torch.zeros_like(ungated)

    capped = _cap_removed_decoder_update(ungated, gated, cap=0.5)

    removed = ungated - capped
    assert torch.allclose(removed.norm(dim=-1), 0.5 * ungated.norm(dim=-1))
    assert torch.allclose(capped, 0.5 * ungated)


def test_decoder_late4_gate_selects_only_suffix() -> None:
    selected = [
        index
        for index in range(8)
        if _decoder_read_layer_enabled(index, layer_count=8, start=4)
    ]

    assert selected == [4, 5, 6, 7]


def _normalized_image(size: int) -> torch.Tensor:
    axis = torch.linspace(0.0, 1.0, size)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    rgb = torch.stack([xx, yy, 0.5 * (xx + yy)], dim=0).unsqueeze(0)
    mean = rgb.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = rgb.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (rgb - mean) / std


def _feature_tokens(side: int, channels: int = 12) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, side)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    fields = [
        xx,
        yy,
        xx * yy,
        xx.square(),
        yy.square(),
        torch.sin(torch.pi * xx),
        torch.cos(torch.pi * yy),
        torch.sin(torch.pi * (xx + yy)),
        torch.cos(torch.pi * (xx - yy)),
        (xx.square() + yy.square()).sqrt(),
        torch.ones_like(xx),
        torch.zeros_like(xx),
    ][:channels]
    feature = torch.stack(fields, dim=0).unsqueeze(0)
    return feature.flatten(2).transpose(1, 2).contiguous()


def test_fixed_prior_uses_a_scale_consistent_reference_grid() -> None:
    config = GuidedPrototypeConfig(
        scale_consistent=True,
        prior_reference_side=32,
        image_reference_size=448,
    )
    tokens_448 = _feature_tokens(32)
    feature_448 = tokens_448.transpose(1, 2).reshape(1, -1, 32, 32)
    feature_672 = F.interpolate(feature_448, size=(48, 48), mode="bilinear", align_corners=False)
    tokens_672 = feature_672.flatten(2).transpose(1, 2).contiguous()
    image_448 = _normalized_image(448)
    image_672 = F.interpolate(image_448, size=(672, 672), mode="bilinear", align_corners=False)

    prior_448 = _build_priors(tokens_448, config, image_448)
    prior_672 = _build_priors(tokens_672, config, image_672)

    prior_448_map = prior_448.transpose(1, 2).reshape(1, 3, 32, 32)
    prior_672_map = prior_672.transpose(1, 2).reshape(1, 3, 48, 48)
    prior_672_map = F.interpolate(prior_672_map, size=(32, 32), mode="bilinear", align_corners=False)

    difference = (prior_448_map - prior_672_map).abs()
    assert float(difference.mean()) < 0.025
    assert torch.allclose(
        prior_448_map.mean(dim=(0, 2, 3)),
        prior_672_map.mean(dim=(0, 2, 3)),
        atol=0.01,
        rtol=0.03,
    )


def test_image_prior_does_not_create_padding_edges_on_a_constant_image() -> None:
    rgb = torch.full((1, 3, 448, 448), 0.4)
    mean = rgb.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = rgb.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    normalized = (rgb - mean) / std

    edge, texture = _image_edge_texture(
        normalized,
        side=32,
        device=normalized.device,
        reference_size=448,
        physical_rgb=True,
        replicate_padding=True,
    )

    assert edge is not None and texture is not None
    assert float(edge.max()) < 1e-5
    assert float(texture.max()) < 1e-5


def test_trainable_prior_head_uses_the_same_reference_grid() -> None:
    torch.manual_seed(7)
    model = torch.nn.Module()
    model.guided_prior_head = PrototypePriorHead(dim=12, hidden_dim=8)
    torch.nn.init.normal_(model.guided_prior_head.net[-1].weight, std=0.2)
    torch.nn.init.normal_(model.guided_prior_head.net[-1].bias, std=0.05)
    config = GuidedPrototypeConfig(
        trainable_prior=True,
        scale_consistent=True,
        prior_reference_side=32,
        image_reference_size=448,
    )

    tokens_448 = _feature_tokens(32)
    feature_448 = tokens_448.transpose(1, 2).reshape(1, -1, 32, 32)
    feature_672 = F.interpolate(feature_448, size=(48, 48), mode="bilinear", align_corners=False)
    tokens_672 = feature_672.flatten(2).transpose(1, 2).contiguous()
    image_448 = _normalized_image(448)
    image_672 = F.interpolate(image_448, size=(672, 672), mode="bilinear", align_corners=False)

    prior_448, _ = _build_trainable_priors(model, tokens_448, config, image_448)
    prior_672, _ = _build_trainable_priors(model, tokens_672, config, image_672)
    prior_448_map = prior_448.transpose(1, 2).reshape(1, 3, 32, 32)
    prior_672_map = prior_672.transpose(1, 2).reshape(1, 3, 48, 48)
    prior_672_map = F.interpolate(prior_672_map, size=(32, 32), mode="bilinear", align_corners=False)

    difference = (prior_448_map - prior_672_map).abs()
    assert float(difference.mean()) < 0.025
    assert torch.allclose(
        prior_448_map.mean(dim=(0, 2, 3)),
        prior_672_map.mean(dim=(0, 2, 3)),
        atol=0.01,
        rtol=0.03,
    )


def test_group_weights_calibrate_fixed_prior_channels() -> None:
    tokens = _feature_tokens(48)
    image = _normalized_image(672)
    uncalibrated = _build_priors(
        tokens,
        GuidedPrototypeConfig(group_weights=(1.0, 1.0, 1.0)),
        image,
    ).mean(dim=(0, 1))
    calibrated = _build_priors(
        tokens,
        GuidedPrototypeConfig(group_weights=(0.92, 1.10, 1.10)),
        image,
    ).mean(dim=(0, 1))

    assert calibrated[0] < uncalibrated[0]
    assert calibrated[1] > uncalibrated[1]
    assert calibrated[2] > uncalibrated[2]
    assert torch.allclose(calibrated.sum(), calibrated.new_tensor(1.0), atol=1e-6)


def test_multiscale_direct_prior_stays_on_the_current_grid() -> None:
    tokens = _feature_tokens(48)
    image = _normalized_image(672)
    config = GuidedPrototypeConfig(multiscale_direct=True)

    prior = _build_priors(tokens, config, image)

    assert prior.shape == (1, 48 * 48, 3)
    assert torch.isfinite(prior).all()
    assert torch.allclose(prior.sum(dim=-1), torch.ones_like(prior[..., 0]), atol=1e-6)


def test_feature_only_prior_skips_image_gradient_branch(monkeypatch) -> None:
    tokens = _feature_tokens(32)
    image = _normalized_image(448)
    config = GuidedPrototypeConfig(
        image_texture_weight=0.0,
        image_object_weight=0.0,
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("feature-only prior unexpectedly computed image gradients")

    monkeypatch.setattr(
        prototype_guidance,
        "_image_edge_texture",
        fail_if_called,
    )
    prior = _build_priors(tokens, config, image)

    assert prior.shape == (1, 32 * 32, 3)
    assert torch.isfinite(prior).all()
    assert torch.allclose(
        prior.sum(dim=-1),
        torch.ones_like(prior[..., 0]),
        atol=1e-6,
    )


def test_adaptive_objectness_override_replaces_fixed_object_scales() -> None:
    tokens = _feature_tokens(32)
    image = _normalized_image(448)
    override = torch.linspace(0.0, 1.0, 32 * 32).reshape(1, -1)
    first = GuidedPrototypeConfig(
        multiscale_direct=True,
        object_kernel_weights=(1.0, 0.0, 0.0),
    )
    second = GuidedPrototypeConfig(
        multiscale_direct=True,
        object_kernel_weights=(0.0, 0.0, 1.0),
    )

    prior_first = _build_priors(tokens, first, image, objectness_override=override)
    prior_second = _build_priors(tokens, second, image, objectness_override=override)

    assert torch.allclose(prior_first, prior_second, atol=1e-6)


def test_free_prototype_gather_is_exactly_native() -> None:
    torch.manual_seed(11)
    model = torch.nn.Module()
    model._guided_prototype_config = GuidedPrototypeConfig(
        context_transport=True,
        context_adaptive_scale=True,
        free_prototypes=True,
    )
    query = F.normalize(torch.randn(2, 9, 8), dim=-1)
    keys = F.normalize(torch.randn(2, 6, 8), dim=-1)
    expected = (
        1.0
        - F.cosine_similarity(query.unsqueeze(2), keys.unsqueeze(1), dim=-1)
    ).min(dim=2).values.mean()

    actual = _guided_gather_loss(model, query, keys)

    assert torch.allclose(actual, expected, atol=1e-7)


def test_integer_spatial_scaling_uses_448_as_the_reference_grid() -> None:
    config = GuidedPrototypeConfig(
        multiscale_direct=True,
        spatial_scale_mode="integer",
        spatial_reference_side=32,
        object_kernels=(3, 5, 7),
        texture_offsets=(1, 2),
    )

    kernels_448, offsets_448, scale_448 = resolve_spatial_prior_parameters(config, token_side=32)
    kernels_672, offsets_672, scale_672 = resolve_spatial_prior_parameters(config, token_side=48)

    assert kernels_448 == (3, 5, 7)
    assert offsets_448 == (1, 2)
    assert scale_448 == 1.0
    assert kernels_672 == (5, 7, 11)
    assert offsets_672 == (2, 3)
    assert scale_672 == 1.5


def test_legacy_spatial_scaling_keeps_configured_windows_at_672() -> None:
    config = GuidedPrototypeConfig(
        multiscale_direct=True,
        spatial_scale_mode="legacy",
        spatial_reference_side=32,
        object_kernels=(3, 5, 7),
        texture_offsets=(1, 2),
    )

    kernels, offsets, scale = resolve_spatial_prior_parameters(config, token_side=48)

    assert kernels == (3, 5, 7)
    assert offsets == (1, 2)
    assert scale == 1.0


def test_robust_prior_normalizer_does_not_stretch_each_image_independently() -> None:
    normalizer = RobustPriorNormalizer(low=0.5, high=0.99, momentum=0.0)
    weak = torch.linspace(0.0, 0.1, 64).reshape(1, 1, 8, 8)
    strong = torch.linspace(0.0, 1.0, 64).reshape(1, 1, 8, 8)
    values = torch.cat([weak, strong], dim=0)

    normalized = normalizer.normalize(values, "feature_object", update=True)

    assert float(normalized[0].max()) < 0.2
    assert float(normalized[1].max()) > 0.9
    assert bool(normalizer.initialized[2])


def test_robust_prior_normalizer_can_freeze_normal_statistics() -> None:
    normalizer = RobustPriorNormalizer(low=0.5, high=0.99, momentum=0.5)
    normal = torch.linspace(0.0, 1.0, 64).reshape(1, 1, 8, 8)
    normalizer.normalize(normal, "feature_texture", update=True)
    low = normalizer.low.clone()
    high = normalizer.high.clone()

    shifted = normal + 10.0
    normalizer.normalize(shifted, "feature_texture", update=False)

    assert torch.equal(normalizer.low, low)
    assert torch.equal(normalizer.high, high)


def test_group_contrastive_loss_rewards_the_prior_selected_group() -> None:
    priors = torch.tensor(
        [[[0.90, 0.05, 0.05], [0.05, 0.90, 0.05], [0.05, 0.05, 0.90]]],
        dtype=torch.float32,
    )
    correct_distance = torch.tensor(
        [[
            [0.05, 0.10, 0.80, 0.85, 0.90, 0.95],
            [0.80, 0.85, 0.05, 0.10, 0.90, 0.95],
            [0.90, 0.95, 0.80, 0.85, 0.05, 0.10],
        ]],
        requires_grad=True,
    )
    wrong_distance = correct_distance.detach().roll(shifts=2, dims=2)

    correct_loss, target, confidence, valid = _group_contrastive_loss(
        correct_distance, priors, (2, 2, 2), temperature=0.1, confidence_margin=0.15
    )
    wrong_loss, _, _, _ = _group_contrastive_loss(
        wrong_distance, priors, (2, 2, 2), temperature=0.1, confidence_margin=0.15
    )
    correct_loss.backward()

    assert float(correct_loss) < float(wrong_loss)
    assert target.tolist() == [[0, 1, 2]]
    assert valid.all()
    assert torch.all(confidence > 0.8)
    assert correct_distance.grad is not None


def test_intra_group_balance_penalizes_a_single_winner() -> None:
    target = torch.zeros(1, 4, dtype=torch.long)
    confidence = torch.ones(1, 4)
    valid = torch.ones(1, 4, dtype=torch.bool)
    collapsed = torch.tensor(
        [[[0.0, 1.0, 2.0, 2.0, 2.0, 2.0]] * 4],
        requires_grad=True,
    )
    balanced = torch.tensor(
        [[
            [0.0, 1.0, 2.0, 2.0, 2.0, 2.0],
            [1.0, 0.0, 2.0, 2.0, 2.0, 2.0],
            [0.0, 1.0, 2.0, 2.0, 2.0, 2.0],
            [1.0, 0.0, 2.0, 2.0, 2.0, 2.0],
        ]],
    )

    collapsed_loss, collapsed_min_usage = _intra_group_balance_loss(
        collapsed, target, confidence, valid, (2, 2, 2), temperature=0.1
    )
    balanced_loss, balanced_min_usage = _intra_group_balance_loss(
        balanced, target, confidence, valid, (2, 2, 2), temperature=0.1
    )
    collapsed_loss.backward()

    assert float(collapsed_loss) > float(balanced_loss) + 0.5
    assert float(collapsed_min_usage) < 1e-3
    assert torch.allclose(balanced_min_usage, balanced_min_usage.new_tensor(0.5), atol=1e-4)
    assert collapsed.grad is not None


def test_intra_group_repulsion_penalizes_redundant_dynamic_prototypes() -> None:
    collapsed = F.normalize(
        torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]]),
        dim=-1,
    ).requires_grad_()
    separated = F.normalize(
        torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0], [1.0, 0.0], [-1.0, 0.0]]]),
        dim=-1,
    )

    collapsed_loss, collapsed_similarity = _intra_group_repulsion_loss(
        collapsed, (2, 2, 2), margin=0.2
    )
    separated_loss, separated_similarity = _intra_group_repulsion_loss(
        separated, (2, 2, 2), margin=0.2
    )
    collapsed_loss.backward()

    assert float(collapsed_loss) > float(separated_loss)
    assert float(collapsed_similarity) > float(separated_similarity)
    assert collapsed.grad is not None


def test_aggregation_gate_prevents_a_high_objectness_token_from_dominating() -> None:
    attention = torch.nn.Module()
    attention.num_heads = 1
    attention.scale = 1.0
    attention.q = torch.nn.Linear(2, 2, bias=False)
    attention.kv = torch.nn.Linear(2, 4, bias=False)
    attention.attn_drop = torch.nn.Identity()
    attention.proj = torch.nn.Identity()
    attention.proj_drop = torch.nn.Identity()
    with torch.no_grad():
        attention.q.weight.copy_(torch.eye(2))
        attention.kv.weight.copy_(torch.cat([torch.eye(2), torch.eye(2)], dim=0))

    prototype = torch.tensor([[[1.0, 0.0]]])
    tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    ungated = _aggregation_attention_with_gate(
        attention, prototype, tokens, torch.ones(1, 2)
    )
    gated = _aggregation_attention_with_gate(
        attention, prototype, tokens, torch.tensor([[0.05, 1.0]])
    )

    assert float(gated[0, 0, 0]) < float(ungated[0, 0, 0])
    assert float(gated[0, 0, 1]) > float(ungated[0, 0, 1])


def test_hard_roi_mask_assigns_exactly_zero_attention_outside_roi() -> None:
    attention = torch.nn.Module()
    attention.num_heads = 1
    attention.scale = 1.0
    attention.q = torch.nn.Linear(2, 2, bias=False)
    attention.kv = torch.nn.Linear(2, 4, bias=False)
    attention.attn_drop = torch.nn.Identity()
    attention.proj = torch.nn.Identity()
    attention.proj_drop = torch.nn.Identity()
    prototype = torch.randn(2, 3, 2)
    tokens = torch.randn(2, 4, 2)
    gate = torch.tensor([[0.4, 0.8, 0.2, 1.0], [1.0, 0.3, 0.7, 0.5]])
    valid = torch.tensor([[True, False, True, False], [False, True, True, True]])

    _, probabilities = _aggregation_attention_with_gate(
        attention,
        prototype,
        tokens,
        gate,
        valid_token_mask=valid,
        return_attention=True,
    )

    expanded_invalid = (~valid)[:, None, None, :].expand_as(probabilities)
    assert torch.count_nonzero(probabilities[expanded_invalid]) == 0
    assert torch.allclose(
        probabilities.sum(dim=-1), torch.ones_like(probabilities.sum(dim=-1))
    )


def test_hard_roi_mask_is_independent_of_prototype_specific_risk_gate() -> None:
    attention = torch.nn.Module()
    attention.num_heads = 1
    attention.scale = 1.0
    attention.q = torch.nn.Linear(2, 2, bias=False)
    attention.kv = torch.nn.Linear(2, 4, bias=False)
    attention.attn_drop = torch.nn.Identity()
    attention.proj = torch.nn.Identity()
    attention.proj_drop = torch.nn.Identity()
    prototype = torch.randn(1, 2, 2)
    tokens = torch.randn(1, 3, 2)
    gate = torch.tensor([[[1.0, 0.2, 0.8], [0.4, 1.0, 0.1]]])
    valid = torch.tensor([[True, False, True]])

    _, probabilities = _aggregation_attention_with_gate(
        attention,
        prototype,
        tokens,
        gate,
        valid_token_mask=valid,
        return_attention=True,
    )

    assert torch.count_nonzero(probabilities[..., 1]) == 0
    assert torch.all(probabilities[..., (0, 2)] > 0)


def test_hard_roi_mask_rejects_samples_without_valid_tokens() -> None:
    attention = torch.nn.Module()
    attention.num_heads = 1
    attention.scale = 1.0
    attention.q = torch.nn.Linear(2, 2, bias=False)
    attention.kv = torch.nn.Linear(2, 4, bias=False)
    attention.attn_drop = torch.nn.Identity()
    attention.proj = torch.nn.Identity()
    attention.proj_drop = torch.nn.Identity()

    with pytest.raises(ValueError, match="at least one valid ROI token"):
        _aggregation_attention_with_gate(
            attention,
            torch.randn(1, 2, 2),
            torch.randn(1, 3, 2),
            torch.ones(1, 3),
            valid_token_mask=torch.zeros(1, 3, dtype=torch.bool),
        )


def test_transition_roi_prior_interpolates_token_coverage() -> None:
    roi = torch.tensor(
        [[[[1.0, 1.0, 1.0, 0.0],
           [1.0, 1.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0]]]]
    )

    coverage = _token_roi_coverage(roi, batch=1, token_count=4)
    prior = _transition_token_roi_prior(
        roi,
        batch=1,
        token_count=4,
        floor=1e-6,
    )

    assert torch.allclose(coverage, torch.tensor([[1.0, 0.25, 0.0, 0.0]]))
    expected = 1e-6 + (1.0 - 1e-6) * coverage
    assert torch.allclose(prior, expected)
    assert prior[0, 0] == 1.0
    assert torch.isclose(prior[0, 2], torch.tensor(1e-6))


def test_transition_roi_prior_combines_multiplicatively_with_risk_gate() -> None:
    attention = torch.nn.Module()
    attention.num_heads = 1
    attention.scale = 1.0
    attention.q = torch.nn.Linear(2, 2, bias=False)
    attention.kv = torch.nn.Linear(2, 4, bias=False)
    attention.attn_drop = torch.nn.Identity()
    attention.proj = torch.nn.Identity()
    attention.proj_drop = torch.nn.Identity()
    with torch.no_grad():
        attention.q.weight.zero_()
        attention.kv.weight.copy_(torch.cat([torch.eye(2), torch.eye(2)], dim=0))

    prototype = torch.zeros(1, 1, 2)
    tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    risk_gate = torch.tensor([[0.5, 1.0]])
    transition_prior = torch.tensor([[1.0, 0.25]])
    combined = risk_gate * transition_prior

    _, probabilities = _aggregation_attention_with_gate(
        attention,
        prototype,
        tokens,
        combined,
        return_attention=True,
    )

    expected = combined / combined.sum(dim=-1, keepdim=True)
    assert torch.allclose(probabilities[0, 0, 0], expected[0], atol=1e-7)


def test_mode_routing_maps_each_physical_slot_to_its_teacher_mode() -> None:
    scalar = torch.tensor([[0.8, 0.6]])
    probability = torch.tensor([[[0.9, 0.1], [0.2, 0.8]]])
    slot_to_mode = torch.tensor([0, 0, 1])

    routed, factor = _mode_specific_routing_weights(
        scalar,
        probability,
        slot_to_mode,
        floor=0.1,
        strength=1.0,
    )

    expected_factor = torch.tensor(
        [[[0.91, 0.28], [0.91, 0.28], [0.19, 0.82]]]
    )
    assert torch.allclose(factor, expected_factor, atol=1e-7)
    assert torch.allclose(routed, scalar[:, None, :] * expected_factor, atol=1e-7)


def test_zero_strength_mode_routing_is_exactly_the_scalar_gate() -> None:
    scalar = torch.tensor([[0.8, 0.6]])
    probability = torch.tensor([[[0.9, 0.1], [0.2, 0.8]]])

    routed, factor = _mode_specific_routing_weights(
        scalar,
        probability,
        torch.tensor([0, 1]),
        floor=0.05,
        strength=0.0,
    )

    assert torch.equal(factor, torch.ones_like(factor))
    assert torch.equal(routed, scalar[:, None, :].expand(-1, 2, -1))


def test_expanded_unit_routing_preserves_scalar_attention_exactly() -> None:
    torch.manual_seed(17)
    attention = torch.nn.Module()
    attention.num_heads = 2
    attention.scale = 0.5
    attention.q = torch.nn.Linear(4, 4, bias=False)
    attention.kv = torch.nn.Linear(4, 8, bias=False)
    attention.attn_drop = torch.nn.Identity()
    attention.proj = torch.nn.Identity()
    attention.proj_drop = torch.nn.Identity()
    prototype = torch.randn(2, 3, 4)
    tokens = torch.randn(2, 5, 4)
    scalar = torch.rand(2, 5).clamp_min(0.1)
    expanded = scalar[:, None, :].expand(-1, 3, -1)

    scalar_output, scalar_attention = _aggregation_attention_with_gate(
        attention, prototype, tokens, scalar, return_attention=True
    )
    expanded_output, expanded_attention = _aggregation_attention_with_gate(
        attention, prototype, tokens, expanded, return_attention=True
    )

    assert torch.equal(scalar_attention, expanded_attention)
    assert torch.equal(scalar_output, expanded_output)


def test_prototype_specific_gate_changes_attention_and_keeps_token_gradients() -> None:
    attention = torch.nn.Module()
    attention.num_heads = 1
    attention.scale = 1.0
    attention.q = torch.nn.Linear(2, 2, bias=False)
    attention.kv = torch.nn.Linear(2, 4, bias=False)
    attention.attn_drop = torch.nn.Identity()
    attention.proj = torch.nn.Identity()
    attention.proj_drop = torch.nn.Identity()
    with torch.no_grad():
        attention.q.weight.zero_()
        attention.kv.weight.copy_(
            torch.cat([torch.zeros(2, 2), torch.eye(2)], dim=0)
        )

    prototype = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    tokens = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True
    )
    gate = torch.tensor([[[1.0, 0.05], [0.05, 1.0]]])

    output, probabilities = _aggregation_attention_with_gate(
        attention,
        prototype,
        tokens,
        gate,
        return_attention=True,
    )
    output.sum().backward()

    assert probabilities[0, 0, 0, 0] > 0.95
    assert probabilities[0, 0, 1, 1] > 0.95
    assert tokens.grad is not None
    assert torch.count_nonzero(tokens.grad) > 0


def test_mode_routing_reaches_end_to_end_aggregation_attention() -> None:
    class ZeroMLP(torch.nn.Module):
        def forward(self, value):
            return torch.zeros_like(value)

    attention = torch.nn.Module()
    attention.num_heads = 1
    attention.scale = 1.0
    attention.q = torch.nn.Linear(2, 2, bias=False)
    attention.kv = torch.nn.Linear(2, 4, bias=False)
    attention.attn_drop = torch.nn.Identity()
    attention.proj = torch.nn.Identity()
    attention.proj_drop = torch.nn.Identity()
    with torch.no_grad():
        attention.q.weight.zero_()
        attention.kv.weight.copy_(
            torch.cat([torch.zeros(2, 2), torch.eye(2)], dim=0)
        )
    block = torch.nn.Module()
    block.norm1 = torch.nn.Identity()
    block.norm2 = torch.nn.Identity()
    block.attn = attention
    block.drop_path = torch.nn.Identity()
    block.mlp = ZeroMLP()

    model = torch.nn.Module()
    model.prototype_token = torch.nn.Parameter(torch.eye(2))
    model.aggregation = torch.nn.ModuleList([block])
    model._guided_prototype_config = GuidedPrototypeConfig(
        groups=(1, 1, 0),
        center6_balanced=True,
        center6_teacher_temperature=0.01,
        aggregation_gate=True,
        mode_specific_routing=True,
        mode_routing_floor=0.01,
        mode_routing_strength=1.0,
    )
    model.guided_center6_teacher = FrozenCenter6Teacher(
        torch.eye(2),
        torch.ones(2),
        torch.ones(2, dtype=torch.long),
        groups=(1, 1, 0),
    )
    model._guided_capture_aggregation_attention = True
    tokens = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]],
        requires_grad=True,
    )

    output = _aggregate_guided_prototypes(
        model,
        tokens,
        torch.zeros(1, 3, 2, 2),
    )
    output.sum().backward()
    captured = model._guided_last_aggregation_attention[0][0, 0]

    assert captured[0, [0, 2]].sum() > 0.99
    assert captured[1, [1, 3]].sum() > 0.99
    assert model._guided_aggregation_diag["guided_mode_routing_active"] == 1.0
    assert model._guided_aggregation_diag["guided_mode_routing_factor_std"] > 0.0
    assert tokens.grad is not None
    assert torch.count_nonzero(tokens.grad) > 0

    routed_context = tokens.detach().flip(1).requires_grad_(True)
    context_output = _aggregate_guided_prototypes(
        model,
        tokens.detach(),
        torch.zeros(1, 3, 2, 2),
        prototype_context=routed_context,
    )
    context_output.sum().backward()
    context_attention = model._guided_last_aggregation_attention[0][0, 0]

    assert context_attention[0, [1, 3]].sum() > 0.99
    assert context_attention[1, [0, 2]].sum() > 0.99
    assert routed_context.grad is not None
    assert torch.count_nonzero(routed_context.grad) > 0


def test_mode_routing_cli_is_off_by_default_and_rejects_silent_noop() -> None:
    parser = argparse.ArgumentParser()
    add_guided_prototype_args(parser)
    default = guided_config_from_args(parser.parse_args([]))
    assert not default.mode_specific_routing

    enabled = guided_config_from_args(
        parser.parse_args(
            [
                "--guided-prototype-mode-routing",
                "--guided-prototype-center6-balanced",
                "--guided-prototype-aggregation-gate",
            ]
        )
    )
    assert enabled.mode_specific_routing
    assert enabled.mode_routing_floor == 0.05
    assert enabled.mode_routing_strength == 1.0

    missing_center = parser.parse_args(
        [
            "--guided-prototype",
            "--guided-prototype-mode-routing",
            "--guided-prototype-aggregation-gate",
        ]
    )
    with pytest.raises(ValueError, match="requires Center6"):
        # configure_guided_prototypes owns cross-feature validation because the
        # frozen teacher and physical slot count are model-level properties.
        model = torch.nn.Module()
        model.prototype_token = torch.nn.Parameter(torch.randn(6, 8))
        model.gather_loss = lambda *_args: torch.tensor(0.0)
        prototype_guidance.configure_guided_prototypes(
            model, missing_center, "inpformer"
        )


def test_normal_object_memory_protects_familiar_tokens_and_marks_novel_tokens() -> None:
    memory = NormalObjectTokenMemory(
        dim=2,
        size=4,
        tokens_per_image=2,
        min_count=2,
        novelty_floor=0.08,
        temperature=0.03,
    )
    normal = torch.tensor([[[1.0, 0.0], [0.9, 0.1]]])
    memory.update(normal, torch.tensor([[0.9, 0.8]]))

    tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    novelty, distance = memory.novelty(tokens)

    assert float(novelty[0, 0]) < 0.1
    assert float(novelty[0, 1]) > 0.9
    assert float(distance[0, 0]) < float(distance[0, 1])


def test_hard_familiarity_mapping_uses_only_the_calibrated_threshold() -> None:
    memory = NormalObjectTokenMemory(
        dim=2,
        size=2,
        tokens_per_image=2,
        min_count=2,
        novelty_floor=0.08,
        temperature=0.03,
        novelty_mapping="hard",
    )
    memory.bank.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    memory.count.fill_(2)
    memory.novelty_threshold.fill_(0.20)
    tokens = torch.tensor(
        [[[1.0, 0.0], [0.8, 0.6], [2.0**-0.5, 2.0**-0.5]]]
    )

    novelty, distance = memory.novelty(tokens)

    assert torch.equal(novelty, (distance >= 0.20).to(novelty.dtype))
    assert set(novelty.flatten().tolist()) <= {0.0, 1.0}


def test_normal_object_memory_stops_updating_at_capacity() -> None:
    memory = NormalObjectTokenMemory(
        dim=2,
        size=2,
        tokens_per_image=2,
        min_count=2,
        novelty_floor=0.08,
        temperature=0.03,
    )
    first = torch.tensor([[[1.0, 0.0], [0.9, 0.1]]])
    memory.update(first, torch.tensor([[0.9, 0.8]]))
    frozen = memory.bank.clone()
    memory.update(torch.tensor([[[0.0, 1.0], [0.1, 0.9]]]), torch.ones(1, 2))

    assert int(memory.count.item()) == 2
    assert torch.equal(memory.bank, frozen)


def test_normal_object_memory_can_normalize_distance_by_nearest_local_radius() -> None:
    memory = NormalObjectTokenMemory(
        dim=2,
        size=2,
        tokens_per_image=2,
        min_count=2,
        novelty_floor=0.08,
        temperature=0.03,
    )
    memory.bank.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    memory.count.fill_(2)
    tokens = torch.tensor([[[3.0**0.5 / 2.0, 0.5]]])

    _, raw_distance = memory.novelty(tokens)
    memory.local_radii.copy_(torch.tensor([0.5, 2.0]))
    memory.local_radius_reference.fill_(1.0)
    memory.local_radius_enabled.fill_(True)
    _, normalized_distance = memory.novelty(tokens)

    assert torch.allclose(normalized_distance, 2.0 * raw_distance, atol=1e-6)


def test_cross_group_memory_calibration_excludes_same_source_image() -> None:
    tokens = torch.tensor(
        [
            [[1.0, 0.0], [0.999, 0.01]],
            [[0.8, 0.6], [0.79, 0.61]],
        ]
    )
    objectness = torch.ones(2, 2)
    legacy = NormalObjectTokenMemory(
        dim=2,
        size=4,
        tokens_per_image=2,
        min_count=4,
        novelty_floor=0.001,
        temperature=0.03,
    )
    cross_group = NormalObjectTokenMemory(
        dim=2,
        size=4,
        tokens_per_image=2,
        min_count=4,
        novelty_floor=0.001,
        temperature=0.03,
        calibration_mode="cross_group",
    )

    legacy.update(tokens, objectness)
    cross_group.update(tokens, objectness, torch.tensor([10, 20]))

    assert float(cross_group.novelty_threshold) > float(legacy.novelty_threshold) + 0.1
    assert int(cross_group.calibration_group_count.item()) == 2
    assert int(cross_group.calibrated_count.item()) == 4


def test_cross_group_memory_waits_for_a_second_source_group() -> None:
    memory = NormalObjectTokenMemory(
        dim=2,
        size=6,
        tokens_per_image=2,
        min_count=4,
        novelty_floor=0.01,
        temperature=0.03,
        calibration_mode="cross_group",
    )
    first = torch.tensor(
        [
            [[1.0, 0.0], [0.99, 0.01]],
            [[0.98, 0.02], [0.97, 0.03]],
        ]
    )
    memory.update(first, torch.ones(2, 2), torch.tensor([7, 7]))
    assert int(memory.calibrated_count.item()) == 0

    memory.update(
        torch.tensor([[[0.0, 1.0], [0.01, 0.99]]]),
        torch.ones(1, 2),
        torch.tensor([8]),
    )
    assert int(memory.calibrated_count.item()) == 6
    assert int(memory.calibration_group_count.item()) == 2


def test_normal_object_memory_loads_legacy_state_strictly() -> None:
    legacy = NormalObjectTokenMemory(
        dim=2,
        size=4,
        tokens_per_image=2,
        min_count=2,
        novelty_floor=0.08,
        temperature=0.03,
    )
    state = legacy.state_dict()
    del state["bank_group_ids"]
    del state["calibrated_count"]
    del state["calibration_group_count"]
    del state["local_radii"]
    del state["local_radius_reference"]
    del state["local_radius_enabled"]

    restored = NormalObjectTokenMemory(
        dim=2,
        size=4,
        tokens_per_image=2,
        min_count=2,
        novelty_floor=0.08,
        temperature=0.03,
    )
    restored.load_state_dict(state, strict=True)


def test_cross_group_mode_rejects_a_full_legacy_memory() -> None:
    legacy = NormalObjectTokenMemory(
        dim=2,
        size=2,
        tokens_per_image=2,
        min_count=2,
        novelty_floor=0.08,
        temperature=0.03,
    )
    legacy.update(torch.tensor([[[1.0, 0.0], [0.9, 0.1]]]), torch.ones(1, 2))
    state = legacy.state_dict()
    del state["bank_group_ids"]
    del state["calibrated_count"]
    del state["calibration_group_count"]
    cross_group = NormalObjectTokenMemory(
        dim=2,
        size=2,
        tokens_per_image=2,
        min_count=2,
        novelty_floor=0.08,
        temperature=0.03,
        calibration_mode="cross_group",
    )
    cross_group.load_state_dict(state, strict=True)

    try:
        cross_group.update(
            torch.tensor([[[0.0, 1.0], [0.1, 0.9]]]),
            torch.ones(1, 2),
            torch.tensor([2]),
        )
    except RuntimeError as exc:
        assert "Rebuild the memory" in str(exc)
    else:
        raise AssertionError("A full legacy memory must not silently enable cross-group calibration.")


def test_source_group_resolver_uses_manifest_and_crop_source() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        images = root / "images"
        images.mkdir()
        (root / "index.csv").write_text(
            "patch_file,source_file\nimages/patch.jpg,source_a.jpg\n",
            encoding="utf-8",
        )
        resolver = SourceGroupResolver()

        assert resolver.resolve(images / "patch.jpg") == "source_a.jpg"
        assert resolver.resolve(f"{root / 'source_b.jpg'}:16:32") == str(
            (root / "source_b.jpg").resolve()
        )


def test_native_anchor_selects_largest_normal_safe_alpha() -> None:
    selected = select_safe_alpha(
        native_q99=[0.10, 0.20],
        candidate_q99={
            0.25: [0.101, 0.198],
            0.50: [0.102, 0.204],
            0.75: [0.110, 0.201],
        },
        relative_tolerance=0.03,
    )

    assert selected == 0.50


def test_memory_mode_supervision_distinguishes_prototypes_inside_each_group() -> None:
    groups = (2, 2, 2)
    centers = torch.eye(6)
    teacher = FrozenMemoryModeTeacher(centers, groups)
    query = centers.unsqueeze(0)
    aligned_keys = centers.unsqueeze(0)
    swapped_keys = centers[torch.tensor([1, 0, 3, 2, 5, 4])].unsqueeze(0)
    target = torch.tensor([[0, 0, 1, 1, 2, 2]])
    confidence = torch.ones_like(target, dtype=torch.float32)
    valid = torch.ones_like(target, dtype=torch.bool)

    def loss_for(keys: torch.Tensor):
        distribution = 1.0 - F.cosine_similarity(
            query.unsqueeze(2), keys.unsqueeze(1), dim=-1
        )
        return _memory_mode_supervision_loss(
            query,
            distribution,
            target,
            confidence,
            valid,
            teacher,
            groups,
            temperature=0.1,
        )

    aligned_loss, accuracy, min_usage = loss_for(aligned_keys)
    swapped_loss, _, _ = loss_for(swapped_keys)

    assert float(aligned_loss) < 1e-3
    assert float(swapped_loss) > 5.0
    assert float(accuracy) == 1.0
    assert torch.allclose(min_usage, torch.tensor(0.5))


def test_memory_mode_teacher_is_loaded_from_p1_sidecar_deterministically() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "p1_memory.npz"
        bank = torch.cat(
            [
                F.normalize(torch.eye(3).repeat_interleave(4, dim=0), dim=-1),
                F.normalize((-torch.eye(3)).repeat_interleave(4, dim=0), dim=-1),
            ],
            dim=0,
        ).numpy()
        semantic_ids = torch.arange(3).repeat_interleave(4).repeat(2).numpy()
        import numpy as np

        np.savez(path, bank=bank, semantic_group_ids=semantic_ids)
        first = load_frozen_memory_mode_teacher(path, (2, 2, 2))
        second = load_frozen_memory_mode_teacher(path, (2, 2, 2))

    assert first.centers.shape == (6, 3)
    assert torch.equal(first.centers, second.centers)
    assert not list(first.parameters())


def test_center6_teacher_calibrates_one_radius_per_within_group_mode() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "p1_memory.npz"
        basis = torch.eye(6)
        samples = []
        semantic_ids = []
        for mode_index in range(6):
            group_index = mode_index // 2
            for scale in (0.05, 0.10, 0.15):
                neighbor = basis[(mode_index + 1) % 6]
                samples.append(F.normalize(basis[mode_index] + scale * neighbor, dim=0))
                semantic_ids.append(group_index)
        import numpy as np

        np.savez(
            path,
            bank=torch.stack(samples).numpy(),
            semantic_group_ids=np.asarray(semantic_ids, dtype=np.int64),
        )
        first = load_frozen_center6_teacher(path, (2, 2, 2), radius_quantile=0.95)
        second = load_frozen_center6_teacher(path, (2, 2, 2), radius_quantile=0.95)

    assert first.centers.shape == (6, 6)
    assert first.radii.shape == (6,)
    assert torch.equal(first.centers, second.centers)
    assert torch.equal(first.radii, second.radii)
    assert torch.equal(first.member_counts, second.member_counts)
    assert int(first.member_counts.sum()) == len(samples)
    assert torch.all(first.radii > 0.0)
    assert not list(first.parameters())


def test_adaptive_center_teacher_loads_fewer_modes_with_six_slots() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "adaptive_memory.npz"
        centers = torch.eye(3, 6)
        bank = torch.cat([centers[index].repeat(4, 1) for index in range(3)])
        semantic_ids = torch.arange(3).repeat_interleave(4)
        import numpy as np

        np.savez_compressed(
            path,
            bank=bank.numpy(),
            semantic_group_ids=semantic_ids.numpy(),
            adaptive_mode_centers=centers.numpy(),
            adaptive_mode_groups=np.asarray([1, 1, 1], dtype=np.int64),
            adaptive_slot_groups=np.asarray([2, 2, 2], dtype=np.int64),
            adaptive_slot_to_mode=np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64),
        )
        teacher = load_frozen_center6_teacher(path, (2, 2, 2))

    assert teacher.centers.shape == (3, 6)
    assert teacher.mode_groups == (1, 1, 1)
    assert teacher.slot_to_mode.tolist() == [0, 0, 1, 1, 2, 2]
    assert teacher.mode_group_ids.tolist() == [0, 1, 2]
    assert int(teacher.member_counts.sum()) == 12


def test_adaptive_center_teacher_prefers_saved_full_pool_radii() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "adaptive_full_pool_radii.npz"
        centers = torch.eye(3, 6)
        bank = torch.cat([centers[index].repeat(4, 1) for index in range(3)])
        semantic_ids = torch.arange(3).repeat_interleave(4)
        import numpy as np

        expected_radii = np.asarray([0.11, 0.22, 0.33], dtype=np.float32)
        expected_counts = np.asarray([101, 202, 303], dtype=np.int64)
        np.savez_compressed(
            path,
            bank=bank.numpy(),
            semantic_group_ids=semantic_ids.numpy(),
            adaptive_mode_centers=centers.numpy(),
            adaptive_mode_groups=np.asarray([1, 1, 1], dtype=np.int64),
            adaptive_slot_groups=np.asarray([2, 2, 2], dtype=np.int64),
            adaptive_slot_to_mode=np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64),
            adaptive_mode_radii=expected_radii,
            adaptive_mode_radius_member_counts=expected_counts,
        )
        teacher = load_frozen_center6_teacher(path, (2, 2, 2))

    assert torch.allclose(teacher.radii, torch.from_numpy(expected_radii))
    assert torch.equal(teacher.member_counts, torch.from_numpy(expected_counts))


def test_adaptive_center_teacher_supports_one_slot_per_mode() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "adaptive_slot5_memory.npz"
        centers = torch.eye(5, 6)
        bank = torch.cat([centers[index].repeat(4, 1) for index in range(5)])
        semantic_ids = torch.tensor([0, 1, 1, 2, 2]).repeat_interleave(4)
        import numpy as np

        np.savez_compressed(
            path,
            bank=bank.numpy(),
            semantic_group_ids=semantic_ids.numpy(),
            adaptive_mode_centers=centers.numpy(),
            adaptive_mode_groups=np.asarray([1, 2, 2], dtype=np.int64),
            adaptive_slot_groups=np.asarray([1, 2, 2], dtype=np.int64),
            adaptive_slot_to_mode=np.arange(5, dtype=np.int64),
        )
        teacher = load_frozen_center6_teacher(path, (1, 2, 2))

    assert teacher.centers.shape == (5, 6)
    assert teacher.groups == (1, 2, 2)
    assert teacher.mode_groups == (1, 2, 2)
    assert teacher.slot_to_mode.tolist() == [0, 1, 2, 3, 4]
    assert teacher.mode_group_ids.tolist() == [0, 1, 1, 2, 2]
    assert int(teacher.member_counts.sum()) == 20


def test_hierarchical_center_teacher_loads_group_reliability() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "hierarchical_memory.npz"
        centers = torch.eye(6)
        bank = torch.cat([centers[index].repeat(4, 1) for index in range(6)])
        semantic_ids = torch.arange(3).repeat_interleave(8)
        import numpy as np

        np.savez_compressed(
            path,
            bank=bank.numpy(),
            semantic_group_ids=semantic_ids.numpy(),
            adaptive_mode_centers=centers.numpy(),
            adaptive_mode_groups=np.asarray([2, 2, 2], dtype=np.int64),
            adaptive_slot_groups=np.asarray([2, 2, 2], dtype=np.int64),
            adaptive_slot_to_mode=np.arange(6, dtype=np.int64),
            adaptive_group_reliability=np.asarray([0.25, 0.5, 1.0], dtype=np.float32),
        )
        teacher = load_frozen_center6_teacher(path, (2, 2, 2))

    assert teacher.group_reliability_provided
    assert torch.allclose(teacher.group_reliability, torch.tensor([0.25, 0.5, 1.0]))


def test_hierarchical_center_loss_matches_flat_kl_at_unit_reliability() -> None:
    torch.manual_seed(7)
    centers = F.normalize(torch.randn(6, 8), dim=-1)
    teacher = FrozenCenter6Teacher(
        centers,
        torch.ones(6),
        torch.ones(6, dtype=torch.long),
        (2, 2, 2),
        group_reliability=torch.ones(3),
    )
    query = torch.randn(2, 9, 8)
    keys = torch.randn(2, 6, 8)

    flat = _center6_balanced_loss(
        query,
        keys,
        teacher,
        student_temperature=0.1,
        teacher_temperature=0.1,
        reduction="token_mean",
    )
    hierarchical = _center6_balanced_loss(
        query,
        keys,
        teacher,
        student_temperature=0.1,
        teacher_temperature=0.1,
        reduction="token_mean",
        hierarchical_reliability=True,
    )

    assert torch.allclose(flat[0], hierarchical[0], atol=1e-5)
    assert torch.allclose(flat[2], hierarchical[2], atol=1e-7)
    assert torch.allclose(flat[3], hierarchical[3], atol=1e-7)


def test_zero_child_reliability_ignores_slot_order_inside_group() -> None:
    centers = torch.eye(6)
    teacher = FrozenCenter6Teacher(
        centers,
        torch.ones(6),
        torch.ones(6, dtype=torch.long),
        (2, 2, 2),
        group_reliability=torch.zeros(3),
    )
    query = centers.unsqueeze(0)
    keys = centers.unsqueeze(0)
    swapped = centers[torch.tensor([1, 0, 3, 2, 5, 4])].unsqueeze(0)

    base = _center6_balanced_loss(
        query,
        keys,
        teacher,
        student_temperature=0.1,
        teacher_temperature=0.1,
        reduction="token_mean",
        hierarchical_reliability=True,
    )
    reordered = _center6_balanced_loss(
        query,
        swapped,
        teacher,
        student_temperature=0.1,
        teacher_temperature=0.1,
        reduction="token_mean",
        hierarchical_reliability=True,
    )

    assert torch.allclose(base[0], reordered[0], atol=1e-6)
    assert float(base[1]["conditional_loss"]) == 0.0
    assert float(reordered[1]["conditional_loss"]) == 0.0


def test_adaptive_center_loss_sums_child_slot_probabilities_by_parent_mode() -> None:
    centers = torch.eye(3, 6)
    teacher = FrozenCenter6Teacher(
        centers,
        torch.ones(3),
        torch.ones(3, dtype=torch.long),
        (2, 2, 2),
        mode_groups=(1, 1, 1),
        slot_to_mode=torch.tensor([0, 0, 1, 1, 2, 2]),
    )
    keys = centers.repeat_interleave(2, dim=0).unsqueeze(0)
    query = centers.unsqueeze(0)

    loss, diagnostics, teacher_probability, student_probability, *_ = (
        _center6_balanced_loss(
            query,
            keys,
            teacher,
            student_temperature=0.1,
            teacher_temperature=0.1,
            reduction="token_mean",
        )
    )

    assert float(loss) < 1e-6
    assert teacher_probability.shape == student_probability.shape == (1, 3, 3)
    assert torch.allclose(teacher_probability, student_probability, atol=1e-6)
    assert float(diagnostics["effective_mode_count"]) == 3.0
    assert float(diagnostics["prototype_slot_count"]) == 6.0
    assert float(diagnostics["shared_parent_pair_count"]) == 3.0
    assert torch.allclose(diagnostics["shared_parent_cosine_mean"], torch.tensor(1.0))
    grouped = _center6_group_priors(query, teacher, (2, 2, 2), temperature=0.1)
    assert grouped.shape == (1, 3, 3)
    assert torch.allclose(grouped.sum(dim=-1), torch.ones(1, 3), atol=1e-6)


def test_hierarchical_teacher_removes_effective_mode_count_bias() -> None:
    center = F.normalize(torch.ones(6), dim=0)
    centers = center.repeat(5, 1)
    teacher = FrozenCenter6Teacher(
        centers,
        torch.ones(5),
        torch.ones(5, dtype=torch.long),
        (2, 2, 2),
        mode_groups=(1, 2, 2),
        slot_to_mode=torch.tensor([0, 0, 1, 2, 3, 4]),
    )
    query = center.view(1, 1, -1)

    flat = _center6_group_priors(
        query, teacher, (2, 2, 2), temperature=0.1, hierarchical=False
    )
    hierarchical = _center6_group_priors(
        query, teacher, (2, 2, 2), temperature=0.1, hierarchical=True
    )

    assert torch.allclose(flat[0, 0], torch.tensor([0.2, 0.4, 0.4]))
    assert torch.allclose(
        hierarchical[0, 0], torch.full((3,), 1.0 / 3.0), atol=1e-6
    )


def test_center6_novelty_veto_softly_downweights_outside_tokens() -> None:
    centers = torch.eye(3)
    teacher = FrozenCenter6Teacher(
        centers,
        torch.full((3,), 0.10),
        torch.ones(3, dtype=torch.long),
        (1, 1, 1),
    )
    query = torch.stack(
        [centers[0], F.normalize(torch.tensor([-1.0, -1.0, -1.0]), dim=0)]
    ).unsqueeze(0)
    keys = centers.unsqueeze(0)

    _, diagnostics, *_ = _center6_balanced_loss(
        query,
        keys,
        teacher,
        student_temperature=0.1,
        teacher_temperature=0.1,
        reduction="token_mean",
        novelty_veto=True,
        novelty_threshold=1.0,
        novelty_temperature=0.1,
        novelty_min_weight=0.05,
    )

    assert 0.0 < float(diagnostics["normal_support_weight"]) < 1.0
    assert torch.allclose(
        diagnostics["normal_support_fraction"], torch.tensor(0.5)
    )


def test_hard_center6_loss_is_nearest_mode_cross_entropy() -> None:
    centers = torch.eye(3, 6)
    teacher = FrozenCenter6Teacher(
        centers,
        torch.ones(3),
        torch.ones(3, dtype=torch.long),
        (2, 2, 2),
        mode_groups=(1, 1, 1),
        slot_to_mode=torch.tensor([0, 0, 1, 1, 2, 2]),
    )
    query = F.normalize(
        torch.tensor([[[1.0, 0.1, 0, 0, 0, 0], [0.1, 1.0, 0, 0, 0, 0]]]),
        dim=-1,
    )
    keys = F.normalize(
        torch.tensor(
            [[
                [1.0, 0.0, 0, 0, 0, 0],
                [0.9, 0.1, 0, 0, 0, 0],
                [0.0, 1.0, 0, 0, 0, 0],
                [0.1, 0.9, 0, 0, 0, 0],
                [0.0, 0.0, 1, 0, 0, 0],
                [0.0, 0.0, 0.9, 0.1, 0, 0],
            ]]
        ),
        dim=-1,
    )

    loss, _, teacher_probability, student_probability, teacher_hard, _ = (
        _center6_balanced_loss(
            query,
            keys,
            teacher,
            student_temperature=0.1,
            teacher_temperature=0.1,
            teacher_mapping="hard",
            reduction="token_mean",
        )
    )
    expected = -student_probability.clamp_min(
        torch.finfo(student_probability.dtype).tiny
    ).log().gather(-1, teacher_hard.unsqueeze(-1)).mean()

    assert torch.equal(
        teacher_probability,
        F.one_hot(teacher_hard, num_classes=3).to(teacher_probability),
    )
    assert torch.allclose(loss, expected, atol=1e-7)


def test_collapsed_slot_diversity_penalizes_conditioned_shared_parent_pair() -> None:
    centers = torch.eye(3, 6)
    teacher = FrozenCenter6Teacher(
        centers,
        torch.ones(3),
        torch.ones(3, dtype=torch.long),
        (2, 2, 2),
        mode_groups=(1, 1, 1),
        slot_to_mode=torch.tensor([0, 0, 1, 1, 2, 2]),
    )
    keys = centers.repeat_interleave(2, dim=0).unsqueeze(0)
    query = centers.unsqueeze(0)
    base = _center6_balanced_loss(
        query,
        keys,
        teacher,
        student_temperature=0.1,
        teacher_temperature=0.1,
        reduction="token_mean",
    )
    diverse = _center6_balanced_loss(
        query,
        keys,
        teacher,
        student_temperature=0.1,
        teacher_temperature=0.1,
        reduction="token_mean",
        collapsed_diversity_weight=0.5,
        collapsed_diversity_margin=0.90,
    )

    assert torch.allclose(diverse[0] - base[0], torch.tensor(0.05), atol=1e-6)
    assert torch.allclose(diverse[1]["collapsed_diversity_loss"], torch.tensor(0.10))
    assert float(diverse[1]["shared_parent_pair_count"]) == 3.0


def test_center6_balanced_loss_is_invariant_to_duplicate_tokens_in_one_mode() -> None:
    centers = torch.eye(6)
    teacher = FrozenCenter6Teacher(
        centers,
        torch.ones(6),
        torch.ones(6, dtype=torch.long),
        (2, 2, 2),
    )
    swapped_keys = centers[torch.tensor([1, 0, 2, 3, 4, 5])].unsqueeze(0)
    base_query = centers.unsqueeze(0)
    repeated_query = torch.cat([base_query, centers[0].reshape(1, 1, 6).repeat(1, 9, 1)], dim=1)

    base_loss, base_diag, *_ = _center6_balanced_loss(
        base_query,
        swapped_keys,
        teacher,
        student_temperature=0.1,
        teacher_temperature=0.1,
    )
    repeated_loss, repeated_diag, *_ = _center6_balanced_loss(
        repeated_query,
        swapped_keys,
        teacher,
        student_temperature=0.1,
        teacher_temperature=0.1,
    )

    assert float(base_loss) > 1.0
    assert torch.allclose(base_loss, repeated_loss, atol=1e-6)
    assert float(base_diag["teacher_dead_modes"]) == 0.0
    assert float(repeated_diag["teacher_dead_modes"]) == 0.0


def test_center6_reductions_match_their_declared_mode_weights() -> None:
    centers = torch.eye(6)
    teacher = FrozenCenter6Teacher(
        centers,
        torch.ones(6),
        torch.ones(6, dtype=torch.long),
        (2, 2, 2),
    )
    query = torch.cat(
        [
            centers[0].reshape(1, 1, 6).repeat(1, 8, 1),
            centers[1].reshape(1, 1, 6).repeat(1, 2, 1),
            centers[2:].unsqueeze(0),
        ],
        dim=1,
    )
    keys = centers[torch.tensor([1, 0, 3, 2, 4, 5])].unsqueeze(0)

    outputs = {}
    for reduction in ("equal_mode", "token_mean", "sqrt_balanced"):
        outputs[reduction] = _center6_balanced_loss(
            query,
            keys,
            teacher,
            student_temperature=0.1,
            teacher_temperature=0.1,
            reduction=reduction,
        )

    _, _, teacher_probability, student_probability, teacher_hard, _ = outputs[
        "token_mean"
    ]
    token_kl = (
        teacher_probability
        * (
            teacher_probability.clamp_min(1e-12).log()
            - student_probability.clamp_min(1e-12).log()
        )
    ).sum(dim=-1)
    mode_losses = []
    counts = []
    for mode_index in range(6):
        selected = teacher_hard == mode_index
        mode_losses.append(token_kl[selected].mean())
        counts.append(selected.sum().to(token_kl.dtype))
    mode_losses = torch.stack(mode_losses)
    counts = torch.stack(counts)

    expected_weights = {
        "equal_mode": torch.ones_like(counts) / 6.0,
        "token_mean": counts / counts.sum(),
        "sqrt_balanced": counts.sqrt() / counts.sqrt().sum(),
    }
    for reduction, (loss, diagnostics, *_rest) in outputs.items():
        weights = expected_weights[reduction]
        assert torch.allclose(loss, (mode_losses * weights).sum(), atol=1e-6)
        logged_weights = torch.stack(
            [diagnostics[f"mode_weight_{mode_index}"] for mode_index in range(6)]
        )
        assert torch.allclose(logged_weights, weights, atol=1e-7)


def test_center6_group_prior_exposes_objectness_for_familiarity_gate() -> None:
    centers = torch.eye(6)
    teacher = FrozenCenter6Teacher(
        centers,
        torch.ones(6),
        torch.ones(6, dtype=torch.long),
        (2, 2, 2),
    )
    query = centers.unsqueeze(0)

    grouped = _center6_group_priors(
        query,
        teacher,
        (2, 2, 2),
        temperature=0.01,
    )

    assert grouped.shape == (1, 6, 3)
    assert torch.allclose(grouped.sum(dim=-1), torch.ones(1, 6), atol=1e-6)
    assert torch.all(grouped[0, :2, 0] > 0.999)
    assert torch.all(grouped[0, 2:4, 1] > 0.999)
    assert torch.all(grouped[0, 4:, 2] > 0.999)


def test_hard_center6_group_prior_is_one_hot_nearest_mode_group() -> None:
    centers = torch.eye(6)
    teacher = FrozenCenter6Teacher(
        centers,
        torch.ones(6),
        torch.ones(6, dtype=torch.long),
        (2, 2, 2),
    )
    query = centers.unsqueeze(0)

    grouped = _center6_group_priors(
        query,
        teacher,
        (2, 2, 2),
        temperature=0.10,
        mapping="hard",
    )

    expected = F.one_hot(
        torch.tensor([[0, 0, 1, 1, 2, 2]]),
        num_classes=3,
    ).to(grouped)
    assert torch.equal(grouped, expected)


def test_center6_teacher_prior_uses_radius_normalized_distance() -> None:
    centers = torch.eye(6)
    teacher = FrozenCenter6Teacher(
        centers,
        torch.tensor([0.5, 1.0, 1.0, 1.0, 1.0, 1.0]),
        torch.ones(6, dtype=torch.long),
        (2, 2, 2),
    )
    query = F.normalize(torch.tensor([[[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]]), dim=-1)

    _, _, teacher_probability, _, teacher_hard, _ = _center6_balanced_loss(
        query,
        centers.unsqueeze(0),
        teacher,
        student_temperature=0.1,
        teacher_temperature=0.1,
    )

    assert float(teacher_probability[0, 0, 1]) > float(teacher_probability[0, 0, 0])
    assert int(teacher_hard[0, 0]) == 1


def test_stable_memory_mode_teacher_loads_frozen_centers_and_calibration() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "stable_mode_teacher.npz"
        centers = F.normalize(torch.eye(6), dim=-1)
        import numpy as np

        np.savez_compressed(
            path,
            mode_teacher_centers=centers.numpy(),
            mode_teacher_groups=np.asarray([2, 2, 2], dtype=np.int64),
            mode_teacher_group_reliability=np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
            mode_teacher_margin_floor=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            mode_teacher_margin_scale=np.asarray([0.4, 0.5, 0.6], dtype=np.float32),
            mode_teacher_construction_version=np.asarray("stable_view_bootstrap_v1"),
        )
        teacher = load_frozen_memory_mode_teacher(path, (2, 2, 2))

    assert torch.equal(teacher.centers, centers)
    assert teacher.has_confidence_calibration
    assert torch.allclose(teacher.group_reliability, torch.tensor([0.9, 0.8, 0.7]))
    assert set(teacher.state_dict()) == {"centers"}


def test_memory_mode_margin_weighting_suppresses_ambiguous_teacher_label() -> None:
    groups = (2, 2, 2)
    centers = torch.eye(6)
    teacher = FrozenMemoryModeTeacher(
        centers,
        groups,
        group_reliability=torch.ones(3),
        margin_floor=torch.full((3,), 0.20),
        margin_scale=torch.full((3,), 0.50),
    )
    query = F.normalize(torch.tensor([[[1.0, 0.99, 0.0, 0.0, 0.0, 0.0]]]), dim=-1)
    swapped_keys = centers[torch.tensor([1, 0, 2, 3, 4, 5])].unsqueeze(0)
    distribution = 1.0 - F.cosine_similarity(
        query.unsqueeze(2), swapped_keys.unsqueeze(1), dim=-1
    )
    target = torch.zeros((1, 1), dtype=torch.long)
    confidence = torch.ones((1, 1))
    valid = torch.ones((1, 1), dtype=torch.bool)

    hard_loss, _, _ = _memory_mode_supervision_loss(
        query,
        distribution,
        target,
        confidence,
        valid,
        teacher,
        groups,
        temperature=0.1,
    )
    weighted_loss, _, _ = _memory_mode_supervision_loss(
        query,
        distribution,
        target,
        confidence,
        valid,
        teacher,
        groups,
        temperature=0.1,
        margin_weighting=True,
    )

    assert float(hard_loss) > 0.1
    assert float(weighted_loss) == 0.0


def test_memory_mode_soft_semantic_weights_supervise_boundary_tokens() -> None:
    groups = (2, 2, 2)
    centers = torch.eye(6)
    teacher = FrozenMemoryModeTeacher(centers, groups)
    query = centers[0].reshape(1, 1, 6)
    aligned_keys = centers.unsqueeze(0)
    swapped_keys = centers[torch.tensor([1, 0, 2, 3, 4, 5])].unsqueeze(0)
    target = torch.zeros((1, 1), dtype=torch.long)
    confidence = torch.zeros((1, 1))
    valid = torch.zeros((1, 1), dtype=torch.bool)
    semantic_weights = torch.tensor([[[1.0, 0.0, 0.0]]])

    def loss_for(keys: torch.Tensor) -> torch.Tensor:
        distribution = 1.0 - F.cosine_similarity(
            query.unsqueeze(2), keys.unsqueeze(1), dim=-1
        )
        return _memory_mode_supervision_loss(
            query,
            distribution,
            target,
            confidence,
            valid,
            teacher,
            groups,
            temperature=0.1,
            semantic_weights=semantic_weights,
        )[0]

    assert float(loss_for(aligned_keys)) < 1e-3
    assert float(loss_for(swapped_keys)) > 5.0
