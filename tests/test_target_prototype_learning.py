import torch
from torch import nn

from fod_recon_ad.target_prototype_learning import (
    aggregation_attention_exclusion_loss,
    DecoderAttentionCapture,
    decoder_read_attention_exclusion_loss,
    nearest_prototype_distance,
    prototype_invariance_loss,
    prototype_repulsion_loss,
    token_occupancy,
)


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_heads = 1
        self.q = nn.Linear(2, 2, bias=False)
        self.kv = nn.Linear(2, 4, bias=False)
        self.learn_scale = nn.Parameter(torch.ones(()))

    def forward(self, tokens: torch.Tensor, prototypes: torch.Tensor):
        # The capture only requires the standard (update, attention) contract.
        attention = torch.ones(tokens.shape[0], 1, tokens.shape[1], prototypes.shape[1])
        return tokens, attention


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _Attention()


def test_token_occupancy_preserves_soft_area_fraction() -> None:
    mask = torch.zeros(1, 1, 4, 4)
    mask[:, :, :2, :2] = 1.0
    occupancy = token_occupancy(mask, 4)
    assert torch.equal(occupancy, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))


def test_prototype_invariance_is_slotwise_and_backpropagates() -> None:
    clean = torch.eye(3).unsqueeze(0)
    composite = clean.clone().requires_grad_(True)
    loss, diag = prototype_invariance_loss(composite, clean)
    loss.backward()
    assert loss.item() == 0.0
    assert diag["target_proto_cosine"] == 1.0
    assert composite.grad is not None


def test_repulsion_hinge_pushes_target_beyond_normal_tail() -> None:
    clean_prototypes = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    clean_tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]])
    target_tokens = clean_tokens.clone()
    composite = clean_prototypes.clone().requires_grad_(True)
    mask = torch.ones(1, 1, 2, 2)
    loss, diag = prototype_repulsion_loss(
        target_tokens,
        composite,
        mask,
        clean_tokens,
        clean_prototypes,
        margin_delta=0.2,
    )
    loss.backward()
    assert loss.item() > 0.0
    assert diag["target_proto_repulsion_active"] == 1.0
    assert composite.grad is not None


def test_effective_mode_repulsion_uses_matched_normal_tail() -> None:
    clean_prototypes = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
    )
    clean_tokens = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]]]
    )
    target_tokens = torch.tensor(
        [[[0.9, 0.0, 0.4358899]]]
    )
    mask = torch.ones(1, 1, 1, 1)

    global_prototypes = clean_prototypes.clone().requires_grad_(True)
    global_loss, global_diag = prototype_repulsion_loss(
        target_tokens,
        global_prototypes,
        mask,
        clean_tokens,
        clean_prototypes,
        normal_quantile=1.0,
        margin_delta=0.0,
    )
    mode_prototypes = clean_prototypes.clone().requires_grad_(True)
    mode_loss, mode_diag = prototype_repulsion_loss(
        target_tokens,
        mode_prototypes,
        mask,
        clean_tokens,
        clean_prototypes,
        normal_quantile=1.0,
        margin_delta=0.0,
        prototype_to_mode=torch.tensor([0, 1]),
        minimum_normal_tokens_per_mode=1,
    )

    assert global_loss.item() > 0.0
    assert mode_loss.item() == 0.0
    assert mode_diag["target_proto_effective_mode_count"] == 2.0
    assert mode_diag["target_proto_margin_fallback_modes"] == 0.0
    assert mode_diag["target_proto_repulsion_margin"] < global_diag["target_proto_repulsion_margin"]


def test_effective_mode_repulsion_falls_back_for_unsupported_mode() -> None:
    prototypes = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], requires_grad=True
    )
    clean_tokens = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]]]
    )
    target_tokens = torch.tensor([[[0.0, 1.0, 0.0]]])
    loss, diag = prototype_repulsion_loss(
        target_tokens,
        prototypes,
        torch.ones(1, 1, 1, 1),
        clean_tokens,
        prototypes.detach(),
        normal_quantile=1.0,
        margin_delta=0.1,
        prototype_to_mode=torch.tensor([0, 1]),
        minimum_normal_tokens_per_mode=2,
    )
    loss.backward()
    assert diag["target_proto_margin_fallback_modes"] == 2.0
    assert prototypes.grad is not None


def test_effective_mode_diagnostics_report_mode_shares_and_drift() -> None:
    prototypes = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]], requires_grad=True
    )
    clean_tokens = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]]
    )
    target_tokens = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]
    )
    mapping = torch.tensor([0, 0, 1])
    loss, diag = prototype_repulsion_loss(
        target_tokens,
        prototypes,
        torch.ones(1, 1, 2, 2),
        clean_tokens,
        prototypes.detach(),
        normal_quantile=1.0,
        margin_delta=0.1,
        prototype_to_mode=mapping,
        minimum_normal_tokens_per_mode=1,
    )
    assert loss.item() > 0.0
    assert diag["target_proto_mode_0_target_share"] == 1.0
    assert diag["target_proto_mode_1_target_share"] == 0.0
    assert diag["target_proto_mode_0_suggested_budget_weight"] < 1.0
    _, inv_diag = prototype_invariance_loss(
        prototypes,
        prototypes.detach(),
        prototype_to_mode=mapping,
    )
    assert inv_diag["target_proto_mode_0_drift"] == 0.0
    assert inv_diag["target_proto_mode_1_drift"] == 0.0


def test_clean_target_ratio_budget_reduces_overrepresented_mode_loss() -> None:
    prototypes = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True
    )
    clean_tokens = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]]
    )
    target_tokens = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]
    )
    kwargs = dict(
        target_mask=torch.ones(1, 1, 2, 2),
        clean_tokens=clean_tokens,
        clean_prototypes=prototypes.detach(),
        normal_quantile=1.0,
        margin_delta=0.2,
        prototype_to_mode=torch.tensor([0, 1]),
        minimum_normal_tokens_per_mode=1,
    )
    baseline_loss, baseline_diag = prototype_repulsion_loss(
        target_tokens,
        prototypes,
        mode_budget="none",
        **kwargs,
    )
    budget_loss, budget_diag = prototype_repulsion_loss(
        target_tokens,
        prototypes,
        mode_budget="clean_target_ratio",
        **kwargs,
    )
    assert budget_loss.item() < baseline_loss.item()
    assert baseline_diag["target_proto_mode_0_applied_budget_weight"] == 1.0
    assert budget_diag["target_proto_mode_0_applied_budget_weight"] == 0.25
    budget_loss.backward()
    assert prototypes.grad is not None


def test_target_side_repulsion_stops_prototype_gradient() -> None:
    target_tokens = torch.tensor(
        [[[0.9, 0.1, 0.0]]], requires_grad=True
    )
    prototypes = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], requires_grad=True
    )
    clean_tokens = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    loss, diag = prototype_repulsion_loss(
        target_tokens,
        prototypes,
        torch.ones(1, 1, 1, 1),
        clean_tokens,
        prototypes.detach(),
        normal_quantile=1.0,
        margin_delta=0.2,
        gradient_side="target",
    )
    loss.backward()
    assert loss.item() > 0.0
    assert diag["target_proto_gradient_side_target"] == 1.0
    assert target_tokens.grad is not None
    assert torch.isfinite(target_tokens.grad).all()
    assert prototypes.grad is None


def test_nearest_distance_is_zero_for_matching_prototype() -> None:
    prototypes = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    assert torch.equal(nearest_prototype_distance(tokens, prototypes), torch.zeros(1, 2))


def test_aggregation_exclusion_penalizes_target_attention_mass() -> None:
    clean = [torch.full((1, 1, 1, 4), 0.25)]
    composite = [torch.tensor([[[[0.7, 0.1, 0.1, 0.1]]]], requires_grad=True)]
    mask = torch.zeros(1, 1, 2, 2)
    mask[:, :, 0, 0] = 1.0
    loss, diag = aggregation_attention_exclusion_loss(composite, clean, mask)
    loss.backward()
    assert loss.item() > 0.0
    assert diag["target_aggregation_attention_density_ratio"] > 1.0
    assert composite[0].grad is not None


def test_decoder_read_loss_suppresses_target_and_anchors_background() -> None:
    clean = [torch.ones(1, 1, 4, 2)]
    values = torch.ones(1, 1, 4, 2)
    values[:, :, 0, :] = 2.0
    composite = [values.requires_grad_(True)]
    mask = torch.zeros(1, 1, 2, 2)
    mask[:, :, 0, 0] = 1.0
    loss, diag = decoder_read_attention_exclusion_loss(
        composite,
        clean,
        mask,
        target_to_background_ratio=0.25,
    )
    loss.backward()
    assert loss.item() > 0.0
    assert diag["target_read_attention_ratio"] > 1.0
    assert composite[0].grad is not None


def test_decoder_attention_recompute_detaches_inputs_but_updates_attention() -> None:
    block = _Block()
    tokens = torch.randn(1, 4, 2, requires_grad=True)
    prototypes = torch.randn(1, 2, 2, requires_grad=True)
    capture = DecoderAttentionCapture([block])
    capture.start()
    block.attn(tokens, prototypes)
    recomputed = capture.recompute_last(1)[0]
    capture.stop()
    recomputed.sum().backward()
    assert tokens.grad is None
    assert prototypes.grad is None
    assert block.attn.q.weight.grad is not None
    assert block.attn.kv.weight.grad is not None
