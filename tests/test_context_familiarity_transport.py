from __future__ import annotations

import argparse

import pytest
import torch

from fod_recon_ad.context_familiarity_transport import (
    ContextFamiliarityTransportMemory,
    _retrieval_source_diversity,
    _temporary_eval,
)
from fod_recon_ad.prototype_guidance import add_guided_prototype_args, guided_config_from_args


def _memory() -> ContextFamiliarityTransportMemory:
    memory = ContextFamiliarityTransportMemory(
        dim=2,
        radii=(1, 2),
        size=8,
        topk=1,
        temperature=0.1,
        query_chunk_size=32,
    )
    keys = torch.tensor(
        [
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
            [[0.8, 0.2], [0.7, 0.3], [0.2, 0.8], [0.3, 0.7]],
        ]
    )
    values = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    memory.set_entries(keys, values, torch.tensor([1, 1, 2, 2]))
    return memory


def test_context_memory_is_leave_source_out_calibrated() -> None:
    memory = _memory()

    assert memory.ready
    assert memory.calibration_counts.tolist() == [4, 4]
    assert torch.all(memory.calibration_residuals[:, :4] > 0.5)


def test_transport_changes_only_context_surprising_token() -> None:
    memory = _memory()
    target = torch.zeros(1, 9, 2)
    target[..., 0] = 1.0
    target[:, 4] = torch.tensor([0.0, 1.0])

    transported, objectness, diagnostics = memory.transport(
        target,
        objectness=torch.ones(1, 9),
        fixed_scale_weights=(0.35, 0.65),
        adaptive_scale=False,
    )

    assert transported.shape == target.shape
    assert objectness.shape == target.shape[:2]
    assert torch.allclose(transported[:, 0], target[:, 0])
    assert not torch.allclose(transported[:, 4], target[:, 4])
    assert 0.0 < diagnostics["guided_transport_gate"] < 1.0
    assert diagnostics["guided_transport_scale_r1"] == pytest.approx(0.35, abs=1e-6)
    assert diagnostics["guided_transport_scale_r2"] == pytest.approx(0.65, abs=1e-6)


def test_transport_preserves_value_dim_when_retrieval_keys_are_compressed() -> None:
    torch.manual_seed(20260715)
    memory = ContextFamiliarityTransportMemory(
        dim=768,
        radii=(1,),
        size=8,
        topk=1,
        temperature=0.1,
        query_chunk_size=32,
        key_dim=64,
    )
    keys = torch.randn(1, 4, 768)
    values = torch.randn(4, 768)
    memory.set_entries(keys, values, torch.tensor([1, 1, 2, 2]))
    target = torch.randn(1, 9, 768)

    transported, objectness, _ = memory.transport(
        target,
        objectness=torch.ones(1, 9),
        fixed_scale_weights=(1.0,),
        adaptive_scale=False,
    )

    assert memory.keys.shape[-1] == 64
    assert memory.values.shape[-1] == 768
    assert transported.shape == target.shape
    assert objectness.shape == target.shape[:2]


def test_adaptive_scale_uses_normal_reliability_not_fixed_weights() -> None:
    memory = _memory()
    memory.scale_reliability.copy_(torch.tensor([0.8, 0.2]))
    target = torch.zeros(1, 9, 2)
    target[..., 0] = 1.0

    _, _, diagnostics = memory.transport(
        target,
        objectness=torch.ones(1, 9),
        fixed_scale_weights=(0.05, 0.95),
        adaptive_scale=True,
    )

    assert diagnostics["guided_transport_scale_r1"] > diagnostics["guided_transport_scale_r2"]


def test_context_familiarity_atomic_presets() -> None:
    parser = argparse.ArgumentParser()
    add_guided_prototype_args(parser)
    a1 = guided_config_from_args(parser.parse_args(["--guided-prototype-context-variant", "a1"]))
    a2 = guided_config_from_args(parser.parse_args(["--guided-prototype-context-variant", "a2"]))
    a2_safe = guided_config_from_args(
        parser.parse_args(["--guided-prototype-context-variant", "a2_safe"])
    )
    a3 = guided_config_from_args(parser.parse_args(["--guided-prototype-context-variant", "a3"]))

    assert (a1.context_transport, a1.context_adaptive_scale, a1.free_prototypes) == (True, False, False)
    assert (a2.context_transport, a2.context_adaptive_scale, a2.free_prototypes) == (True, True, False)
    assert (a2_safe.context_transport, a2_safe.context_adaptive_scale, a2_safe.free_prototypes) == (
        True,
        True,
        True,
    )
    assert (a3.context_transport, a3.context_adaptive_scale, a3.free_prototypes) == (True, True, True)
    for config in (a1, a2, a2_safe, a3):
        assert config.multiscale_direct
        assert config.groups == (2, 2, 2)
        assert config.group_weights == (0.92, 1.10, 1.10)
        assert config.object_kernels == (3, 5, 7)
        assert config.object_kernel_weights == (0.35, 0.25, 0.40)


def test_low_level_context_switch_dependencies() -> None:
    parser = argparse.ArgumentParser()
    add_guided_prototype_args(parser)
    with pytest.raises(ValueError, match="requires"):
        guided_config_from_args(parser.parse_args(["--guided-prototype-context-adaptive-scale"]))


def test_context_checkpoint_rejects_incompatible_configuration() -> None:
    source = ContextFamiliarityTransportMemory(
        dim=4,
        radii=(1, 2),
        size=8,
        topk=2,
        temperature=0.1,
        query_chunk_size=16,
        key_dim=2,
        mode_signature=(1.0, 2.0),
    )
    state = source.state_dict()

    wrong_radius = ContextFamiliarityTransportMemory(
        dim=4,
        radii=(1, 3),
        size=8,
        topk=2,
        temperature=0.1,
        query_chunk_size=16,
        key_dim=2,
        mode_signature=(1.0, 2.0),
    )
    with pytest.raises(RuntimeError, match="config mismatch"):
        wrong_radius.load_state_dict(state, strict=True)

    wrong_mode = ContextFamiliarityTransportMemory(
        dim=4,
        radii=(1, 2),
        size=8,
        topk=2,
        temperature=0.1,
        query_chunk_size=16,
        key_dim=2,
        mode_signature=(1.0, 3.0),
    )
    with pytest.raises(RuntimeError, match="config mismatch"):
        wrong_mode.load_state_dict(state, strict=True)

    wrong_temperature = ContextFamiliarityTransportMemory(
        dim=4,
        radii=(1, 2),
        size=8,
        topk=2,
        temperature=0.2,
        query_chunk_size=16,
        key_dim=2,
        mode_signature=(1.0, 2.0),
    )
    with pytest.raises(RuntimeError, match="config mismatch"):
        wrong_temperature.load_state_dict(state, strict=True)


def test_source_diversity_counts_distinct_valid_images() -> None:
    groups = torch.tensor([[1, 1, 1], [1, 2, 3], [-1, 2, 3]])
    diversity = _retrieval_source_diversity(groups)

    assert torch.allclose(diversity, torch.tensor([1.0 / 3.0, 1.0, 2.0 / 3.0]))


def test_calibrated_scale_reliability_is_bounded() -> None:
    reliability = _memory().scale_reliability

    assert torch.all(reliability > 0.0)
    assert torch.all(reliability < 1.0)


def test_temporary_eval_restores_mixed_module_states() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
    model.train()
    model[0].eval()

    with _temporary_eval(model):
        assert not any(module.training for module in model.modules())

    assert model.training
    assert not model[0].training
    assert model[1].training
