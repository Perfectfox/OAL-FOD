from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from fod_recon_ad.normal_descriptor_gate import NormalDescriptorMemory
from fod_recon_ad.normal_calibration import SourceGroupResolver
from fod_recon_ad.prototype_guidance import (
    _apply_descriptor_read_route,
    _descriptor_aux_objectness,
    _descriptor_write_weights,
    add_guided_prototype_args,
    configure_guided_prototypes,
    guided_config_from_args,
)


def _memory() -> NormalDescriptorMemory:
    memory = NormalDescriptorMemory(
        dim=4,
        radii=(1, 2),
        size=8,
        topk=2,
        temperature=0.1,
        query_chunk_size=16,
        key_dim=3,
        mode_signature=(1.0, 2.0),
    )
    values = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
            [0.7, 0.3, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.1, 0.9, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0],
            [0.3, 0.7, 0.0, 0.0],
        ]
    )
    keys = torch.stack(
        [values.roll(shifts=1, dims=0), values.roll(shifts=2, dims=0)]
    )
    groups = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    memory.set_entries(keys, values, groups)
    return memory


def test_descriptor_memory_returns_calibrated_continuous_evidence() -> None:
    memory = _memory()
    tokens = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]] * 8 + [[0.0, 0.0, 1.0, 0.0]]]
    )

    evidence, diagnostics = memory.describe(tokens)

    assert memory.ready
    assert evidence.expected_normal.shape == tokens.shape
    for value in (
        evidence.appearance_novelty,
        evidence.context_surprise,
        evidence.confidence,
        evidence.local_objectness,
        evidence.write_risk,
        evidence.read_risk,
    ):
        assert value.shape == tokens.shape[:2]
        assert bool(((0.0 <= value) & (value <= 1.0)).all())
    assert diagnostics["guided_descriptor_memory_count"] == 8.0
    assert evidence.appearance_novelty[0, -1] >= evidence.appearance_novelty[0, 0]


def test_descriptor_checkpoint_round_trip_is_strict() -> None:
    source = _memory()
    restored = NormalDescriptorMemory(
        dim=4,
        radii=(1, 2),
        size=8,
        topk=2,
        temperature=0.1,
        query_chunk_size=16,
        key_dim=3,
        mode_signature=(1.0, 2.0),
    )

    restored.load_state_dict(source.state_dict(), strict=True)

    assert restored.ready
    assert torch.equal(
        restored.calibration_appearance_distances,
        source.calibration_appearance_distances,
    )


def test_descriptor_read_route_preserves_mass_and_prefers_expected_normal() -> None:
    attention = torch.tensor([[[[1.0, 1.0]]]])
    expected = torch.tensor([[[0.0, 1.0]]])
    prototypes = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    risk = torch.ones(1, 1)

    routed, activation = _apply_descriptor_read_route(
        attention,
        expected,
        prototypes,
        risk,
        strength=1.0,
        tail_threshold=0.0,
        tail_upper=1.0,
        tail_power=1.0,
    )

    assert activation.item() == pytest.approx(1.0)
    assert routed.sum().item() == pytest.approx(attention.sum().item(), abs=1e-6)
    assert routed[0, 0, 0, 1] > routed[0, 0, 0, 0]


def test_descriptor_read_route_zero_risk_is_exact_identity() -> None:
    attention = torch.rand(2, 3, 4, 6)
    expected = torch.randn(2, 4, 8)
    prototypes = torch.randn(2, 6, 8)

    routed, activation = _apply_descriptor_read_route(
        attention,
        expected,
        prototypes,
        torch.zeros(2, 4),
        strength=0.7,
        tail_threshold=0.2,
        tail_upper=0.9,
        tail_power=2.0,
    )

    assert torch.equal(activation, torch.zeros_like(activation))
    assert torch.equal(routed, attention)


def test_descriptor_write_has_exact_native_anchor_and_d_is_monotone() -> None:
    risk = torch.tensor([[0.0, 0.5, 1.0]])
    native = _descriptor_write_weights(risk, minimum=0.05, power=1.0, alpha=0.0)
    guided = _descriptor_write_weights(risk, minimum=0.05, power=1.0, alpha=1.0)
    modulated, factor = _descriptor_aux_objectness(
        risk,
        torch.tensor([[0.0, 0.5, 1.0]]),
        floor=0.5,
    )

    assert torch.equal(native, torch.ones_like(native))
    assert guided.tolist()[0] == pytest.approx([1.0, 0.525, 0.05])
    assert factor.tolist()[0] == pytest.approx([0.5, 0.75, 1.0])
    assert bool((modulated <= risk).all())


def test_descriptor_b_c_d_are_atomic_free_prototype_presets() -> None:
    parser = argparse.ArgumentParser()
    add_guided_prototype_args(parser)

    configs = {}
    for variant in ("b", "c", "d"):
        args = parser.parse_args(
            ["--guided-prototype", "--guided-prototype-descriptor-variant", variant]
        )
        configs[variant] = guided_config_from_args(args)

    assert all(config.free_prototypes for config in configs.values())
    assert all(not config.context_transport for config in configs.values())
    assert configs["b"].descriptor_variant == "b"
    assert configs["c"].descriptor_variant == "c"
    assert configs["d"].descriptor_variant == "d"

    conflict = parser.parse_args(
        [
            "--guided-prototype",
            "--guided-prototype-descriptor-variant",
            "c",
            "--guided-prototype-aggregation-gate",
        ]
    )
    with pytest.raises(ValueError, match="atomic semantic-free ablation"):
        guided_config_from_args(conflict)


def test_descriptor_manifest_groups_overlapping_patches_by_source(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "normal.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "source_file"])
        writer.writeheader()
        writer.writerow({"image": "/data/c00_frame_x0_y0.jpg", "source_file": "frame.jpg"})
        writer.writerow({"image": "/data/c01_frame_x1_y1.jpg", "source_file": "frame.jpg"})

    resolver = SourceGroupResolver([manifest])

    assert resolver.resolve("/symlink/c00_frame_x0_y0.jpg") == "frame.jpg"
    assert resolver.resolve("/symlink/c01_frame_x1_y1.jpg") == "frame.jpg"
    assert resolver.ids(
        ["/symlink/c00_frame_x0_y0.jpg", "/symlink/c01_frame_x1_y1.jpg"]
    ).unique().numel() == 1


class _ConfigOnlyINP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prototype_token = nn.Parameter(torch.randn(6, 8))

    def gather_loss(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        return query.sum() * 0.0 + keys.sum() * 0.0

    def forward(self, images: torch.Tensor):
        return images


def test_descriptor_d_configuration_registers_frozen_memory_and_prior_normalizer() -> None:
    parser = argparse.ArgumentParser()
    add_guided_prototype_args(parser)
    args = parser.parse_args(
        [
            "--guided-prototype",
            "--guided-prototype-descriptor-variant",
            "d",
            "--guided-prototype-multiscale-direct",
            "--guided-prototype-descriptor-memory-size",
            "8",
            "--guided-prototype-descriptor-topk",
            "2",
        ]
    )
    model = _ConfigOnlyINP()

    configure_guided_prototypes(model, args, "inpformer")

    assert isinstance(model.guided_normal_descriptor_memory, NormalDescriptorMemory)
    assert hasattr(model, "guided_prior_normalizer")
    assert model._guided_prototype_config.free_prototypes
    assert model._guided_prototype_config.descriptor_variant == "d"
