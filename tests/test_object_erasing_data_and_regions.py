from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_reconstruct_object_erasing.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("train_reconstruct_object_erasing", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_manifest(path: Path, image_path: Path, rows: int, source: str) -> None:
    payload = {
        "source": source,
        "image_path": str(image_path),
        "annotation_path": "",
        "width": 64,
        "height": 64,
        "boxes": [{"bbox": [16, 16, 32, 32]}],
        "all_object_boxes": [[16, 16, 32, 32], [40, 40, 48, 48]],
    }
    with path.open("w", encoding="utf-8") as handle:
        for idx in range(rows):
            row = dict(payload)
            row["image_path"] = str(image_path)
            row["row_id"] = idx
            handle.write(json.dumps(row) + "\n")


def test_multi_manifest_quota_and_manifest_clean_mask(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (64, 64), "white").save(image_path)
    first = tmp_path / "visdrone.jsonl"
    second = tmp_path / "soda.jsonl"
    _write_manifest(first, image_path, rows=12, source="visdrone")
    _write_manifest(second, image_path, rows=12, source="soda_a")

    dataset = MODULE.SmallObjectManifestDataset(
        [first, second],
        image_size=64,
        max_images=8,
        max_boxes=4,
        seed=7,
        source_weights=(3.0, 1.0),
        crop_augment=False,
    )

    assert dataset.source_counts == {"visdrone": 6, "soda_a": 2}
    _, _, _, all_object_mask, _ = dataset[0]
    assert all_object_mask[0, 20, 20] == 1
    assert all_object_mask[0, 44, 44] == 1
    assert all_object_mask[0, 0, 0] == 0


def test_prior_manifest_can_preserve_exact_row_order(tmp_path: Path) -> None:
    image_a = tmp_path / "a.png"
    image_z = tmp_path / "z.png"
    Image.new("RGB", (64, 64), "white").save(image_a)
    Image.new("RGB", (64, 64), "white").save(image_z)
    manifest = tmp_path / "prior.jsonl"
    rows = []
    for idx, image_path in enumerate((image_z, image_a)):
        rows.append(
            {
                "source": "visdrone",
                "image_path": str(image_path),
                "annotation_path": "",
                "width": 64,
                "height": 64,
                "boxes": [{"bbox": [16, 16, 32, 32], "token_prior_bin": idx}],
                "all_object_boxes": [[16, 16, 32, 32]],
            }
        )
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    dataset = MODULE.SmallObjectManifestDataset(
        [manifest], image_size=64, max_images=0, max_boxes=1, seed=7, preserve_row_order=True
    )

    assert [Path(str(row["image_path"])).name for row in dataset.rows] == ["z.png", "a.png"]


def _loss_args(instance_local: bool) -> Namespace:
    return Namespace(
        image_size=56,
        det_clean_bg_dilate=0,
        ring_radius=1,
        det_background_mode="clean",
        det_objectness_min_weight=0.05,
        det_objectness_power=2.0,
        det_size_weighting=False,
        det_size_focus_min_side=14.0,
        det_size_focus_max_side=56.0,
        det_size_focus_weight=2.0,
        det_core_ring_erasing=True,
        det_core_frac=0.6,
        det_hard_residual_mining=False,
        det_hard_residual_frac=0.5,
        det_hard_residual_min_weight=0.1,
        det_instance_local_erasing=instance_local,
        lambda_det_bg=1.0,
        lambda_obj_bg=1.0,
        lambda_obj_sep=0.1,
        lambda_smooth=0.1,
        lambda_score_rank=0.0,
        lambda_score_cover=0.0,
        lambda_boundary_preserve=0.0,
        sep_sim_margin=0.3,
        score_rank_margin=0.05,
        score_cover_margin=0.02,
        score_rank_background_mode="local_ring",
        score_rank_global_topk_frac=0.01,
        score_rank_object_low_frac=0.25,
    )


def test_instance_local_erasing_counts_each_box() -> None:
    torch.manual_seed(3)
    enc = torch.nn.functional.normalize(torch.randn(1, 8, 4, 4), dim=1)
    dec = torch.nn.functional.normalize(torch.randn(1, 8, 4, 4), dim=1)
    boxes = torch.tensor([[[0.0, 0.0, 14.0, 14.0], [42.0, 42.0, 56.0, 56.0]]])
    valid = torch.tensor([[True, True]])
    all_object_mask = torch.zeros(1, 1, 56, 56)
    all_object_mask[:, :, :14, :14] = 1
    all_object_mask[:, :, 42:, 42:] = 1

    local_loss, local_diag = MODULE.object_erasing_losses(
        enc,
        dec,
        boxes,
        valid,
        _loss_args(True),
        all_object_mask=all_object_mask,
    )
    union_loss, union_diag = MODULE.object_erasing_losses(
        enc,
        dec,
        boxes,
        valid,
        _loss_args(False),
        all_object_mask=all_object_mask,
    )

    assert torch.isfinite(local_loss)
    assert torch.isfinite(union_loss)
    assert local_diag["det_regions"] == 2
    assert union_diag["det_regions"] == 1
    assert local_diag["det_regions_skipped"] == 0


def test_global_clean_rank_uses_hard_background_and_object_low_tail() -> None:
    torch.manual_seed(4)
    enc = torch.nn.functional.normalize(torch.randn(1, 8, 4, 4), dim=1)
    dec = torch.nn.functional.normalize(torch.randn(1, 8, 4, 4), dim=1)
    boxes = torch.tensor([[[14.0, 14.0, 42.0, 42.0]]])
    valid = torch.tensor([[True]])
    all_object_mask = torch.zeros(1, 1, 56, 56)
    all_object_mask[:, :, 14:42, 14:42] = 1
    score_map = torch.full((1, 1, 4, 4), 0.50, requires_grad=True)
    with torch.no_grad():
        score_map[:, :, 1:3, 1:3] = 0.40
    args = _loss_args(False)
    args.lambda_score_rank = 1.0
    args.score_rank_background_mode = "global_clean"
    args.score_rank_global_topk_frac = 0.25
    args.score_rank_object_low_frac = 0.5

    _, rank_loss, _, diag = MODULE.object_erasing_loss_terms(
        enc,
        dec,
        boxes,
        valid,
        args,
        all_object_mask=all_object_mask,
        score_map=score_map,
    )
    rank_loss.backward()

    assert rank_loss.item() > 0
    assert diag["score_rank_bg_mean"] == 0.5
    assert abs(diag["score_rank_obj_mean"] - 0.4) < 1e-6
    assert diag["score_rank_active_fraction"] == 1.0
    # Hard-background scores are detached; rank only pushes the object side.
    assert torch.count_nonzero(score_map.grad[:, :, 0, :]) == 0
    assert torch.count_nonzero(score_map.grad[:, :, 1:3, 1:3]) > 0


def test_hard_residual_mining_focuses_low_residual_object_tokens() -> None:
    enc = torch.zeros(1, 2, 4, 4)
    enc[:, 0] = 1.0
    dec = enc.clone()
    # The four center object tokens have residuals 0, 0, 1, 1.  With a 50%
    # hard fraction and zero floor, only the two still-well-reconstructed tokens
    # should carry legacy OE weight.
    dec[:, :, 2, 1:3] = torch.tensor([0.0, 1.0]).view(1, 2, 1)
    boxes = torch.tensor([[[14.0, 14.0, 42.0, 42.0]]])
    valid = torch.tensor([[True]])
    all_object_mask = torch.zeros(1, 1, 56, 56)
    all_object_mask[:, :, 14:42, 14:42] = 1
    args = _loss_args(False)
    args.det_hard_residual_mining = True
    args.det_hard_residual_min_weight = 0.0

    legacy_loss, _, _, diag = MODULE.object_erasing_loss_terms(
        enc,
        dec,
        boxes,
        valid,
        args,
        all_object_mask=all_object_mask,
    )

    assert torch.isfinite(legacy_loss)
    assert abs(diag["hard_residual_weight_mean"] - 0.5) < 1e-6
    assert abs(diag["hard_residual_score_mean"] - 0.5) < 1e-6


def test_warmup_cosine_learning_rate_hits_control_points() -> None:
    fn = MODULE.scheduled_learning_rate
    assert abs(fn(1e-5, 1e-6, 0, 180, "cosine", 18, 0.2) - 2e-6) < 1e-15
    assert abs(fn(1e-5, 1e-6, 18, 180, "cosine", 18, 0.2) - 1e-5) < 1e-15
    assert abs(fn(1e-5, 1e-6, 99, 180, "cosine", 18, 0.2) - 5.5e-6) < 1e-12
    assert abs(fn(1e-5, 1e-6, 180, 180, "cosine", 18, 0.2) - 1e-6) < 1e-15


def test_late_cosine_learning_rate_stays_fixed_until_decay_phase() -> None:
    fn = MODULE.scheduled_learning_rate
    assert abs(fn(1e-5, 1e-6, 1, 180, "cosine", decay_start_step=144) - 1e-5) < 1e-15
    assert abs(fn(1e-5, 1e-6, 144, 180, "cosine", decay_start_step=144) - 1e-5) < 1e-15
    assert abs(fn(1e-5, 1e-6, 162, 180, "cosine", decay_start_step=144) - 5.5e-6) < 1e-15
    assert abs(fn(1e-5, 1e-6, 180, 180, "cosine", decay_start_step=144) - 1e-6) < 1e-15
    assert abs(fn(5e-5, 5e-6, 162, 180, "cosine", decay_start_step=144) - 2.75e-5) < 1e-15


def test_cosine_learning_rate_can_reach_min_before_training_ends() -> None:
    fn = MODULE.scheduled_learning_rate
    kwargs = {
        "base_lr": 1e-5,
        "min_lr": 1e-6,
        "total_steps": 360,
        "schedule": "cosine",
        "decay_start_step": 144,
        "decay_end_step": 180,
    }
    assert abs(fn(step=144, **kwargs) - 1e-5) < 1e-15
    assert abs(fn(step=162, **kwargs) - 5.5e-6) < 1e-15
    assert abs(fn(step=180, **kwargs) - 1e-6) < 1e-15
    assert abs(fn(step=270, **kwargs) - 1e-6) < 1e-15
    assert abs(fn(step=360, **kwargs) - 1e-6) < 1e-15


def test_oe_phase_controls_switch_at_step_91() -> None:
    fn = MODULE.activated_step_value
    assert fn(1.08, 90, 91, inactive_value=0.0) == 0.0
    assert fn(1.08, 91, 91, inactive_value=0.0) == 1.08
    assert fn(0.24, 90, 91, inactive_value=0.0) == 0.0
    assert fn(0.24, 91, 91, inactive_value=0.0) == 0.24
    assert fn(0.0, 90, 91, inactive_value=1.0) == 1.0
    assert fn(0.0, 91, 91, inactive_value=1.0) == 0.0


def test_target_background_composition_is_deterministic_and_local() -> None:
    source = torch.zeros(2, 3, 16, 16)
    source[0, :, 2:6, 3:8] = 1.0
    source[1, :, 7:11, 8:13] = 2.0
    boxes = torch.zeros(2, 4, 4)
    boxes[0, 0] = torch.tensor([3.0, 2.0, 8.0, 6.0])
    boxes[1, 0] = torch.tensor([8.0, 7.0, 13.0, 11.0])
    valid = torch.zeros(2, 4, dtype=torch.bool)
    valid[:, 0] = True
    background = torch.full((2, 3, 16, 16), -1.0)

    first = MODULE.compose_target_background_batch(
        source, boxes, valid, background, seed=17, max_pastes=2, feather=1
    )
    second = MODULE.compose_target_background_batch(
        source, boxes, valid, background, seed=17, max_pastes=2, feather=1
    )
    images, pasted_boxes, pasted_valid, pasted_mask, diag = first

    assert torch.equal(images, second[0])
    assert torch.equal(pasted_boxes, second[1])
    assert torch.equal(pasted_valid, second[2])
    assert torch.equal(pasted_mask, second[3])
    assert pasted_valid[:, 0].all()
    assert not pasted_valid[:, 1:].any()
    assert torch.count_nonzero(pasted_mask) == 40
    assert torch.all(images[pasted_mask.expand_as(images) == 0] == -1.0)
    assert torch.all(background == -1.0), "input backgrounds must not be modified in place"
    assert diag["target_bg_pastes"] == 2.0
    assert diag["target_bg_samples_with_paste"] == 2.0


def test_target_background_composition_limits_pastes_without_overlap() -> None:
    source = torch.ones(1, 3, 24, 24)
    boxes = torch.tensor([[[0.0, 0.0, 4.0, 4.0], [4.0, 0.0, 8.0, 4.0], [8.0, 0.0, 12.0, 4.0]]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    background = torch.zeros(1, 3, 24, 24)

    _, pasted_boxes, pasted_valid, pasted_mask, diag = MODULE.compose_target_background_batch(
        source, boxes, valid, background, seed=5, max_pastes=2, feather=0
    )

    assert pasted_valid.sum().item() == 2
    first = tuple(int(value) for value in pasted_boxes[0, 0].tolist())
    second = tuple(int(value) for value in pasted_boxes[0, 1].tolist())
    assert not MODULE._boxes_overlap(first, second)
    assert pasted_mask.sum().item() == 32
    assert diag["target_bg_pastes_per_image"] == 2.0


def test_ellipse_alpha_removes_box_corners_and_keeps_center() -> None:
    alpha = MODULE._soft_box_alpha(
        15,
        21,
        4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        shape="ellipse",
    )

    assert alpha.shape == (1, 15, 21)
    assert alpha[0, 0, 0] == 0
    assert alpha[0, 7, 10] == 1
    assert 0 < torch.count_nonzero(alpha) < alpha.numel()


def test_paired_normal_prototype_context_matches_composition_mapping() -> None:
    normal = torch.tensor(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[3.0, 30.0], [4.0, 40.0]],
            [[5.0, 50.0], [6.0, 60.0]],
        ],
        requires_grad=True,
    )

    paired = MODULE.paired_normal_prototype_context(normal, target_batch=5)

    assert paired.shape == (5, 2, 2)
    assert torch.equal(paired[0], normal[0])
    assert torch.equal(paired[1], normal[1])
    assert torch.equal(paired[2], normal[2])
    assert torch.equal(paired[3], normal[0])
    assert torch.equal(paired[4], normal[1])
    assert not paired.requires_grad


def test_paired_normal_prototype_context_rejects_invalid_inputs() -> None:
    with torch.no_grad():
        invalid = torch.zeros(2, 3)
    try:
        MODULE.paired_normal_prototype_context(invalid, target_batch=2)
    except ValueError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("invalid prototype rank must fail")

    try:
        MODULE.paired_normal_prototype_context(torch.zeros(1, 2, 3), target_batch=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("non-positive target batch must fail")


class _NativeAggregationBlock(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale))

    def forward(self, prototypes: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        return prototypes + self.scale * tokens.mean(dim=1, keepdim=True)


class _NativePrototypeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prototype_token = torch.nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        self.aggregation = torch.nn.ModuleList(
            [_NativeAggregationBlock(0.5), _NativeAggregationBlock(0.25)]
        )


def test_native_e2_aggregation_uses_original_prototype_modules() -> None:
    model = _NativePrototypeModel()
    tokens = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 1.0], [4.0, 3.0]],
        ]
    )

    prototypes = MODULE.inpformer_aggregate_prototypes(model, tokens)
    expected = model.prototype_token.unsqueeze(0).repeat(2, 1, 1)
    expected = expected + 0.75 * tokens.mean(dim=1, keepdim=True)

    assert prototypes.shape == (2, 2, 2)
    assert torch.allclose(prototypes, expected)
    prototypes.sum().backward()
    assert model.prototype_token.grad is not None
    assert all(block.scale.grad is not None for block in model.aggregation)


def test_native_e2_aggregation_does_not_require_guided_images() -> None:
    model = _NativePrototypeModel()
    tokens = torch.zeros(1, 4, 2)
    result = MODULE.inpformer_aggregate_prototypes(model, tokens, images=None)
    assert torch.equal(result[0], model.prototype_token)


def test_native_background_excess_ignores_target_and_lower_adaptive_scores() -> None:
    adaptive = torch.tensor(
        [[[[0.9, 0.1], [0.7, 0.5]]]], requires_grad=True
    )
    native = torch.tensor([[[[0.2, 0.3], [0.4, 0.4]]]])
    target = torch.zeros(1, 1, 8, 8)
    target[:, :, :4, :4] = 1.0

    loss, diag = MODULE.native_background_excess_loss(
        adaptive,
        native,
        target,
        minimum_tokens=3,
        target_dilate=0,
    )
    loss.backward()

    assert torch.isclose(loss, torch.tensor((0.3 + 0.1) / 3.0))
    assert adaptive.grad is not None
    assert adaptive.grad[0, 0, 0, 0] == 0
    assert adaptive.grad[0, 0, 0, 1] == 0
    assert adaptive.grad[0, 0, 1, 0] > 0
    assert adaptive.grad[0, 0, 1, 1] > 0
    assert abs(diag["native_bg_excess_active_fraction"] - 2.0 / 3.0) < 1e-6
    assert diag["native_bg_excess_selected_tokens"] == 3.0
