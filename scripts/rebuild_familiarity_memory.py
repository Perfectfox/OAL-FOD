#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fod_recon_ad.data import get_tensor_transform  # noqa: E402
from fod_recon_ad.ext import build_reconstruction_model, load_checkpoint  # noqa: E402
from fod_recon_ad.familiarity_memory_rebuild import (  # noqa: E402
    adaptive_modes_from_stable_teacher,
    boundary_soft_semantic_weights,
    calibrate_cross_view_local_radii,
    calibrate_view_balanced_semantic_mode_radii,
    hierarchical_mode_conditioned_memory,
    local_radius_adjusted_distance,
    stable_view_balanced_soft_semantic_teacher,
    stable_view_balanced_mode_teacher,
    view_balanced_kcenter_memory,
)
from fod_recon_ad.geometric_prior import (  # noqa: E402
    FrozenDepthResidualPrior,
    cross_view_feature_support_gate,
    transfer_unsupported_object_mass_to_background,
)
from fod_recon_ad.prototype_guidance import (  # noqa: E402
    NormalObjectTokenMemory,
    _build_guided_objectness,
    _build_trainable_priors,
    configure_guided_prototypes,
)
from fod_recon_ad.prototype_visualization import tokenize_binary_mask  # noqa: E402
from visualize_familiarity_gate_effect import (  # noqa: E402
    binary_auroc,
    binary_average_precision,
    crop_case,
    guided_args_from_config,
    select_cases,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild a stratified familiarity memory and calibrate it on Val background.")
    parser.add_argument("--fod-root", type=Path, required=True)
    parser.add_argument("--roi-mask", type=Path, required=True)
    parser.add_argument(
        "--eval-fod-root",
        type=Path,
        default=None,
        help="Optional Val dataset root; Normal memory still comes from --fod-root.",
    )
    parser.add_argument("--eval-roi-mask", type=Path, default=None)
    parser.add_argument("--normal-crops-csv", type=Path, required=True)
    parser.add_argument(
        "--view-column",
        default="",
        help=(
            "Optional CSV column used as the normal source-view/fold identity. "
            "The default keeps the historical behavior and uses the image column."
        ),
    )
    parser.add_argument(
        "--skip-eval-diagnostics",
        action="store_true",
        help=(
            "Build and calibrate the memory from Normal data only without opening "
            "a validation split. The saved novelty threshold remains the normal-only "
            "cross-view quantile."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--vlm-prior-checkpoint", type=Path, default=None)
    parser.add_argument("--vlm-prior-physical-weight", type=float, default=0.0)
    parser.add_argument(
        "--vlm-prior-fusion-mode",
        choices=("global", "low_confidence_correction"),
        default="global",
    )
    parser.add_argument("--vlm-prior-correction-max-weight", type=float, default=0.5)
    parser.add_argument("--vlm-prior-correction-physical-margin", type=float, default=0.15)
    parser.add_argument("--vlm-prior-correction-vlm-margin", type=float, default=0.15)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--external-repo", type=Path, default=PROJECT_ROOT.parent / "Dinomaly")
    parser.add_argument("--inpformer-repo", type=Path, default=PROJECT_ROOT.parent / "INP-Former")
    parser.add_argument("--encoder", default="dinov2reg_vit_base_14")
    parser.add_argument("--inp-num", type=int, default=6)
    parser.add_argument("--crop-size", type=int, default=448)
    parser.add_argument("--model-input-size", type=int, default=672)
    parser.add_argument("--stride", type=int, default=224)
    parser.add_argument("--distances", nargs="+", default=["05", "10", "15", "20", "25", "30"])
    parser.add_argument("--memory-size", type=int, default=256)
    parser.add_argument(
        "--allow-memory-resize",
        action="store_true",
        help=(
            "Resize only the familiarity-memory buffers while loading all other "
            "source-checkpoint weights strictly. Intended for controlled memory-capacity "
            "ablations; the rebuilt memory is still derived only from Normal data."
        ),
    )
    parser.add_argument("--max-candidates-per-group", type=int, default=4096)
    parser.add_argument(
        "--adaptive-mode-teacher",
        action="store_true",
        help=(
            "Select one or two effective Normal modes per semantic group from "
            "stable candidate-pool evidence and final-memory support."
        ),
    )
    parser.add_argument("--adaptive-mode-min-memory-members", type=int, default=2)
    parser.add_argument("--adaptive-mode-min-memory-views", type=int, default=2)
    parser.add_argument("--mode-teacher-groups", default="2,2,2")
    parser.add_argument("--mode-teacher-bootstrap-repeats", type=int, default=12)
    parser.add_argument("--mode-teacher-bootstrap-fraction", type=float, default=0.80)
    parser.add_argument("--mode-teacher-max-candidates-per-view-group", type=int, default=2048)
    parser.add_argument("--mode-teacher-min-mode-fraction", type=float, default=0.10)
    parser.add_argument("--mode-teacher-min-assignment-stability", type=float, default=0.75)
    parser.add_argument("--mode-teacher-min-separation-ratio", type=float, default=1.0)
    parser.add_argument("--mode-teacher-margin-quantile", type=float, default=0.10)
    parser.add_argument("--soft-semantic-boundary", action="store_true")
    parser.add_argument("--soft-semantic-confidence-margin", type=float, default=0.15)
    parser.add_argument(
        "--prior-simplification",
        choices=(
            "none",
            "feature_only",
            "feature_single",
            "feature_minimal",
            "simple_feature",
            "simple_depth",
        ),
        default="none",
        help=(
            "Normal-teacher-only prior ablation. feature_only removes image "
            "Sobel/Laplacian responses; feature_single also removes multiscale "
            "responses and robust EMA normalization; feature_minimal further "
            "removes texture suppression, the common floor, and group weights."
        ),
    )
    parser.add_argument(
        "--depth-prior-model",
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="Frozen relative-depth model used only by --prior-simplification simple_depth.",
    )
    parser.add_argument("--depth-prior-cache-dir", type=Path, default=None)
    parser.add_argument("--depth-prior-local-kernel", type=int, default=5)
    parser.add_argument("--depth-cross-view-gate", action="store_true")
    parser.add_argument("--depth-cross-view-gate-reference-size", type=int, default=2048)
    parser.add_argument("--depth-cross-view-gate-radius-quantile", type=float, default=0.95)
    parser.add_argument("--depth-cross-view-gate-temperature", type=float, default=0.10)
    parser.add_argument("--depth-cross-view-gate-min-weight", type=float, default=0.25)
    parser.add_argument("--depth-cross-view-gate-query-chunk-size", type=int, default=1024)
    parser.add_argument("--hierarchical-atlas-memory", action="store_true")
    parser.add_argument("--hierarchical-memory-group-quota-power", type=float, default=0.70)
    parser.add_argument(
        "--hierarchical-memory-fixed-group-quotas",
        default="",
        help="Optional exact total quotas for the three semantic groups, e.g. 276,37,39.",
    )
    parser.add_argument("--local-radius-quantile", type=float, default=0.95)
    parser.add_argument("--local-radius-min", type=float, default=1e-3)
    parser.add_argument("--roi-token-threshold", type=float, default=0.50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260718)
    return parser.parse_args()


def parse_mode_groups(text: str) -> tuple[int, int, int]:
    values = tuple(int(item) for item in text.replace(",", " ").split() if item.strip())
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError(f"--mode-teacher-groups expects three positive integers, got {text!r}.")
    return values  # type: ignore[return-value]


def parse_optional_group_quotas(text: str, size: int) -> tuple[int, int, int] | None:
    if not text.strip():
        return None
    values = tuple(int(item) for item in text.replace(",", " ").split() if item.strip())
    if len(values) != 3 or any(value <= 0 for value in values) or sum(values) != size:
        raise ValueError(
            "--hierarchical-memory-fixed-group-quotas expects three positive integers "
            f"summing to {size}, got {text!r}."
        )
    return values  # type: ignore[return-value]


def extract_target_tokens(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    x = model.encoder.prepare_tokens(images)
    features = []
    for index, block in enumerate(model.encoder.blocks):
        if index > model.target_layers[-1]:
            continue
        x = block(x)
        if index in model.target_layers:
            features.append(x)
    token_start = 1 + model.encoder.num_register_tokens
    if model.remove_class_token:
        features = [feature[:, token_start:, :] for feature in features]
    return model.fuse_feature(features)


def load_normal_crop(
    root: Path,
    row: dict[str, str],
    crop_size: int,
    model_input_size: int,
) -> tuple[Image.Image, np.ndarray]:
    x, y = int(row["x"]), int(row["y"])
    image_path = root / row["image"]
    roi_path = root / row["roi"]
    with Image.open(image_path) as handle:
        image = handle.convert("RGB").crop((x, y, x + crop_size, y + crop_size))
    with Image.open(roi_path) as handle:
        roi = np.asarray(
            handle.convert("L").crop((x, y, x + crop_size, y + crop_size)),
            dtype=np.uint8,
        ) > 0
    image = image.resize((model_input_size, model_input_size), Image.BILINEAR)
    return image, roi


def build_row_view_mapping(
    rows: list[dict[str, str]],
    view_column: str = "",
) -> tuple[list[str], dict[str, int]]:
    """Map a manifest view/fold label to deterministic contiguous integer IDs."""

    column = view_column.strip() or "image"
    if not rows:
        raise ValueError("Normal crop manifest is empty.")
    if column not in rows[0]:
        raise ValueError(f"Normal crop manifest has no view column {column!r}.")
    labels = [str(row[column]).strip() for row in rows]
    if any(not label for label in labels):
        raise ValueError(f"Normal crop manifest contains an empty {column!r} view label.")
    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        raise ValueError("Normal-only cross-view calibration requires at least two views/folds.")
    return unique_labels, {label: index for index, label in enumerate(unique_labels)}


def collect_normal_candidates(
    args: argparse.Namespace,
    model: torch.nn.Module,
    config,
    device: torch.device,
    depth_prior: FrozenDepthResidualPrior | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, int],
    dict[str, int],
]:
    transform = get_tensor_transform()
    features = []
    groups = []
    views = []
    semantic_priors = []
    group_counts = np.zeros(3, dtype=np.int64)
    with args.normal_crops_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    view_labels, label_to_view = build_row_view_mapping(rows, args.view_column)
    view_column = args.view_column.strip() or "image"
    view_counts = np.zeros(len(view_labels), dtype=np.int64)
    with torch.no_grad():
        for index, row in enumerate(rows, start=1):
            image, roi = load_normal_crop(args.fod_root, row, args.crop_size, args.model_input_size)
            batch = transform(image).unsqueeze(0).to(device)
            tokens = extract_target_tokens(model, batch)
            side = int(math.sqrt(tokens.shape[1]))
            objectness_override = (
                None
                if depth_prior is None
                else depth_prior.predict(image, side).to(
                    device=tokens.device, dtype=tokens.dtype
                )
            )
            priors, _ = _build_trainable_priors(
                model,
                tokens,
                config,
                batch,
                objectness_override=objectness_override,
            )
            valid = tokenize_binary_mask(
                roi,
                args.model_input_size,
                side,
                mode="fraction",
                threshold=args.roi_token_threshold,
            ).reshape(-1)
            valid_tensor = torch.from_numpy(valid).to(device=tokens.device, dtype=torch.bool)
            group = priors[0].argmax(dim=-1).cpu().numpy()
            features.append(tokens[0, valid_tensor].detach().float().cpu())
            groups.append(torch.from_numpy(group[valid]).long())
            semantic_priors.append(priors[0, valid_tensor].detach().float().cpu())
            view = label_to_view[str(row[view_column]).strip()]
            views.append(torch.full((int(valid.sum()),), view, dtype=torch.long))
            group_counts += np.bincount(group[valid], minlength=3)
            view_counts[view] += int(valid.sum())
            if index % 16 == 0 or index == len(rows):
                print(f"[normal] {index}/{len(rows)}", flush=True)
    return (
        torch.cat(features, dim=0),
        torch.cat(groups, dim=0),
        torch.cat(views, dim=0),
        torch.cat(semantic_priors, dim=0),
        {str(index): int(value) for index, value in enumerate(group_counts.tolist())},
        {str(index): int(value) for index, value in enumerate(view_counts.tolist())},
    )


def collect_val_tokens(
    args: argparse.Namespace,
    model: torch.nn.Module,
    config,
    memory: NormalObjectTokenMemory,
    local_radii: torch.Tensor,
    local_radius_reference: float,
    device: torch.device,
) -> dict[str, np.ndarray]:
    selector_args = argparse.Namespace(
        fod_root=args.eval_fod_root or args.fod_root,
        roi_mask=args.eval_roi_mask or args.roi_mask,
        distances=args.distances,
        crop_size=args.crop_size,
        stride=args.stride,
        layout="original",
        eval_split="val",
    )
    cases = select_cases(selector_args)
    transform = get_tensor_transform()
    output = {key: [] for key in ("distance", "objectness", "gt", "valid")}
    with torch.no_grad():
        for index, case in enumerate(cases, start=1):
            image, gt, valid = crop_case(case, args.crop_size)
            resized = image.resize((args.model_input_size, args.model_input_size), Image.BILINEAR)
            batch = transform(resized).unsqueeze(0).to(device)
            tokens = extract_target_tokens(model, batch)
            objectness = _build_guided_objectness(model, tokens, config, batch)
            distance, _, _ = local_radius_adjusted_distance(
                tokens,
                memory.bank[: int(memory.count.item())],
                local_radii,
                reference_radius=local_radius_reference,
            )
            side = int(math.sqrt(tokens.shape[1]))
            output["distance"].append(distance[0].float().cpu().numpy())
            output["objectness"].append(objectness[0].float().cpu().numpy())
            output["gt"].append(tokenize_binary_mask(gt, args.model_input_size, side, mode="any").reshape(-1))
            output["valid"].append(
                tokenize_binary_mask(
                    valid,
                    args.model_input_size,
                    side,
                    mode="fraction",
                    threshold=args.roi_token_threshold,
                ).reshape(-1)
            )
            if index % 8 == 0 or index == len(cases):
                print(f"[val] {index}/{len(cases)}", flush=True)
    return {key: np.concatenate(value) for key, value in output.items()}


def gate_metrics(
    arrays: dict[str, np.ndarray],
    threshold: float,
    temperature: float,
    min_weight: float,
    power: float,
) -> dict[str, float | int]:
    valid = arrays["valid"].astype(bool)
    gt = arrays["gt"].astype(bool) & valid
    bg = (~arrays["gt"].astype(bool)) & valid
    novelty_logit = np.clip((arrays["distance"] - threshold) / temperature, -60.0, 60.0)
    novelty = 1.0 / (1.0 + np.exp(-novelty_logit))
    risk = arrays["objectness"] * novelty
    weight = min_weight + (1.0 - min_weight) * np.power(1.0 - risk, power)
    suppression = 1.0 - weight
    labels = gt[valid]
    scores = suppression[valid]
    gt_count = int(gt.sum())
    top = np.argsort(-scores, kind="mergesort")[:gt_count]
    total_budget = float(scores.sum())
    return {
        "novelty_threshold": float(threshold),
        "background_novelty_fpr": float((arrays["distance"][bg] > threshold).mean()),
        "gt_novelty_mean": float(novelty[gt].mean()),
        "background_novelty_mean": float(novelty[bg].mean()),
        "gt_suppression_mean": float(suppression[gt].mean()),
        "background_suppression_mean": float(suppression[bg].mean()),
        "suppression_auroc": float(binary_auroc(labels, scores)),
        "suppression_ap": float(binary_average_precision(labels, scores)),
        "top_n_gt_precision": float(labels[top].mean()),
        "gt_suppression_budget_share": float(suppression[gt].sum() / total_budget),
        "gt_tokens": gt_count,
        "background_tokens": int(bg.sum()),
    }


def save_checkpoint(
    model: torch.nn.Module,
    memory: NormalObjectTokenMemory,
    output: Path,
    metadata: dict,
    local_radii: torch.Tensor,
    local_radius_reference: float,
) -> None:
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    state["guided_normal_object_memory.bank"] = memory.bank.detach().cpu()
    state["guided_normal_object_memory.count"] = memory.count.detach().cpu()
    state["guided_normal_object_memory.bank_group_ids"] = memory.bank_group_ids.detach().cpu()
    state["guided_normal_object_memory.calibrated_count"] = memory.calibrated_count.detach().cpu()
    state["guided_normal_object_memory.calibration_group_count"] = memory.calibration_group_count.detach().cpu()
    state["guided_normal_object_memory.novelty_threshold"] = memory.novelty_threshold.detach().cpu()
    state["guided_normal_object_memory.local_radii"] = local_radii.detach().float().cpu()
    state["guided_normal_object_memory.local_radius_reference"] = torch.tensor(
        float(local_radius_reference), dtype=torch.float32
    )
    state["guided_normal_object_memory.local_radius_enabled"] = torch.tensor(True)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state, "metadata": metadata}, output)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    model, _ = build_reconstruction_model(
        architecture="inpformer",
        dinomaly_repo=args.external_repo,
        inpformer_repo=args.inpformer_repo,
        encoder=args.encoder,
        device=device,
        inp_num=args.inp_num,
    )
    guided_args = guided_args_from_config(args.model_config, args.model_input_size)
    if args.vlm_prior_checkpoint is not None:
        guided_args.guided_prototype_vlm_prior_checkpoint = args.vlm_prior_checkpoint
        guided_args.guided_prototype_vlm_prior_physical_weight = args.vlm_prior_physical_weight
        guided_args.guided_prototype_vlm_prior_fusion_mode = args.vlm_prior_fusion_mode
        guided_args.guided_prototype_vlm_prior_correction_max_weight = (
            args.vlm_prior_correction_max_weight
        )
        guided_args.guided_prototype_vlm_prior_correction_physical_margin = (
            args.vlm_prior_correction_physical_margin
        )
        guided_args.guided_prototype_vlm_prior_correction_vlm_margin = (
            args.vlm_prior_correction_vlm_margin
        )
    source_memory_size = int(guided_args.guided_prototype_familiarity_memory_size)
    resize_memory = source_memory_size != args.memory_size
    if resize_memory and not args.allow_memory_resize:
        raise ValueError(
            "Controlled P1 rebuild requires the checkpoint memory size to remain unchanged "
            "unless --allow-memory-resize is explicitly provided."
        )
    if resize_memory:
        guided_args.guided_prototype_familiarity_memory_size = args.memory_size
        guided_args.guided_prototype_familiarity_min_count = args.memory_size
    configure_guided_prototypes(model, guided_args, "inpformer")
    if resize_memory:
        payload = torch.load(args.checkpoint, map_location=device)
        state = payload.get("model", payload) if isinstance(payload, dict) else payload
        memory_prefix = "guided_normal_object_memory."
        state_without_memory = {
            key: value for key, value in state.items() if not key.startswith(memory_prefix)
        }
        incompatible = model.load_state_dict(state_without_memory, strict=False)
        allowed_missing = {
            key for key in model.state_dict() if key.startswith(memory_prefix)
        }
        disallowed_missing = sorted(set(incompatible.missing_keys) - allowed_missing)
        unexpected = sorted(incompatible.unexpected_keys)
        if disallowed_missing or unexpected:
            raise RuntimeError(
                "Memory-resize checkpoint mismatch outside familiarity buffers: "
                f"missing={disallowed_missing}, unexpected={unexpected}"
            )
    else:
        load_checkpoint(model, args.checkpoint, device, strict=True)
    model.eval()
    config = model._guided_prototype_config
    if args.prior_simplification != "none":
        changes = {
            "feature_texture_weight": 1.0,
            "image_texture_weight": 0.0,
            "feature_object_weight": 1.0,
            "image_object_weight": 0.0,
        }
        if args.prior_simplification in {
            "feature_single",
            "feature_minimal",
            "simple_feature",
            "simple_depth",
        }:
            changes["multiscale_direct"] = False
        if args.prior_simplification in {
            "feature_minimal",
            "simple_feature",
            "simple_depth",
        }:
            changes.update(
                object_texture_suppress=0.0,
                prior_floor=0.0,
                group_weights=(1.0, 1.0, 1.0),
            )
        config = replace(config, **changes)
        model._guided_prototype_config = config
    depth_prior = None
    if args.prior_simplification == "simple_depth":
        depth_prior = FrozenDepthResidualPrior.from_pretrained(
            args.depth_prior_model,
            device=device,
            cache_dir=args.depth_prior_cache_dir,
            local_kernel=args.depth_prior_local_kernel,
        )
    memory = model.guided_normal_object_memory

    (
        candidates,
        candidate_groups,
        candidate_views,
        candidate_priors,
        candidate_counts,
        candidate_view_counts,
    ) = collect_normal_candidates(args, model, config, device, depth_prior)
    cross_view_gate_diagnostics = {}
    if args.depth_cross_view_gate:
        if args.prior_simplification != "simple_depth":
            raise ValueError(
                "--depth-cross-view-gate requires --prior-simplification simple_depth."
            )
        support_gate, cross_view_gate_diagnostics = cross_view_feature_support_gate(
            candidates,
            candidate_views,
            reference_size_per_view=args.depth_cross_view_gate_reference_size,
            radius_quantile=args.depth_cross_view_gate_radius_quantile,
            temperature=args.depth_cross_view_gate_temperature,
            min_weight=args.depth_cross_view_gate_min_weight,
            query_chunk_size=args.depth_cross_view_gate_query_chunk_size,
            seed=args.seed,
            device=device,
        )
        before_counts = dict(candidate_counts)
        object_probability_mean_before = float(candidate_priors[:, 2].mean())
        candidate_priors = transfer_unsupported_object_mass_to_background(
            candidate_priors, support_gate
        )
        candidate_groups = candidate_priors.argmax(dim=1)
        candidate_counts = {
            str(index): int((candidate_groups == index).sum())
            for index in range(3)
        }
        cross_view_gate_diagnostics.update(
            candidate_group_counts_before=before_counts,
            candidate_group_counts_after=candidate_counts,
            object_probability_mean_before=object_probability_mean_before,
            object_probability_mean_after=float(candidate_priors[:, 2].mean()),
        )
    mode_groups = parse_mode_groups(args.mode_teacher_groups)
    if mode_groups != tuple(config.groups):
        raise ValueError(
            f"Stable teacher groups {mode_groups} do not match model prototype groups {config.groups}."
        )
    if args.soft_semantic_boundary:
        semantic_weights = boundary_soft_semantic_weights(
            candidate_priors, args.soft_semantic_confidence_margin
        )
        teacher_builder = stable_view_balanced_soft_semantic_teacher
        teacher_semantic_input = semantic_weights
    else:
        semantic_weights = torch.nn.functional.one_hot(
            candidate_groups, num_classes=3
        ).float()
        teacher_builder = stable_view_balanced_mode_teacher
        teacher_semantic_input = candidate_groups
    (
        mode_centers,
        mode_group_reliability,
        mode_margin_floor,
        mode_margin_scale,
        mode_diagnostics,
    ) = teacher_builder(
        candidates,
        candidate_views,
        teacher_semantic_input,
        mode_groups,
        bootstrap_repeats=args.mode_teacher_bootstrap_repeats,
        bootstrap_fraction=args.mode_teacher_bootstrap_fraction,
        max_candidates_per_view_group=args.mode_teacher_max_candidates_per_view_group,
        min_mode_fraction=args.mode_teacher_min_mode_fraction,
        min_assignment_stability=args.mode_teacher_min_assignment_stability,
        min_separation_ratio=args.mode_teacher_min_separation_ratio,
        margin_quantile=args.mode_teacher_margin_quantile,
        seed=args.seed,
        device=device,
    )
    atlas_diagnostics = None
    bank_within_modes = torch.full((args.memory_size,), -1, dtype=torch.long)
    bank_mode_probability = torch.zeros(args.memory_size, dtype=torch.float32)
    bank_mode_distance = torch.zeros(args.memory_size, dtype=torch.float32)
    if args.hierarchical_atlas_memory:
        fixed_group_quotas = parse_optional_group_quotas(
            args.hierarchical_memory_fixed_group_quotas,
            args.memory_size,
        )
        (
            bank,
            bank_views,
            bank_groups,
            bank_within_modes,
            bank_mode_probability,
            bank_mode_distance,
            atlas_diagnostics,
        ) = hierarchical_mode_conditioned_memory(
            candidates,
            candidate_views,
            semantic_weights,
            mode_centers,
            mode_groups,
            args.memory_size,
            mode_temperature=0.10,
            group_reliability=mode_group_reliability,
            group_quota_power=args.hierarchical_memory_group_quota_power,
            fixed_group_quotas=fixed_group_quotas,
            device=device,
        )
    else:
        bank, bank_views, bank_groups = view_balanced_kcenter_memory(
            candidates,
            candidate_views,
            args.memory_size,
            semantic_group_ids=candidate_groups,
            max_candidates_per_view=args.max_candidates_per_group,
            seed=args.seed,
            device=device,
        )
    adaptive_payload = None
    adaptive_diagnostics = None
    adaptive_mode_radii = None
    adaptive_mode_member_counts = None
    adaptive_radius_diagnostics = None
    if args.adaptive_mode_teacher:
        (
            adaptive_centers,
            adaptive_groups,
            adaptive_slot_to_mode,
            adaptive_mode_group_ids,
            adaptive_diagnostics,
        ) = adaptive_modes_from_stable_teacher(
            mode_centers,
            mode_groups,
            mode_diagnostics,
            bank,
            bank_views,
            bank_groups,
            min_memory_members_per_mode=args.adaptive_mode_min_memory_members,
            min_memory_views_per_mode=args.adaptive_mode_min_memory_views,
        )
        adaptive_payload = {
            "adaptive_mode_centers": adaptive_centers.numpy().astype(np.float32),
            "adaptive_mode_groups": np.asarray(adaptive_groups, dtype=np.int64),
            "adaptive_slot_groups": np.asarray(mode_groups, dtype=np.int64),
            "adaptive_slot_to_mode": adaptive_slot_to_mode.numpy().astype(np.int64),
            "adaptive_mode_group_ids": adaptive_mode_group_ids.numpy().astype(np.int64),
            "adaptive_mode_construction_version": np.asarray(
                adaptive_diagnostics["version"]
            ),
        }
        (
            adaptive_mode_radii,
            adaptive_mode_member_counts,
            adaptive_radius_diagnostics,
        ) = calibrate_view_balanced_semantic_mode_radii(
            candidates,
            candidate_views,
            semantic_weights,
            adaptive_centers,
            adaptive_groups,
            quantile=args.local_radius_quantile,
            min_radius=args.local_radius_min,
        )
        adaptive_payload.update(
            adaptive_mode_radii=adaptive_mode_radii.numpy().astype(np.float32),
            adaptive_mode_radius_member_counts=(
                adaptive_mode_member_counts.numpy().astype(np.int64)
            ),
            adaptive_mode_radius_calibration_version=np.asarray(
                adaptive_radius_diagnostics["version"]
            ),
        )
    local_radii, local_radius_reference, normal_only_threshold, radius_diagnostics = calibrate_cross_view_local_radii(
        candidates.to(device),
        candidate_views.to(device),
        bank.to(device),
        bank_views.to(device),
        radius_quantile=args.local_radius_quantile,
        threshold_quantile=config.familiarity_calibration_quantile,
        min_radius=args.local_radius_min,
    )
    with torch.no_grad():
        memory.bank.copy_(bank.to(memory.bank))
        # bank_group_ids are source IDs used by cross-view calibration.  Semantic
        # prior groups are diagnostics only and must not overwrite these IDs.
        memory.bank_group_ids.copy_(bank_views.to(memory.bank_group_ids))
        memory.count.fill_(args.memory_size)
        memory.calibrated_count.fill_(args.memory_size)
        memory.calibration_group_count.fill_(int(torch.unique(bank_views).numel()))

    selected_threshold = max(
        float(config.familiarity_novelty_floor), float(normal_only_threshold)
    )
    if args.skip_eval_diagnostics:
        selected = {
            "novelty_threshold": selected_threshold,
            "selection_source": "normal_only_cross_view_quantile",
            "eval_diagnostics_skipped": True,
        }
    else:
        arrays = collect_val_tokens(
            args, model, config, memory, local_radii.to(device), local_radius_reference, device
        )
        selected = gate_metrics(
            arrays,
            selected_threshold,
            config.familiarity_temperature,
            config.aggregation_min_weight,
            config.aggregation_power,
        )
    with torch.no_grad():
        memory.novelty_threshold.fill_(float(selected["novelty_threshold"]))

    diagnostics = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "memory_capacity_ablation": {
            "source_checkpoint_memory_size": source_memory_size,
            "rebuilt_memory_size": args.memory_size,
            "memory_resized": resize_memory,
            "non_memory_checkpoint_state_loaded_strictly": True,
        },
        "candidate_group_counts": candidate_counts,
        "candidate_view_counts": candidate_view_counts,
        "vlm_prior": dict(getattr(model, "_guided_vlm_prior_diag", {})),
        "depth_prior": {} if depth_prior is None else depth_prior.diagnostics(),
        "depth_cross_view_gate": cross_view_gate_diagnostics,
        "memory_group_counts": {
            str(index): int((bank_groups == index).sum()) for index in range(3)
        },
        "memory_view_counts": {
            str(index): int((bank_views == index).sum())
            for index in torch.unique(bank_views, sorted=True).tolist()
        },
        "hierarchical_normal_atlas": atlas_diagnostics,
        "stable_mode_teacher": mode_diagnostics,
        "adaptive_mode_teacher": adaptive_diagnostics,
        "adaptive_mode_radius_calibration": adaptive_radius_diagnostics,
        "local_radius": {
            **radius_diagnostics,
            "reference_radius": local_radius_reference,
            "calibration_source": (
                f"{int(torch.unique(bank_views).numel())}_normal_train_views_leave_one_view_out"
            ),
            "val_gt_used_for_threshold": False,
        },
        "selected": selected,
    }
    (args.output_dir / "memory_rebuild_summary.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    stratified_payload = dict(
        bank=bank.numpy(),
        source_view_ids=bank_views.numpy(),
        semantic_group_ids=bank_groups.numpy(),
        mode_within_group_ids=bank_within_modes.numpy(),
        mode_probabilities=bank_mode_probability.numpy(),
        mode_distances=bank_mode_distance.numpy(),
        local_radii=local_radii.numpy(),
        local_radius_reference=np.asarray(local_radius_reference, dtype=np.float32),
        novelty_threshold=np.asarray(float(selected["novelty_threshold"]), dtype=np.float32),
    )
    if adaptive_payload is not None:
        stratified_payload.update(adaptive_payload)
    np.savez_compressed(
        args.output_dir / "stratified_memory.npz",
        **stratified_payload,
    )
    stable_payload = dict(
        mode_teacher_centers=mode_centers.numpy().astype(np.float32),
        mode_teacher_groups=np.asarray(mode_groups, dtype=np.int64),
        mode_teacher_group_reliability=mode_group_reliability.numpy().astype(np.float32),
        mode_teacher_margin_floor=mode_margin_floor.numpy().astype(np.float32),
        mode_teacher_margin_scale=mode_margin_scale.numpy().astype(np.float32),
        mode_teacher_construction_version=np.asarray(mode_diagnostics["version"]),
        source_view_count=np.asarray(
            int(torch.unique(candidate_views, sorted=True).numel()), dtype=np.int64
        ),
        normal_only=np.asarray(True),
    )
    if adaptive_payload is not None:
        stable_payload.update(adaptive_payload)
        stable_payload.update(
            bank=bank.numpy(),
            source_view_ids=bank_views.numpy(),
            semantic_group_ids=bank_groups.numpy(),
        )
    np.savez_compressed(
        args.output_dir / "stable_mode_teacher.npz",
        **stable_payload,
    )
    (args.output_dir / "stable_mode_teacher_summary.json").write_text(
        json.dumps(mode_diagnostics, indent=2), encoding="utf-8"
    )
    if args.hierarchical_atlas_memory:
        np.savez_compressed(
            args.output_dir / "hierarchical_normal_atlas.npz",
            bank=bank.numpy(),
            source_view_ids=bank_views.numpy(),
            semantic_group_ids=bank_groups.numpy(),
            mode_within_group_ids=bank_within_modes.numpy(),
            mode_probabilities=bank_mode_probability.numpy(),
            mode_distances=bank_mode_distance.numpy(),
            local_radii=local_radii.numpy(),
            local_radius_reference=np.asarray(local_radius_reference, dtype=np.float32),
            novelty_threshold=np.asarray(float(selected["novelty_threshold"]), dtype=np.float32),
            mode_teacher_centers=mode_centers.numpy().astype(np.float32),
            mode_teacher_groups=np.asarray(mode_groups, dtype=np.int64),
            mode_teacher_group_reliability=mode_group_reliability.numpy().astype(np.float32),
            mode_teacher_margin_floor=mode_margin_floor.numpy().astype(np.float32),
            mode_teacher_margin_scale=mode_margin_scale.numpy().astype(np.float32),
            mode_teacher_construction_version=np.asarray(mode_diagnostics["version"]),
            atlas_construction_version=np.asarray(atlas_diagnostics["version"]),
            source_view_count=np.asarray(
                int(torch.unique(candidate_views, sorted=True).numel()), dtype=np.int64
            ),
            normal_only=np.asarray(True),
        )
    save_checkpoint(
        model,
        memory,
        args.output_checkpoint,
        {
            "p1_familiarity_memory_rebuild": diagnostics,
            "source_checkpoint": str(args.checkpoint),
        },
        local_radii,
        local_radius_reference,
    )
    print(json.dumps(selected, indent=2), flush=True)
    print(f"[done] {args.output_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
