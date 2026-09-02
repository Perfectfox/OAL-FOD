#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fod_recon_ad.data import (  # noqa: E402
    FODTestDataset,
    default_distances,
    get_tensor_transform,
    normalize_distance,
    normalize_distances,
    resize_preserve_height,
    sliding_windows,
)
from fod_recon_ad.blindspot_context import (  # noqa: E402
    add_blindspot_context_args,
    attach_blindspot_context_head,
    blindspot_context_score,
    fuse_blindspot_scores,
    load_blindspot_calibration,
)
from fod_recon_ad.ext import build_reconstruction_model, load_checkpoint  # noqa: E402
from fod_recon_ad.guided_residual import normalized_residuals, reconstruction_score  # noqa: E402
from fod_recon_ad.inpformer_plus import inpformer_plus_reconstruction_map  # noqa: E402
from fod_recon_ad.masking import (  # noqa: E402
    PerspectiveMaskConfig,
    attach_adaptive_mask_planner,
    attach_local_context_reconstructor,
    masked_reconstruction_components,
    parse_block_sizes,
)
from fod_recon_ad.learned_mask import load_planner_state  # noqa: E402
from fod_recon_ad.metrics import METRIC_KEYS, add_mean_row, evaluate_predictions  # noqa: E402
from fod_recon_ad.prototype_guidance import (  # noqa: E402
    add_guided_prototype_args,
    center6_mode_normalized_features,
    configure_guided_prototypes,
    inpformer_forward_with_prototype_context,
    set_guided_prototype_valid_roi,
)
from fod_recon_ad.scoring import normalize_per_image, reconstruction_components, smooth_np, upsample  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sliding-window encoder-decoder reconstruction eval for FOD.")
    parser.add_argument("--fod-root", type=Path, default=PROJECT_ROOT.parent / "datasets" / "FOD_dataset_ad_50_50_grouped")
    parser.add_argument("--layout", choices=["auto", "original", "mvtec"], default="auto")
    parser.add_argument("--external-repo", type=Path, default=PROJECT_ROOT.parent / "Dinomaly")
    parser.add_argument("--inpformer-repo", type=Path, default=PROJECT_ROOT.parent / "INP-Former")
    parser.add_argument("--architecture", choices=["dinomaly", "inpformer", "mamba"], default="dinomaly")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-split", choices=["val", "test"], default="val")
    parser.add_argument("--eval-distances", nargs="+", default=list(default_distances()))
    parser.add_argument("--encoder", default="dinov2reg_vit_base_14")
    parser.add_argument("--inp-num", type=int, default=6)
    parser.add_argument("--mamba-layers", type=int, default=4)
    parser.add_argument("--mamba-scan", choices=["hilbert", "zorder", "snake", "raster"], default="hilbert")
    parser.add_argument("--mamba-d-state", type=int, default=16)
    parser.add_argument("--mamba-d-conv", type=int, default=4)
    parser.add_argument("--mamba-expand", type=int, default=2)
    parser.add_argument("--mamba-unidirectional", action="store_true")
    parser.add_argument("--mamba-drop", type=float, default=0.0)
    parser.add_argument("--mamba-multi-output", action="store_true")
    parser.add_argument("--full-height", type=int, default=672)
    parser.add_argument("--crop-size", type=int, default=448)
    parser.add_argument(
        "--model-input-size",
        type=int,
        default=0,
        help="Optional network input size after cropping; 0 keeps the physical crop size.",
    )
    parser.add_argument("--stride", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument(
        "--fusion",
        choices=["max", "mean", "center", "top2_mean", "second_max"],
        default="max",
    )
    parser.add_argument("--normalize-crops", action="store_true")
    parser.add_argument("--inp-plus-reconstruction", action="store_true")
    parser.add_argument("--masked-recon", action="store_true")
    parser.add_argument(
        "--mask-strategy",
        choices=["perspective", "uniform", "uniform_single", "adaptive", "candidate", "prototype_grid"],
        default="perspective",
    )
    parser.add_argument("--mask-patterns", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--mask-band-block-sizes", default="1,1,2,2,4,4")
    parser.add_argument("--mask-fill", choices=["visible_mean", "zero"], default="visible_mean")
    parser.add_argument("--mask-prototype-source", choices=["masked", "full"], default="masked")
    parser.add_argument("--inp-local-mask-recon", action="store_true")
    parser.add_argument("--local-context-radius", type=int, default=2)
    parser.add_argument("--mask-planner-checkpoint", type=Path, default=None)
    parser.add_argument("--adaptive-mask-hidden-dim", type=int, default=192)
    parser.add_argument("--adaptive-mask-ratio", type=float, default=0.25)
    parser.add_argument("--adaptive-mask-segments", type=int, default=4)
    parser.add_argument("--adaptive-mask-temperature", type=float, default=1.0)
    parser.add_argument("--prototype-grid-ratio", type=float, default=0.20)
    parser.add_argument("--prototype-grid-threshold", type=float, default=0.0)
    parser.add_argument("--prototype-grid-block-size", type=int, default=2)
    parser.add_argument(
        "--mask-prototype-handling",
        choices=["standard", "masked_frozen", "exclude_detach"],
        default="standard",
    )
    parser.add_argument(
        "--mask-conditional-blend",
        type=float,
        default=0.0,
        help="Blend masked scores into the full score only on selected prototype-grid support.",
    )
    parser.add_argument("--pro-thresholds", type=int, default=200)
    parser.add_argument("--skip-pro", action="store_true")
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument("--subval-ratio", type=float, default=1.0)
    parser.add_argument("--subval-seed", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=8)
    add_guided_prototype_args(parser)
    add_blindspot_context_args(parser)
    return parser.parse_args()


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_config(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_rows(rows: List[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def stratified_subsample(samples, ratio: float, seed: int):
    if ratio >= 1.0:
        return list(samples)
    if ratio <= 0.0:
        raise ValueError("--subval-ratio must be > 0.")
    rng = random.Random(seed)
    by_label = {}
    for sample in samples:
        by_label.setdefault(int(sample.label), []).append(sample)
    selected = []
    for group in by_label.values():
        if not group:
            continue
        keep = max(1, int(round(len(group) * ratio)))
        keep = min(keep, len(group))
        selected.extend(rng.sample(group, keep))
    return sorted(selected, key=lambda item: str(item.image_path))


def crop_batch(
    image: Image.Image,
    windows: List[tuple[int, int]],
    crop_size: int,
    transform,
    output_size: int | None = None,
) -> torch.Tensor:
    crops = []
    for x, y in windows:
        crop = image.crop((x, y, x + crop_size, y + crop_size))
        if output_size is not None and output_size != crop_size:
            crop = crop.resize((output_size, output_size), Image.BILINEAR)
        crops.append(transform(crop))
    return torch.stack(crops, dim=0)


def predict_crop_maps(
    model,
    image: Image.Image,
    windows: List[tuple[int, int]],
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
    mask_config: PerspectiveMaskConfig,
    return_group_maps: bool = False,
    return_guided_mode_maps: bool = False,
    valid_roi_mask: np.ndarray | None = None,
):
    transform = get_tensor_transform()
    maps: List[np.ndarray] = []
    grouped_maps: List[List[np.ndarray]] | None = None
    mode_maps: List[np.ndarray] = []
    mode_novelty_maps: List[np.ndarray] = []
    fp32_fallbacks = 0
    if return_group_maps and return_guided_mode_maps:
        raise ValueError("Group maps and Guided mode maps cannot be returned together.")
    if return_group_maps and (args.masked_recon or args.inp_plus_reconstruction):
        raise ValueError(
            "Per-group reconstruction maps require the standard unmasked residual path."
        )
    if return_group_maps and args.normalize_crops:
        raise ValueError("Per-group reconstruction maps do not support crop normalization.")
    if return_guided_mode_maps and (
        args.masked_recon or args.inp_plus_reconstruction
    ):
        raise ValueError(
            "Guided mode diagnostics require the standard unmasked residual path."
        )

    def forward_scores(input_batch: torch.Tensor):
        if getattr(args, "blindspot_context", False):
            calibration = getattr(args, "_blindspot_calibration", None)
            if calibration is None:
                if args.blindspot_normal_calibration is None:
                    raise ValueError(
                        "--blindspot-context requires --blindspot-normal-calibration."
                    )
                calibration = load_blindspot_calibration(
                    args.blindspot_normal_calibration
                )
                args._blindspot_calibration = calibration
            en, de, context = inpformer_forward_with_prototype_context(
                model, input_batch
            )
            adaptive_score = reconstruction_components(en, de)["base"]
            context_score = blindspot_context_score(
                model.blindspot_context_head,
                context["target_tokens"],
                context["agg_prototype"],
                side=context["side"],
            )
            fused = fuse_blindspot_scores(
                adaptive_score,
                context_score,
                mode=args.blindspot_fusion,
                adaptive_q99=calibration["adaptive_q99"],
                context_q99=calibration["context_q99"],
                boost_alpha=args.blindspot_boost_alpha,
            )
            return fused, None, None, None
        if args.masked_recon:
            components = masked_reconstruction_components(
                model,
                input_batch,
                patterns=args.mask_patterns,
                config=mask_config,
            )
            masked_score = components["base"]
            blend = float(getattr(args, "mask_conditional_blend", 0.0))
            if blend > 0.0:
                full_en, full_de = model(input_batch)[:2]
                full_score = reconstruction_score(model, full_en, full_de)
                support = components["mask_support"].to(dtype=full_score.dtype)
                masked_score = full_score + blend * support * (masked_score - full_score)
            return masked_score, None, None, None
        en, de = model(input_batch)[:2]
        raw_score = (
            inpformer_plus_reconstruction_map(en, de)
            if getattr(args, "inp_plus_reconstruction", False)
            else reconstruction_score(model, en, de)
        )
        mode_assignment = None
        mode_novelty = None
        if return_guided_mode_maps:
            target_tokens = getattr(model, "_guided_decoder_target_tokens", None)
            teacher = getattr(model, "guided_center6_teacher", None)
            if target_tokens is None or teacher is None:
                raise RuntimeError(
                    "Guided mode maps require decoder target tokens and a Center6 teacher."
                )
            token_count = int(target_tokens.shape[1])
            token_side = int(round(math.sqrt(token_count)))
            if token_side * token_side != token_count:
                raise RuntimeError(
                    f"Guided mode tokens are not square: token_count={token_count}."
                )
            mode_novelty, _, _, mode_assignment = center6_mode_normalized_features(
                target_tokens,
                teacher,
                temperature=float(
                    getattr(
                        getattr(model, "_guided_prototype_config", None),
                        "target_mode_novelty_temperature",
                        0.1,
                    )
                ),
                return_assignment=True,
            )
            mode_assignment = mode_assignment.reshape(
                input_batch.shape[0], 1, token_side, token_side
            )
            mode_novelty = mode_novelty.reshape(
                input_batch.shape[0], 1, token_side, token_side
            )
        if not return_group_maps:
            return raw_score, None, mode_assignment, mode_novelty
        if getattr(model, "_guided_residual_scorer", None) is not None or getattr(
            model, "_residual_diffusion_scorer", None
        ) is not None:
            raise ValueError("Per-group diagnostics require the raw-mean residual scorer.")
        residuals = normalized_residuals(en, de)
        groups = [0.5 * residual.square().sum(dim=1, keepdim=True) for residual in residuals]
        return raw_score, groups, mode_assignment, mode_novelty

    with torch.no_grad():
        for start in range(0, len(windows), args.batch_size):
            batch_windows = windows[start : start + args.batch_size]
            model_input_size = args.model_input_size if args.model_input_size > 0 else args.crop_size
            batch = crop_batch(
                image,
                batch_windows,
                args.crop_size,
                transform,
                output_size=model_input_size,
            ).to(device, non_blocking=True)
            roi_batch = None
            if valid_roi_mask is not None:
                roi_image = Image.fromarray(
                    (np.asarray(valid_roi_mask) > 0).astype(np.uint8) * 255,
                    mode="L",
                )
                roi_crops = []
                for x, y in batch_windows:
                    crop = roi_image.crop(
                        (x, y, x + args.crop_size, y + args.crop_size)
                    )
                    if model_input_size != args.crop_size:
                        crop = crop.resize(
                            (model_input_size, model_input_size), Image.NEAREST
                        )
                    roi_crops.append(
                        torch.from_numpy(
                            (np.asarray(crop, dtype=np.uint8) > 0).astype(np.float32)
                        ).unsqueeze(0)
                    )
                roi_batch = torch.stack(roi_crops).to(device, non_blocking=True)
            set_guided_prototype_valid_roi(model, roi_batch)
            with torch.cuda.amp.autocast(enabled=use_amp):
                raw, group_scores, mode_assignment, mode_novelty = forward_scores(batch)
                raw = upsample(raw, args.crop_size)
                if group_scores is not None:
                    group_scores = [upsample(score, args.crop_size) for score in group_scores]
                if mode_assignment is not None:
                    mode_assignment = F.interpolate(
                        mode_assignment.float(),
                        size=(args.crop_size, args.crop_size),
                        mode="nearest",
                    )
                    mode_novelty = F.interpolate(
                        mode_novelty.float(),
                        size=(args.crop_size, args.crop_size),
                        mode="bilinear",
                        align_corners=False,
                    )
            if not torch.isfinite(raw).all():
                if not use_amp:
                    raise RuntimeError("Non-finite reconstruction map produced in FP32 evaluation.")
                fp32_fallbacks += 1
                with torch.cuda.amp.autocast(enabled=False):
                    raw, group_scores, mode_assignment, mode_novelty = forward_scores(
                        batch.float()
                    )
                    raw = upsample(raw, args.crop_size)
                    if group_scores is not None:
                        group_scores = [upsample(score, args.crop_size) for score in group_scores]
                    if mode_assignment is not None:
                        mode_assignment = F.interpolate(
                            mode_assignment.float(),
                            size=(args.crop_size, args.crop_size),
                            mode="nearest",
                        )
                        mode_novelty = F.interpolate(
                            mode_novelty.float(),
                            size=(args.crop_size, args.crop_size),
                            mode="bilinear",
                            align_corners=False,
                        )
                if not torch.isfinite(raw).all():
                    raise RuntimeError("Non-finite reconstruction map remained after FP32 fallback.")
            if args.normalize_crops:
                raw = normalize_per_image(raw)
            raw_np = raw[:, 0].detach().float().cpu().numpy().astype(np.float32)
            raw_np = smooth_np(raw_np, args.sigma)
            maps.extend([item for item in raw_np])
            if mode_assignment is not None:
                mode_np = mode_assignment[:, 0].detach().cpu().numpy().astype(np.int16)
                novelty_np = (
                    mode_novelty[:, 0]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                mode_maps.extend([item for item in mode_np])
                mode_novelty_maps.extend([item for item in novelty_np])
            if group_scores is not None:
                group_np = [
                    smooth_np(
                        score[:, 0].detach().float().cpu().numpy().astype(np.float32),
                        args.sigma,
                    )
                    for score in group_scores
                ]
                if grouped_maps is None:
                    grouped_maps = [[] for _ in group_np]
                if len(grouped_maps) != len(group_np):
                    raise RuntimeError("Reconstruction residual group count changed across batches.")
                for destination, values in zip(grouped_maps, group_np):
                    destination.extend([item for item in values])
    if return_group_maps:
        if grouped_maps is None:
            raise RuntimeError("Per-group reconstruction maps were requested but not produced.")
        return maps, fp32_fallbacks, grouped_maps
    if return_guided_mode_maps:
        if len(mode_maps) != len(maps) or len(mode_novelty_maps) != len(maps):
            raise RuntimeError("Guided mode diagnostics were requested but not produced.")
        return maps, fp32_fallbacks, mode_maps, mode_novelty_maps
    return maps, fp32_fallbacks


def stitch_maps(
    crop_maps: List[np.ndarray],
    windows: List[tuple[int, int]],
    height: int,
    width: int,
    crop_size: int,
    fusion: str,
) -> np.ndarray:
    if len(crop_maps) != len(windows):
        raise ValueError("crop_maps and windows must have the same length.")
    if fusion == "max":
        full = np.zeros((height, width), dtype=np.float32)
        for crop_map, (x, y) in zip(crop_maps, windows):
            full[y : y + crop_size, x : x + crop_size] = np.maximum(
                full[y : y + crop_size, x : x + crop_size],
                crop_map,
            )
        return full
    if fusion in {"top2_mean", "second_max"}:
        top1 = np.full((height, width), -np.inf, dtype=np.float32)
        top2 = np.full((height, width), -np.inf, dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.uint16)
        for crop_map, (x, y) in zip(crop_maps, windows):
            values = np.asarray(crop_map, dtype=np.float32)
            first = top1[y : y + crop_size, x : x + crop_size]
            second = top2[y : y + crop_size, x : x + crop_size]
            greater = values >= first
            updated_second = np.where(greater, first, np.maximum(second, values))
            updated_first = np.where(greater, values, first)
            first[...] = updated_first
            second[...] = updated_second
            counts[y : y + crop_size, x : x + crop_size] += 1
        single = counts < 2
        if fusion == "second_max":
            return np.where(single, top1, top2).astype(np.float32)
        return np.where(single, top1, 0.5 * (top1 + top2)).astype(np.float32)

    full = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    if fusion == "center":
        coordinate = np.linspace(-1.0, 1.0, crop_size, dtype=np.float32)
        one_dimensional = np.clip(1.0 - np.abs(coordinate), 0.05, None)
        crop_weight = np.outer(one_dimensional, one_dimensional).astype(np.float32)
    elif fusion == "mean":
        crop_weight = np.ones((crop_size, crop_size), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported fusion mode: {fusion}")
    for crop_map, (x, y) in zip(crop_maps, windows):
        full[y : y + crop_size, x : x + crop_size] += crop_map * crop_weight
        counts[y : y + crop_size, x : x + crop_size] += crop_weight
    return full / np.maximum(counts, 1e-8)


def stack_padded(arrays: List[np.ndarray], pad_value: float = 0.0) -> np.ndarray:
    height = max(item.shape[0] for item in arrays)
    width = max(item.shape[1] for item in arrays)
    stacked = np.full((len(arrays), height, width), pad_value, dtype=np.float32)
    for idx, item in enumerate(arrays):
        stacked[idx, : item.shape[0], : item.shape[1]] = item.astype(np.float32)
    return stacked


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)
    if not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"
    if args.blindspot_context:
        if args.architecture != "inpformer":
            raise ValueError("--blindspot-context requires INP-Former.")
        if args.masked_recon or args.inp_plus_reconstruction:
            raise ValueError(
                "--blindspot-context is a one-forward alternative to masked/INP+ scoring."
            )
        if args.blindspot_normal_calibration is None:
            raise ValueError(
                "--blindspot-context requires --blindspot-normal-calibration."
            )
        args._blindspot_calibration = load_blindspot_calibration(
            args.blindspot_normal_calibration
        )
    args.eval_distances = normalize_distances(args.eval_distances)
    mask_config = PerspectiveMaskConfig(
        strategy=args.mask_strategy,
        band_block_sizes=parse_block_sizes(args.mask_band_block_sizes),
        fill=args.mask_fill,
        prototype_source=args.mask_prototype_source,
        adaptive_mask_ratio=args.adaptive_mask_ratio,
        adaptive_segments=args.adaptive_mask_segments,
        adaptive_temperature=args.adaptive_mask_temperature,
        inp_mask_mode="local_context" if args.inp_local_mask_recon else "standard",
        local_context_radius=args.local_context_radius,
        prototype_grid_ratio=args.prototype_grid_ratio,
        prototype_grid_threshold=args.prototype_grid_threshold,
        prototype_grid_block_size=args.prototype_grid_block_size,
        prototype_handling=args.mask_prototype_handling,
    )
    if not 0.0 <= args.mask_conditional_blend <= 1.0:
        raise ValueError("--mask-conditional-blend must be in [0, 1].")
    write_config(args)
    print(
        f"[config] device={device} checkpoint={args.checkpoint} full_height={args.full_height} "
        f"crop={args.crop_size} model_input={args.model_input_size or args.crop_size} "
        f"stride={args.stride} fusion={args.fusion}",
        flush=True,
    )

    model, _ = build_reconstruction_model(
        architecture=args.architecture,
        dinomaly_repo=args.external_repo,
        inpformer_repo=args.inpformer_repo,
        encoder=args.encoder,
        device=device,
        inp_num=args.inp_num,
        mamba_layers=args.mamba_layers,
        mamba_scan=args.mamba_scan,
        mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand,
        mamba_bidirectional=not args.mamba_unidirectional,
        mamba_drop=args.mamba_drop,
        mamba_multi_output=args.mamba_multi_output,
    )
    configure_guided_prototypes(model, args, args.architecture)
    if args.blindspot_context:
        attach_blindspot_context_head(
            model,
            hidden_dim=args.blindspot_context_hidden_dim,
            radius=args.blindspot_context_radius,
            prototype_temperature=args.blindspot_context_prototype_temperature,
        )
    if args.inp_local_mask_recon:
        if args.architecture != "inpformer":
            raise ValueError("--inp-local-mask-recon is only defined for --architecture inpformer.")
        if args.mask_strategy != "adaptive":
            raise ValueError("--inp-local-mask-recon expects --mask-strategy adaptive.")
        if args.mask_planner_checkpoint is None:
            raise ValueError("--inp-local-mask-recon requires --mask-planner-checkpoint.")
    if args.masked_recon and args.mask_strategy == "adaptive":
        attach_adaptive_mask_planner(model, hidden_dim=args.adaptive_mask_hidden_dim)
    if args.inp_local_mask_recon:
        attach_local_context_reconstructor(model, radius=args.local_context_radius)
    load_checkpoint(model, args.checkpoint, device, strict=args.strict_load)
    if args.masked_recon and args.mask_strategy == "adaptive" and args.mask_planner_checkpoint is not None:
        model.adaptive_mask_planner.load_state_dict(load_planner_state(args.mask_planner_checkpoint, device), strict=True)
    model.eval()

    rows = []
    for distance in args.eval_distances:
        distance = normalize_distance(distance)
        item = f"fod{distance}_rgb"
        dataset = FODTestDataset(
            root=args.fod_root,
            distance=distance,
            image_size=args.crop_size,
            crop_size=args.crop_size,
            layout=args.layout,
            split=args.eval_split,
        )
        if args.subval_ratio < 1.0:
            dataset.samples = stratified_subsample(
                dataset.samples,
                ratio=args.subval_ratio,
                seed=args.subval_seed + int(distance),
            )
        labels: List[int] = []
        masks: List[np.ndarray] = []
        pred_maps: List[np.ndarray] = []
        fp32_fallbacks = 0
        start = time.time()
        print(f"[eval] {item} samples={len(dataset)}", flush=True)
        for sample_idx, sample in enumerate(dataset.samples, start=1):
            image = Image.open(sample.image_path).convert("RGB")
            resized = resize_preserve_height(image, args.full_height, Image.BILINEAR)
            windows = sliding_windows(resized.size[0], resized.size[1], args.crop_size, args.stride)
            crop_maps, fallbacks = predict_crop_maps(model, resized, windows, args, device, use_amp, mask_config)
            fp32_fallbacks += fallbacks
            stitched = stitch_maps(
                crop_maps,
                windows,
                height=resized.size[1],
                width=resized.size[0],
                crop_size=args.crop_size,
                fusion=args.fusion,
            )
            if sample.mask_path is not None and sample.mask_path.exists():
                mask = Image.open(sample.mask_path).convert("L")
            else:
                mask = Image.new("L", image.size, 0)
            mask = resize_preserve_height(mask, args.full_height, Image.NEAREST)
            mask_np = (np.asarray(mask) > 0).astype(np.uint8)
            if mask_np.shape != stitched.shape:
                mask_t = torch.from_numpy(mask_np[None, None].astype(np.float32))
                mask_np = (
                    F.interpolate(mask_t, size=stitched.shape, mode="nearest")[0, 0].numpy() > 0.5
                ).astype(np.uint8)
            labels.append(int(sample.label))
            masks.append(mask_np)
            pred_maps.append(stitched.astype(np.float32))
            if args.progress_every > 0 and (sample_idx % args.progress_every == 0 or sample_idx == len(dataset.samples)):
                elapsed = time.time() - start
                print(
                    f"[progress] {item} {sample_idx}/{len(dataset)} "
                    f"windows={len(windows)} elapsed={elapsed:.1f}s",
                    flush=True,
                )
        row = evaluate_predictions(
            labels=np.asarray(labels, dtype=np.uint8),
            masks=stack_padded(masks, pad_value=0.0).astype(np.uint8),
            maps=stack_padded(pred_maps, pad_value=0.0),
            item=item,
            distance=distance,
            seconds=time.time() - start,
            pro_thresholds=args.pro_thresholds,
            skip_pro=args.skip_pro,
        )
        rows.append(row)
        print(
            "[result] {item_name}: P-AUROC={P-AUROC:.4f} P-AP={P-AP:.4f} "
            "P-F1={P-F1:.4f} AUC-PRO={AUC-PRO:.4f}".format(item_name=item, **row),
            flush=True,
        )
        if fp32_fallbacks:
            print(f"[warn] {item} AMP produced non-finite maps; recomputed {fp32_fallbacks} crop batches in FP32.", flush=True)

    rows = add_mean_row(rows)
    write_rows(rows, args.output_dir / "cross_distance_rgb_metrics.csv")
    mean_row = rows[-1]
    with (args.output_dir / "mean_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", *METRIC_KEYS])
        writer.writeheader()
        writer.writerow({"variant": "sliding_raw", **{key: mean_row[key] for key in METRIC_KEYS}})
    print(
        "[mean] sliding_raw: P-AUROC={P-AUROC:.4f} P-AP={P-AP:.4f} "
        "P-F1={P-F1:.4f} AUC-PRO={AUC-PRO:.4f}".format(**mean_row),
        flush=True,
    )
    print(f"[done] saved {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
