#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fod_recon_ad.data import (  # noqa: E402
    FODTestDataset,
    ImageMaskResolver,
    get_tensor_transform,
    normalize_distance,
    sliding_windows,
)
from fod_recon_ad.ext import build_reconstruction_model, load_checkpoint  # noqa: E402
from fod_recon_ad.prototype_guidance import (  # noqa: E402
    NormalObjectTokenMemory,
    add_guided_prototype_args,
    configure_guided_prototypes,
    inpformer_forward_with_prototype_context,
)
from fod_recon_ad.prototype_visualization import (  # noqa: E402
    familiarity_gate_maps,
    first_block_attention_mass,
    tokenize_binary_mask,
)


@dataclass(frozen=True)
class SelectedCase:
    case_id: str
    distance: str
    image_path: Path
    mask_path: Path
    roi_path: Path | None
    window_xy: tuple[int, int]
    gt_pixels: int
    gt_total_pixels: int
    gt_coverage: float
    gt_components: int
    gt_border_clearance: int
    valid_fraction: float


REGION_ORDER = (
    "gt_anomaly",
    "normal_background",
    "normal_bg_objectlike",
    "normal_bg_objectlike_familiar",
    "normal_bg_objectlike_novel",
)
REGION_TITLES = {
    "gt_anomaly": "GT anomaly",
    "normal_background": "normal background",
    "normal_bg_objectlike": "normal BG: objectness>=0.5",
    "normal_bg_objectlike_familiar": "normal BG: objectlike + familiar",
    "normal_bg_objectlike_novel": "normal BG: objectlike + novel",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize and quantify token suppression before/after Familiarity Gate."
    )
    parser.add_argument("--fod-root", type=Path, required=True)
    parser.add_argument("--layout", choices=("auto", "original", "mvtec"), default="original")
    parser.add_argument("--roi-mask", type=Path, default=None)
    parser.add_argument("--distances", nargs="+", default=["05", "10", "15", "20", "25", "30"])
    parser.add_argument("--eval-split", choices=("val", "test"), default="test")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--external-repo", type=Path, default=PROJECT_ROOT.parent / "Dinomaly")
    parser.add_argument("--inpformer-repo", type=Path, default=PROJECT_ROOT.parent / "INP-Former")
    parser.add_argument("--encoder", default="dinov2reg_vit_base_14")
    parser.add_argument("--inp-num", type=int, default=6)
    parser.add_argument("--crop-size", type=int, default=448)
    parser.add_argument("--model-input-size", type=int, default=672)
    parser.add_argument("--stride", type=int, default=224)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument("--roi-token-threshold", type=float, default=0.50)
    parser.add_argument("--objectness-threshold", type=float, default=0.50)
    parser.add_argument("--novelty-threshold", type=float, default=0.50)
    return parser.parse_args()


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def guided_args_from_config(path: Path, model_input_size: int) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    add_guided_prototype_args(parser)
    result = parser.parse_args([])
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key, default in vars(result).items():
        value = payload.get(key, default)
        if value is not None:
            setattr(result, key, value)
    result.guided_prototype = True
    result.guided_prototype_familiarity_gate = True
    result.model_input_size = int(model_input_size)
    result.patch_output_size = int(model_input_size)
    result.image_size = int(model_input_size)
    result.crop_size = int(model_input_size)
    result.masked_recon = False
    return result


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        mask = image.convert("L")
        if mask.size != size:
            mask = mask.resize(size, Image.NEAREST)
        return np.asarray(mask, dtype=np.uint8) > 0


def select_cases(args: argparse.Namespace) -> list[SelectedCase]:
    """Select one high-coverage anomaly crop per Test image."""

    resolver = ImageMaskResolver(args.roi_mask)
    selected: list[SelectedCase] = []
    for raw_distance in args.distances:
        distance = normalize_distance(raw_distance)
        dataset = FODTestDataset(
            args.fod_root,
            distance,
            args.crop_size,
            args.crop_size,
            layout=args.layout,
            split=args.eval_split,
        )
        for sample in dataset.samples:
            if sample.label != 1 or sample.mask_path is None:
                continue
            with Image.open(sample.image_path) as image_handle:
                image_size = image_handle.size
            gt = load_mask(sample.mask_path, image_size)
            gt_total = int(gt.sum())
            if gt_total <= 0:
                continue
            roi_path = resolver.resolve(sample.image_path)
            roi = np.ones(gt.shape, dtype=bool) if roi_path is None else load_mask(roi_path, image_size)
            candidates: list[tuple[tuple[float, int, int, float, int, int], SelectedCase]] = []
            for x, y in sliding_windows(image_size[0], image_size[1], args.crop_size, args.stride):
                gt_crop = gt[y : y + args.crop_size, x : x + args.crop_size]
                gt_pixels = int(gt_crop.sum())
                if gt_pixels <= 0:
                    continue
                gt_y, gt_x = np.where(gt_crop)
                clearance = int(
                    min(
                        gt_x.min(),
                        gt_y.min(),
                        args.crop_size - 1 - gt_x.max(),
                        args.crop_size - 1 - gt_y.max(),
                    )
                )
                components = int(
                    cv2.connectedComponents(gt_crop.astype(np.uint8), connectivity=8)[0] - 1
                )
                valid_fraction = float(
                    roi[y : y + args.crop_size, x : x + args.crop_size].mean()
                )
                coverage = float(gt_pixels / gt_total)
                case = SelectedCase(
                    case_id=f"d{distance}_{sample.image_path.stem}_x{x}_y{y}",
                    distance=distance,
                    image_path=sample.image_path,
                    mask_path=sample.mask_path,
                    roi_path=roi_path,
                    window_xy=(x, y),
                    gt_pixels=gt_pixels,
                    gt_total_pixels=gt_total,
                    gt_coverage=coverage,
                    gt_components=components,
                    gt_border_clearance=clearance,
                    valid_fraction=valid_fraction,
                )
                rank = (coverage, min(clearance, 64), gt_pixels, valid_fraction, -y, -x)
                candidates.append((rank, case))
            if not candidates:
                raise RuntimeError(f"No GT crop found for {sample.image_path}.")
            selected.append(max(candidates, key=lambda item: item[0])[1])
    if not selected:
        raise RuntimeError("No anomaly cases were selected.")
    return selected


def choose_visual_cases(cases: list[SelectedCase]) -> list[SelectedCase]:
    chosen: list[SelectedCase] = []
    for distance in sorted({case.distance for case in cases}):
        candidates = [case for case in cases if case.distance == distance]
        chosen.append(
            max(
                candidates,
                key=lambda case: (
                    min(case.gt_border_clearance, 64),
                    case.gt_components,
                    -case.gt_pixels,
                    case.valid_fraction,
                    case.case_id,
                ),
            )
        )
    return chosen


def crop_case(case: SelectedCase, crop_size: int) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    x, y = case.window_xy
    with Image.open(case.image_path) as handle:
        full_size = handle.size
        image = handle.convert("RGB").crop((x, y, x + crop_size, y + crop_size))
    gt = load_mask(case.mask_path, full_size)[y : y + crop_size, x : x + crop_size]
    valid = (
        np.ones(gt.shape, dtype=bool)
        if case.roi_path is None
        else load_mask(case.roi_path, full_size)[y : y + crop_size, x : x + crop_size]
    )
    return image, gt, valid


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = float(ranks[labels].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def binary_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, ranked.size + 1)
    return float(precision[ranked].sum() / positives)


def region_masks(
    valid: np.ndarray,
    gt: np.ndarray,
    objectness: np.ndarray,
    novelty: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    valid_flat = valid.reshape(-1).astype(bool)
    gt_flat = gt.reshape(-1).astype(bool) & valid_flat
    background = valid_flat & ~gt_flat
    objectlike = background & (objectness >= args.objectness_threshold)
    return {
        "gt_anomaly": gt_flat,
        "normal_background": background,
        "normal_bg_objectlike": objectlike,
        "normal_bg_objectlike_familiar": objectlike & (novelty < args.novelty_threshold),
        "normal_bg_objectlike_novel": objectlike & (novelty >= args.novelty_threshold),
    }


def summarize_region(
    region: np.ndarray,
    arrays: dict[str, np.ndarray],
    valid: np.ndarray,
) -> dict[str, float]:
    count = int(region.sum())
    result: dict[str, float] = {"tokens": float(count)}
    fields = (
        "objectness",
        "novelty",
        "risk",
        "gate_weight",
        "suppression",
        "attention_before_scaled",
        "attention_after_scaled",
        "attention_delta_scaled",
    )
    for field in fields:
        values = arrays[field][region]
        result[f"{field}_mean"] = float(values.mean()) if count else float("nan")
        result[f"{field}_median"] = float(np.median(values)) if count else float("nan")
        result[f"{field}_p90"] = float(np.quantile(values, 0.90)) if count else float("nan")
    weights = arrays["gate_weight"][region]
    result["gate_lt_0.50_rate"] = float((weights < 0.50).mean()) if count else float("nan")
    result["gate_lt_0.75_rate"] = float((weights < 0.75).mean()) if count else float("nan")
    before_total = float(arrays["attention_before"][valid].sum())
    after_total = float(arrays["attention_after"][valid].sum())
    result["attention_share_before"] = (
        float(arrays["attention_before"][region].sum()) / before_total if before_total > 0 else float("nan")
    )
    result["attention_share_after"] = (
        float(arrays["attention_after"][region].sum()) / after_total if after_total > 0 else float("nan")
    )
    region_before = float(arrays["attention_before"][region].sum())
    region_after = float(arrays["attention_after"][region].sum())
    result["attention_mass_relative_change"] = (
        1.0 - region_after / region_before if region_before > 0 else float("nan")
    )
    return result


def infer_cases(
    args: argparse.Namespace,
    cases: list[SelectedCase],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _ = build_reconstruction_model(
        architecture="inpformer",
        dinomaly_repo=args.external_repo,
        inpformer_repo=args.inpformer_repo,
        encoder=args.encoder,
        device=device,
        inp_num=args.inp_num,
    )
    guided_args = guided_args_from_config(args.model_config, args.model_input_size)
    configure_guided_prototypes(model, guided_args, "inpformer")
    load_checkpoint(model, args.checkpoint, device, strict=args.strict_load)
    model.eval()
    config = model._guided_prototype_config
    memory = getattr(model, "guided_normal_object_memory", None)
    if not isinstance(memory, NormalObjectTokenMemory) or not memory.ready:
        raise RuntimeError("The checkpoint does not contain a ready normal familiarity memory.")
    if not model.aggregation:
        raise RuntimeError("The model has no prototype aggregation blocks.")

    transform = get_tensor_transform()
    rows: list[dict[str, Any]] = []
    case_arrays: dict[str, dict[str, np.ndarray]] = {}
    with torch.no_grad():
        for index, case in enumerate(cases, start=1):
            image, gt, valid = crop_case(case, args.crop_size)
            resized = image.resize((args.model_input_size, args.model_input_size), Image.BILINEAR)
            batch = transform(resized).unsqueeze(0).to(device)
            _, _, context = inpformer_forward_with_prototype_context(model, batch)
            tokens = context["target_tokens"]
            side = int(context["side"])
            state = getattr(model, "_guided_last_prior_state", None)
            if not isinstance(state, dict) or state.get("prior") is None:
                raise RuntimeError("Guided prior state was not exposed by the forward pass.")
            objectness_tensor = state["prior"][:, :, 2].detach()
            novelty_tensor, distance_tensor = memory.novelty(tokens)
            gate = familiarity_gate_maps(
                objectness_tensor,
                novelty_tensor,
                min_weight=config.aggregation_min_weight,
                power=config.aggregation_power,
                alpha=config.native_anchor_alpha,
            )
            attention_before, attention_after = first_block_attention_mass(
                model.aggregation[0],
                model.prototype_token,
                tokens,
                gate.token_weight,
            )

            gt_tokens = tokenize_binary_mask(gt, args.model_input_size, side, mode="any")
            valid_tokens = tokenize_binary_mask(
                valid,
                args.model_input_size,
                side,
                mode="fraction",
                threshold=args.roi_token_threshold,
            )
            flat: dict[str, np.ndarray] = {
                "objectness": objectness_tensor[0].float().cpu().numpy(),
                "novelty": novelty_tensor[0].float().cpu().numpy(),
                "novelty_distance": distance_tensor[0].float().cpu().numpy(),
                "risk": gate.risk[0].float().cpu().numpy(),
                "gate_weight": gate.token_weight[0].float().cpu().numpy(),
                "suppression": gate.suppression[0].float().cpu().numpy(),
                "attention_before": attention_before[0].float().cpu().numpy(),
                "attention_after": attention_after[0].float().cpu().numpy(),
            }
            token_count = int(side * side)
            flat["attention_before_scaled"] = flat["attention_before"] * token_count
            flat["attention_after_scaled"] = flat["attention_after"] * token_count
            flat["attention_delta_scaled"] = (
                flat["attention_before"] - flat["attention_after"]
            ) * token_count
            valid_flat = valid_tokens.reshape(-1)
            gt_flat = gt_tokens.reshape(-1) & valid_flat
            regions = region_masks(
                valid_flat,
                gt_flat,
                flat["objectness"],
                flat["novelty"],
                args,
            )
            for region_name in REGION_ORDER:
                row: dict[str, Any] = {
                    "case_id": case.case_id,
                    "distance": case.distance,
                    "region": region_name,
                    **summarize_region(regions[region_name], flat, valid_flat),
                }
                rows.append(row)

            valid_scores = flat["suppression"][valid_flat]
            valid_labels = gt_flat[valid_flat]
            gt_count = int(valid_labels.sum())
            top_k = max(1, gt_count)
            top_indices = np.argsort(-valid_scores, kind="mergesort")[:top_k]
            case_metrics = {
                "gate_suppression_auroc": binary_auroc(valid_labels, valid_scores),
                "gate_suppression_ap": binary_average_precision(valid_labels, valid_scores),
                "gt_token_base_rate": float(valid_labels.mean()),
                "top_gt_count_precision": float(valid_labels[top_indices].mean()),
                "gt_suppression_budget_share": float(flat["suppression"][gt_flat].sum())
                / max(float(flat["suppression"][valid_flat].sum()), 1e-12),
            }
            for row in rows[-len(REGION_ORDER) :]:
                row.update(case_metrics)

            stored = {key: value.reshape(side, side) for key, value in flat.items()}
            stored["gt_tokens"] = gt_tokens
            stored["valid_tokens"] = valid_tokens
            stored["image_rgb"] = np.asarray(image.convert("RGB"), dtype=np.uint8)
            case_arrays[case.case_id] = stored
            array_dir = args.output_dir / "arrays" / case.case_id
            array_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                array_dir / "familiarity_gate.npz",
                **{key: value for key, value in stored.items() if key != "image_rgb"},
            )
            print(f"[gate] {index}/{len(cases)} {case.case_id}", flush=True)

    metadata = {
        "memory_count": int(memory.count.item()),
        "memory_novelty_threshold": float(memory.novelty_threshold.item()),
        "aggregation_min_weight": float(config.aggregation_min_weight),
        "aggregation_power": float(config.aggregation_power),
        "native_anchor_alpha": float(config.native_anchor_alpha),
        "token_side": int(next(iter(case_arrays.values()))["gt_tokens"].shape[0]),
    }
    return rows, case_arrays, metadata


def outline(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    canvas = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(canvas, contours, -1, 1, thickness=1)
    return canvas.astype(bool)


def plot_case(
    args: argparse.Namespace,
    case: SelectedCase,
    arrays: dict[str, np.ndarray],
) -> Path:
    image = arrays["image_rgb"]
    gt = arrays["gt_tokens"]
    valid = arrays["valid_tokens"]
    edge = outline(gt)
    maps = (
        ("objectness", "Objectness prior", "magma", 0.0, 1.0),
        ("novelty", "Novelty", "magma", 0.0, 1.0),
        ("risk", "Risk = objectness x novelty", "magma", 0.0, 1.0),
        ("suppression", "Gate suppression (1-weight)", "inferno", 0.0, 1.0),
        ("attention_before_scaled", "Attention before (x uniform)", "viridis", 0.0, None),
        ("attention_after_scaled", "Attention after (x uniform)", "viridis", 0.0, None),
        ("attention_delta_scaled", "Attention decrease (x uniform)", "coolwarm", None, None),
    )
    fig, axes = plt.subplots(2, 4, figsize=(15.2, 7.6), constrained_layout=True)
    axes = axes.reshape(-1)
    attention_vmax = max(
        float(np.quantile(arrays["attention_before_scaled"][valid], 0.99)),
        float(np.quantile(arrays["attention_after_scaled"][valid], 0.99)),
        1.0,
    )
    axes[0].imshow(image)
    gt_image = cv2.resize(gt.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    axes[0].contour(gt_image, levels=[0.5], colors=["cyan"], linewidths=1.4)
    axes[0].set_title(f"RGB + GT | {case.distance} m")
    axes[0].axis("off")
    for axis, (key, title, cmap, vmin, vmax) in zip(axes[1:], maps):
        values = arrays[key].copy()
        if key == "attention_delta_scaled":
            bound = max(float(np.quantile(np.abs(values[valid]), 0.99)), 0.1)
            vmin, vmax = -bound, bound
        elif key in {"attention_before_scaled", "attention_after_scaled"}:
            vmax = attention_vmax
        elif vmax is None:
            vmax = max(float(np.quantile(values[valid], 0.99)), 1.0)
        masked = np.ma.masked_where(~valid, values)
        shown = axis.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        axis.contour(edge, levels=[0.5], colors=["cyan"], linewidths=0.8)
        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(shown, ax=axis, fraction=0.046, pad=0.02)
    fig.suptitle(case.case_id, fontsize=11)
    output = args.output_dir / "figures" / f"{case.case_id}_gate_before_after.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def save_contact_sheet(paths: list[Path], output: Path) -> None:
    images: list[Image.Image] = []
    try:
        for path in paths:
            with Image.open(path) as handle:
                images.append(handle.convert("RGB").copy())
        target_width = min(2400, max(image.width for image in images))
        resized = [
            image.resize((target_width, round(image.height * target_width / image.width)), Image.BILINEAR)
            for image in images
        ]
        sheet = Image.new("RGB", (target_width, sum(image.height for image in resized)), "white")
        y = 0
        for image in resized:
            sheet.paste(image, (0, y))
            y += image.height
        output.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output)
        for image in resized:
            image.close()
    finally:
        for image in images:
            image.close()


def pool_region_arrays(
    cases: list[SelectedCase],
    all_arrays: dict[str, dict[str, np.ndarray]],
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    pooled: dict[str, dict[str, list[np.ndarray]]] = {
        region: {
            field: []
            for field in (
                "objectness",
                "novelty",
                "suppression",
                "attention_before_scaled",
                "attention_after_scaled",
                "attention_delta_scaled",
            )
        }
        for region in REGION_ORDER
    }
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    for case in cases:
        arrays = all_arrays[case.case_id]
        valid = arrays["valid_tokens"].reshape(-1)
        gt = arrays["gt_tokens"].reshape(-1) & valid
        flat = {key: arrays[key].reshape(-1) for key in arrays if key != "image_rgb"}
        regions = region_masks(valid, gt, flat["objectness"], flat["novelty"], args)
        for region, selection in regions.items():
            for field in pooled[region]:
                pooled[region][field].append(flat[field][selection])
        all_labels.append(gt[valid])
        all_scores.append(flat["suppression"][valid])
    result = {
        region: {
            field: np.concatenate(chunks) if chunks else np.asarray([], dtype=np.float32)
            for field, chunks in fields.items()
        }
        for region, fields in pooled.items()
    }
    return result, np.concatenate(all_labels), np.concatenate(all_scores)


def plot_global_distributions(
    args: argparse.Namespace,
    pooled: dict[str, dict[str, np.ndarray]],
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    display_regions = ("gt_anomaly", "normal_background", "normal_bg_objectlike")
    colors = ("#d62728", "#1f77b4", "#ff9f1c")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)
    for region, color in zip(display_regions, colors):
        values = np.sort(pooled[region]["suppression"])
        if values.size:
            axes[0].plot(values, np.arange(1, values.size + 1) / values.size, label=REGION_TITLES[region], color=color)
    axes[0].set(xlabel="gate suppression (1-weight)", ylabel="empirical CDF", xlim=(0, 1), ylim=(0, 1))
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    x = np.arange(len(display_regions))
    means = [float(pooled[region]["suppression"].mean()) for region in display_regions]
    axes[1].bar(x, means, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        ["GT", "all normal BG", "objectlike normal BG"], rotation=15, ha="right"
    )
    axes[1].set_ylabel("mean gate suppression")
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis="y", alpha=0.25)

    attention_means = []
    for region in display_regions:
        before = float(pooled[region]["attention_before_scaled"].mean())
        after = float(pooled[region]["attention_after_scaled"].mean())
        attention_means.append(1.0 - after / before if before > 0 else float("nan"))
    axes[2].bar(x, attention_means, color=colors)
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(
        ["GT", "all normal BG", "objectlike normal BG"], rotation=15, ha="right"
    )
    axes[2].set_ylabel("first-block regional attention mass drop")
    axes[2].grid(axis="y", alpha=0.25)
    fig.savefig(args.output_dir / "figures" / "gate_region_distributions.png", dpi=180)
    plt.close(fig)

    rng = np.random.default_rng(args.seed)
    bg = pooled["normal_background"]
    sample_count = min(20000, bg["objectness"].size)
    indices = rng.choice(bg["objectness"].size, sample_count, replace=False) if sample_count else np.asarray([], dtype=int)
    fig, axis = plt.subplots(figsize=(6.3, 5.3), constrained_layout=True)
    if indices.size:
        axis.scatter(bg["objectness"][indices], bg["novelty"][indices], s=4, alpha=0.12, color="#1f77b4", label="normal background")
    gt = pooled["gt_anomaly"]
    axis.scatter(gt["objectness"], gt["novelty"], s=13, alpha=0.65, color="#d62728", label="GT anomaly")
    axis.axvline(args.objectness_threshold, color="gray", linestyle="--", linewidth=0.8)
    axis.axhline(args.novelty_threshold, color="gray", linestyle="--", linewidth=0.8)
    axis.set(xlabel="objectness prior", ylabel="novelty", xlim=(0, 1), ylim=(0, 1))
    axis.grid(alpha=0.2)
    axis.legend()
    fig.savefig(args.output_dir / "figures" / "objectness_novelty_scatter.png", dpi=180)
    plt.close(fig)

    gt_count = int(labels.sum())
    order = np.argsort(-scores, kind="mergesort")
    top = labels[order[: max(1, gt_count)]]
    suppression_sum = float(scores.sum())
    gt_budget = float(scores[labels].sum()) / max(suppression_sum, 1e-12)
    gt_base = float(labels.mean())
    gt_attention_before = pooled["gt_anomaly"]["attention_before_scaled"]
    gt_attention_after = pooled["gt_anomaly"]["attention_after_scaled"]
    gt_attention_drop = 1.0 - float(gt_attention_after.sum()) / max(
        float(gt_attention_before.sum()), 1e-12
    )
    gt_near_zero_attention = float((gt_attention_before <= 1e-8).mean())
    gt_mean_suppression = float(pooled["gt_anomaly"]["suppression"].mean())

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)
    all_regions = (
        "gt_anomaly",
        "normal_background",
        "normal_bg_objectlike_familiar",
        "normal_bg_objectlike_novel",
    )
    region_labels = ("GT", "all normal BG", "familiar objectlike BG", "novel objectlike BG")
    region_colors = ("#d62728", "#1f77b4", "#2ca02c", "#ff9f1c")
    region_means = [float(pooled[name]["suppression"].mean()) for name in all_regions]
    axes[0].bar(np.arange(4), np.asarray(region_means) * 100.0, color=region_colors)
    axes[0].set_xticks(np.arange(4))
    axes[0].set_xticklabels(region_labels, rotation=18, ha="right")
    axes[0].set_ylabel("mean gate suppression (%)")
    axes[0].grid(axis="y", alpha=0.25)

    targeting_values = np.asarray([gt_base, gt_budget, float(top.mean())]) * 100.0
    axes[1].bar(
        np.arange(3),
        targeting_values,
        color=("#8c8c8c", "#9467bd", "#d62728"),
    )
    axes[1].set_xticks(np.arange(3))
    axes[1].set_xticklabels(
        ("GT token\nbase share", "GT share of\nsuppression budget", "GT precision in\ntop-Ngt suppressed"),
        rotation=10,
        ha="right",
    )
    axes[1].set_ylabel("share (%)")
    axes[1].grid(axis="y", alpha=0.25)

    mechanism_values = np.asarray(
        [gt_near_zero_attention, gt_mean_suppression, gt_attention_drop]
    ) * 100.0
    axes[2].bar(
        np.arange(3), mechanism_values, color=("#7f7f7f", "#d62728", "#17becf")
    )
    axes[2].set_xticks(np.arange(3))
    axes[2].set_xticklabels(
        ("GT with near-zero\npre-gate attention", "GT mean raw\ngate suppression", "GT total attention\nmass drop"),
        rotation=10,
        ha="right",
    )
    axes[2].set_ylabel("percentage (%)")
    axes[2].grid(axis="y", alpha=0.25)
    fig.savefig(args.output_dir / "figures" / "gate_targeting_summary.png", dpi=180)
    plt.close(fig)

    return {
        "pooled_gate_suppression_auroc": binary_auroc(labels, scores),
        "pooled_gate_suppression_ap": binary_average_precision(labels, scores),
        "gt_token_base_rate": gt_base,
        "top_gt_count_precision": float(top.mean()),
        "gt_suppression_budget_share": gt_budget,
        "suppression_budget_enrichment_vs_base": gt_budget / max(gt_base, 1e-12),
        "gt_pre_gate_attention_near_zero_rate": gt_near_zero_attention,
        "gt_total_attention_mass_relative_change": gt_attention_drop,
    }


def write_outputs(
    args: argparse.Namespace,
    cases: list[SelectedCase],
    visual_cases: list[SelectedCase],
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    global_metrics: dict[str, float],
) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with (args.output_dir / "gate_region_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "gate_region_summary.json").write_text(
        json.dumps(json_ready(rows), indent=2, allow_nan=True), encoding="utf-8"
    )
    payload = {
        "config": vars(args),
        "gate_metadata": metadata,
        "global_metrics": global_metrics,
        "case_count": len(cases),
        "visual_case_ids": [case.case_id for case in visual_cases],
        "selected_cases": [case.__dict__ for case in cases],
    }
    (args.output_dir / "gate_experiment_summary.json").write_text(
        json.dumps(json_ready(payload), indent=2, allow_nan=True), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = select_cases(args)
    visual_cases = choose_visual_cases(cases)
    rows, all_arrays, metadata = infer_cases(args, cases)
    figure_paths = [plot_case(args, case, all_arrays[case.case_id]) for case in visual_cases]
    save_contact_sheet(
        figure_paths,
        args.output_dir / "figures" / "gate_before_after_contact_sheet.png",
    )
    pooled, labels, scores = pool_region_arrays(cases, all_arrays, args)
    global_metrics = plot_global_distributions(args, pooled, labels, scores)
    for region, fields in pooled.items():
        global_metrics[f"{region}_tokens"] = int(fields["suppression"].size)
        for field, values in fields.items():
            global_metrics[f"{region}_{field}_mean"] = (
                float(values.mean()) if values.size else float("nan")
            )
        before = float(fields["attention_before_scaled"].mean())
        after = float(fields["attention_after_scaled"].mean())
        global_metrics[f"{region}_attention_mass_relative_change"] = (
            1.0 - after / before if before > 0 else float("nan")
        )
    write_outputs(args, cases, visual_cases, rows, metadata, global_metrics)
    print(json.dumps(json_ready(global_metrics), indent=2), flush=True)
    print(f"[done] {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
