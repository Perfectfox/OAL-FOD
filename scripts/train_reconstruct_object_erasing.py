#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import sys
import time
import zipfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fod_recon_ad.data import FODCropTrainDataset, normalize_distance  # noqa: E402
from fod_recon_ad.ext import build_reconstruction_model, checkpoint_payload, load_checkpoint  # noqa: E402
from fod_recon_ad.gradient_conflict import (  # noqa: E402
    GradientConflictComponentScaler,
    GradientConflictProbe,
    coarse_parameter_group,
    gradient_triplet_diagnostics,
)
from fod_recon_ad.guided_residual import reconstruction_score  # noqa: E402
from fod_recon_ad.hard_normal_training import (  # noqa: E402
    HardNormalReplay,
    causal_restore_loss,
)
from fod_recon_ad.masking import PerspectiveMaskConfig, forward_masked_reconstruction, parse_block_sizes  # noqa: E402
from fod_recon_ad.native_losses import (  # noqa: E402
    compute_masked_normal_loss,
    compute_normal_loss,
    compute_roi_masked_normal_loss,
)
from fod_recon_ad.inpformer_plus import configure_inp_coherence  # noqa: E402
from fod_recon_ad.normal_calibration import SourceGroupResolver  # noqa: E402
from fod_recon_ad.context_familiarity_transport import (  # noqa: E402
    context_familiarity_transport_memory_ready,
    fit_context_familiarity_transport_memory,
)
from fod_recon_ad.normal_descriptor_gate import (  # noqa: E402
    fit_normal_descriptor_memory,
    normal_descriptor_memory_ready,
)
from fod_recon_ad.prototype_guidance import (  # noqa: E402
    add_guided_prototype_args,
    configure_guided_prototypes,
    get_guided_prototype_diag,
    guided_prototype_extra_loss,
    guided_prototype_trainable_modules,
    guided_target_gate_normal_anchor_loss,
    guided_target_gate_supervision_loss,
    set_guided_prototype_image,
    set_guided_prototype_source_groups,
    set_guided_prototype_valid_roi,
)
from fod_recon_ad.target_prototype_learning import (  # noqa: E402
    DecoderAttentionCapture,
    aggregation_attention_exclusion_loss,
    decoder_read_attention_exclusion_loss,
    prototype_invariance_loss,
    prototype_repulsion_loss,
)
from fod_inline_validation import InlineFODValidator, InlineValidationSchedule  # noqa: E402


def repeat_loader(loader: Iterable) -> Iterator:
    """Repeat a DataLoader without caching its first traversal.

    ``itertools.cycle(loader)`` retains every batch from the first traversal and
    then replays those tensors forever.  Re-entering the DataLoader preserves
    epoch-wise reshuffling and avoids a potentially very large CPU cache.
    """

    while True:
        yield from loader


def parse_scale_bins(spec: str) -> List[Tuple[float, float, float]]:
    bins: List[Tuple[float, float, float]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        low, high, weight = (float(value) for value in part.split(":"))
        if high <= low or weight <= 0.0:
            continue
        bins.append((low, high, weight))
    if not bins:
        raise ValueError(f"No valid scale bins parsed from {spec!r}")
    return bins


def parse_float_list(text: str) -> tuple[float, ...]:
    values = [float(part) for part in text.replace(",", " ").split() if part.strip()]
    if any(value <= 0 for value in values):
        raise ValueError(f"All weights must be positive: {values}")
    return tuple(values)


def token_area_for_resized_box(width: float, height: float, image_size: int, patch_size: int = 14) -> int:
    tx = max(1, int(math.ceil(max(width, 1e-6) / float(patch_size))))
    ty = max(1, int(math.ceil(max(height, 1e-6) / float(patch_size))))
    return tx * ty


def _boxes_overlap(first: Tuple[int, int, int, int], second: Tuple[int, int, int, int]) -> bool:
    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )


def _soft_box_alpha(
    height: int,
    width: int,
    feather: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    shape: str = "box",
) -> torch.Tensor:
    """Return a box alpha with a narrow feathered boundary.

    The source manifests only provide boxes, not instance masks.  Keeping the
    feather narrow preserves the real small-object pixels while removing the
    strongest rectangular cut boundary.  A zero feather is an exact CutPaste.
    """

    if shape not in {"box", "ellipse"}:
        raise ValueError(f"Unsupported paste alpha shape: {shape}")
    if shape == "ellipse" and min(height, width) > 4:
        y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / float(height)
        x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / float(width)
        radius = torch.sqrt(((y[:, None] - 0.5) / 0.5).square() + ((x[None, :] - 0.5) / 0.5).square())
        softness = max(float(feather) / float(max(min(height, width), 1)), 0.08)
        return ((1.0 - radius) / softness).clamp(0.0, 1.0).unsqueeze(0)
    if feather <= 0 or min(height, width) <= 2:
        return torch.ones((1, height, width), device=device, dtype=dtype)
    effective = min(int(feather), max((min(height, width) - 1) // 2, 0))
    if effective <= 0:
        return torch.ones((1, height, width), device=device, dtype=dtype)
    y = torch.minimum(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(height - 1, -1, -1, device=device, dtype=dtype),
    )
    x = torch.minimum(
        torch.arange(width, device=device, dtype=dtype),
        torch.arange(width - 1, -1, -1, device=device, dtype=dtype),
    )
    # The outermost pixel remains partially visible instead of becoming a zero
    # ring, which matters for 2--5 pixel distant objects.
    y_alpha = ((y + 1.0) / float(effective + 1)).clamp(max=1.0)
    x_alpha = ((x + 1.0) / float(effective + 1)).clamp(max=1.0)
    return torch.minimum(y_alpha[:, None], x_alpha[None, :]).unsqueeze(0)


def compose_target_background_batch(
    source_images: torch.Tensor,
    source_boxes: torch.Tensor,
    source_valid: torch.Tensor,
    target_images: torch.Tensor,
    *,
    seed: int,
    max_pastes: int = 24,
    feather: int = 1,
    alpha_shape: str = "box",
    placement_attempts: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Paste real auxiliary objects onto target-domain Normal backgrounds.

    This is deliberately an in-memory training transform: target images remain
    the original Normal samples, no synthetic cache is written, and the output
    object mask contains only pasted source boxes.  Source object size is kept
    unchanged in model pixels for a controlled comparison with the full-image
    auxiliary branch.
    """

    if source_images.ndim != 4 or target_images.ndim != 4:
        raise ValueError("source_images and target_images must be BCHW tensors")
    if source_images.shape[1:] != target_images.shape[1:]:
        raise ValueError(
            "source and target tensors must share C/H/W, got "
            f"{tuple(source_images.shape[1:])} and {tuple(target_images.shape[1:])}"
        )
    if source_boxes.shape[:2] != source_valid.shape:
        raise ValueError("source_boxes/source_valid batch dimensions do not match")
    if source_boxes.shape[0] != source_images.shape[0]:
        raise ValueError("source box batch does not match source image batch")
    if target_images.shape[0] <= 0:
        raise ValueError("at least one target background is required")
    if max_pastes <= 0:
        raise ValueError("max_pastes must be positive")

    batch, _, image_h, image_w = source_images.shape
    composites = torch.empty_like(source_images)
    pasted_boxes = torch.zeros_like(source_boxes)
    pasted_valid = torch.zeros_like(source_valid)
    pasted_mask = torch.zeros(
        (batch, 1, image_h, image_w),
        device=source_images.device,
        dtype=source_images.dtype,
    )
    total_pastes = 0
    samples_with_paste = 0
    for batch_idx in range(batch):
        background = target_images[batch_idx % target_images.shape[0]].detach()
        composite = background.clone()
        rng = random.Random(int(seed) + 1009 * batch_idx)
        valid_indices = torch.nonzero(source_valid[batch_idx], as_tuple=False).flatten().tolist()
        rng.shuffle(valid_indices)
        occupied: List[Tuple[int, int, int, int]] = []
        output_idx = 0
        for source_idx in valid_indices:
            if output_idx >= min(max_pastes, source_boxes.shape[1]):
                break
            x1, y1, x2, y2 = [float(value) for value in source_boxes[batch_idx, source_idx].tolist()]
            sx1 = max(0, min(image_w - 1, int(math.floor(x1))))
            sy1 = max(0, min(image_h - 1, int(math.floor(y1))))
            sx2 = max(sx1 + 1, min(image_w, int(math.ceil(x2))))
            sy2 = max(sy1 + 1, min(image_h, int(math.ceil(y2))))
            patch_w, patch_h = sx2 - sx1, sy2 - sy1
            if patch_w < 2 or patch_h < 2 or patch_w > image_w or patch_h > image_h:
                continue
            destination: Tuple[int, int, int, int] | None = None
            for _ in range(max(1, int(placement_attempts))):
                dx1 = rng.randint(0, image_w - patch_w)
                dy1 = rng.randint(0, image_h - patch_h)
                candidate = (dx1, dy1, dx1 + patch_w, dy1 + patch_h)
                if not any(_boxes_overlap(candidate, previous) for previous in occupied):
                    destination = candidate
                    break
            if destination is None:
                continue
            dx1, dy1, dx2, dy2 = destination
            source_patch = source_images[batch_idx, :, sy1:sy2, sx1:sx2]
            alpha = _soft_box_alpha(
                patch_h,
                patch_w,
                int(feather),
                device=source_patch.device,
                dtype=source_patch.dtype,
                shape=alpha_shape,
            )
            target_patch = composite[:, dy1:dy2, dx1:dx2]
            composite[:, dy1:dy2, dx1:dx2] = alpha * source_patch + (1.0 - alpha) * target_patch
            pasted_boxes[batch_idx, output_idx] = torch.tensor(
                [dx1, dy1, dx2, dy2],
                device=pasted_boxes.device,
                dtype=pasted_boxes.dtype,
            )
            pasted_valid[batch_idx, output_idx] = True
            pasted_mask[batch_idx, :, dy1:dy2, dx1:dx2] = 1.0
            occupied.append(destination)
            output_idx += 1
        composites[batch_idx] = composite
        total_pastes += output_idx
        samples_with_paste += int(output_idx > 0)

    diagnostics = {
        "target_bg_pastes": float(total_pastes),
        "target_bg_pastes_per_image": float(total_pastes / max(batch, 1)),
        "target_bg_mask_fraction": float(pasted_mask.mean().detach().cpu()),
        "target_bg_samples_with_paste": float(samples_with_paste),
    }
    return composites, pasted_boxes, pasted_valid, pasted_mask, diagnostics


class SmallObjectManifestDataset(Dataset):
    def __init__(
        self,
        manifest_paths: List[Path],
        image_size: int,
        max_images: int,
        max_boxes: int,
        seed: int,
        source_weights: tuple[float, ...] = (),
        scale_resample: bool = False,
        scale_bins: str = "0:16:0.10,16:32:0.25,32:64:0.45,64:96:0.20",
        crop_augment: bool = False,
        crop_source_indices: tuple[int, ...] = (),
        crop_prob: float = 1.0,
        crop_target_min_side: float = 32.0,
        crop_target_max_side: float = 80.0,
        crop_target_bins: str = "",
        crop_max_box_side: float = 112.0,
        square_crop: bool = False,
        preserve_row_order: bool = False,
    ) -> None:
        if not manifest_paths:
            raise ValueError("At least one detection manifest is required.")
        source_rows: List[List[Dict[str, object]]] = []
        for source_idx, manifest_path in enumerate(manifest_paths):
            current: List[Dict[str, object]] = []
            with manifest_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    row["_det_source_index"] = source_idx
                    row["_det_source"] = str(row.get("source", manifest_path.parent.name))
                    current.append(row)
            if not current:
                raise RuntimeError(f"No rows found in {manifest_path}")
            source_rows.append(current)

        if source_weights and len(source_weights) != len(source_rows):
            raise ValueError(
                f"Expected {len(source_rows)} --det-source-weights values, got {len(source_weights)}."
            )
        weights = source_weights or tuple(1.0 for _ in source_rows)
        if max_images > 0 and len(source_rows) > 1:
            weight_sum = sum(weights)
            quotas = [int(math.floor(max_images * weight / weight_sum)) for weight in weights]
            for idx in sorted(range(len(weights)), key=lambda item: weights[item], reverse=True):
                if sum(quotas) >= max_images:
                    break
                quotas[idx] += 1
            quotas = [min(quota, len(current)) for quota, current in zip(quotas, source_rows)]
            while sum(quotas) < min(max_images, sum(len(current) for current in source_rows)):
                candidates = [
                    idx
                    for idx, current in enumerate(source_rows)
                    if quotas[idx] < len(current)
                ]
                if not candidates:
                    break
                idx = max(candidates, key=lambda item: weights[item] / max(quotas[item] + 1, 1))
                quotas[idx] += 1
        else:
            quotas = [min(len(current), max_images) if max_images > 0 else len(current) for current in source_rows]

        rows: List[Dict[str, object]] = []
        for source_idx, (current, quota) in enumerate(zip(source_rows, quotas)):
            rng = random.Random(seed + 1009 * source_idx)
            selected = rng.sample(current, quota) if quota < len(current) else list(current)
            rows.extend(selected)
        if not preserve_row_order:
            rows.sort(key=lambda row: (int(row["_det_source_index"]), str(row["image_path"])))
        self.rows = rows
        self.source_counts = Counter(str(row["_det_source"]) for row in rows)
        self.image_size = int(image_size)
        self.max_boxes = int(max_boxes)
        self.seed = int(seed)
        self.scale_resample = bool(scale_resample)
        self.scale_bins = parse_scale_bins(scale_bins)
        self.crop_augment = bool(crop_augment)
        self.crop_source_indices = set(int(value) for value in crop_source_indices)
        self.crop_prob = max(0.0, min(1.0, float(crop_prob)))
        self.crop_target_min_side = float(crop_target_min_side)
        self.crop_target_max_side = float(crop_target_max_side)
        self.crop_target_bins = parse_scale_bins(crop_target_bins) if crop_target_bins.strip() else []
        self.crop_max_box_side = float(crop_max_box_side)
        self.square_crop = bool(square_crop)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def _scale_weight(self, side: float) -> float:
        for low, high, weight in self.scale_bins:
            if low <= side < high:
                return weight
        return self.scale_bins[-1][2] if side >= self.scale_bins[-1][1] else self.scale_bins[0][2]

    def _resized_side(self, box: Dict[str, object], width: float, height: float) -> float:
        x1, y1, x2, y2 = [float(value) for value in box["bbox"]]  # type: ignore[index]
        return max((x2 - x1) * self.image_size / max(width, 1.0), (y2 - y1) * self.image_size / max(height, 1.0))

    def _sample_boxes(
        self,
        boxes: List[Dict[str, object]],
        rng: random.Random,
        width: float,
        height: float,
        max_boxes: int | None = None,
    ) -> List[Dict[str, object]]:
        limit = self.max_boxes if max_boxes is None else max_boxes
        if limit <= 0 or len(boxes) <= limit:
            return list(boxes)
        if not self.scale_resample:
            return rng.sample(boxes, limit)
        keyed = []
        for box in boxes:
            side = self._resized_side(box, width, height)
            keyed.append([box, self._scale_weight(side)])
        selected: List[Dict[str, object]] = []
        while keyed and len(selected) < limit:
            total = sum(float(item[1]) for item in keyed)
            pick = rng.random() * total
            acc = 0.0
            chosen = 0
            for idx, (_, weight) in enumerate(keyed):
                acc += float(weight)
                if acc >= pick:
                    chosen = idx
                    break
            selected.append(keyed.pop(chosen)[0])
        return selected

    def _choose_crop_target(
        self,
        boxes: List[Dict[str, object]],
        rng: random.Random,
        width: float,
        height: float,
    ) -> Dict[str, object]:
        prescribed = [box for box in boxes if bool(box.get("prior_anchor", False))]
        if prescribed:
            return prescribed[0]
        weighted = []
        for box in boxes:
            side = self._resized_side(box, width, height)
            weighted.append((box, self._scale_weight(side)))
        total = sum(weight for _, weight in weighted)
        pick = rng.random() * max(total, 1e-6)
        acc = 0.0
        for box, weight in weighted:
            acc += weight
            if acc >= pick:
                return box
        return boxes[-1]

    def _sample_crop_target_side(self, rng: random.Random) -> float:
        if not self.crop_target_bins:
            return rng.uniform(
                min(self.crop_target_min_side, self.crop_target_max_side),
                max(self.crop_target_min_side, self.crop_target_max_side),
            )
        total = sum(weight for _, _, weight in self.crop_target_bins)
        pick = rng.random() * total
        acc = 0.0
        for low, high, weight in self.crop_target_bins:
            acc += weight
            if acc >= pick:
                return rng.uniform(low, high)
        low, high, _ = self.crop_target_bins[-1]
        return rng.uniform(low, high)

    @staticmethod
    def _bounded_interval(center: float, size: float, limit: float) -> Tuple[float, float]:
        size = min(max(size, 1.0), limit)
        start = center - size * 0.5
        start = min(max(start, 0.0), max(limit - size, 0.0))
        return start, start + size

    def _crop_row(
        self,
        image: Image.Image,
        boxes: List[Dict[str, object]],
        rng: random.Random,
    ) -> Tuple[Image.Image, List[Dict[str, object]], Tuple[int, int, int, int] | None]:
        if not boxes:
            return image, boxes, None
        width, height = image.size
        target = self._choose_crop_target(boxes, rng, float(width), float(height))
        x1, y1, x2, y2 = [float(value) for value in target["bbox"]]  # type: ignore[index]
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        # Prior-controlled manifests may prescribe the model-space target side
        # per anchor.  Keeping the override on the box makes resize-only and
        # crop variants share an auditable token-area schedule.
        target_side = float(target.get("crop_target_side", self._sample_crop_target_side(rng)))
        if self.square_crop:
            crop_side = max(max(bw, bh) * self.image_size / max(target_side, 1.0), max(bw, bh) * 2.0)
            crop_w = crop_h = crop_side
        else:
            crop_w = max(bw * self.image_size / max(target_side, 1.0), bw * 2.0)
            crop_h = max(bh * self.image_size / max(target_side, 1.0), bh * 2.0)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        crop_x1, crop_x2 = self._bounded_interval(cx, crop_w, float(width))
        crop_y1, crop_y2 = self._bounded_interval(cy, crop_h, float(height))
        crop_left = int(math.floor(crop_x1))
        crop_top = int(math.floor(crop_y1))
        crop_right = int(math.ceil(crop_x2))
        crop_bottom = int(math.ceil(crop_y2))
        crop = image.crop((crop_left, crop_top, crop_right, crop_bottom))
        crop_x1 = float(crop_left)
        crop_y1 = float(crop_top)
        crop_x2 = float(crop_right)
        crop_y2 = float(crop_bottom)
        actual_w = max(crop_x2 - crop_x1, 1.0)
        actual_h = max(crop_y2 - crop_y1, 1.0)
        scale_x = self.image_size / actual_w
        scale_y = self.image_size / actual_h
        cropped_boxes: List[Dict[str, object]] = []
        for box in boxes:
            bx1, by1, bx2, by2 = [float(value) for value in box["bbox"]]  # type: ignore[index]
            ix1 = max(bx1, crop_x1)
            iy1 = max(by1, crop_y1)
            ix2 = min(bx2, crop_x2)
            iy2 = min(by2, crop_y2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            nx1 = ix1 - crop_x1
            ny1 = iy1 - crop_y1
            nx2 = ix2 - crop_x1
            ny2 = iy2 - crop_y1
            resized_w = (nx2 - nx1) * scale_x
            resized_h = (ny2 - ny1) * scale_y
            if min(resized_w, resized_h) < 2.0:
                continue
            if max(resized_w, resized_h) > self.crop_max_box_side:
                continue
            token_area = token_area_for_resized_box(resized_w, resized_h, self.image_size)
            if token_area < 1:
                continue
            cropped = dict(box)
            cropped["bbox"] = [nx1, ny1, nx2, ny2]
            cropped["resized_w"] = resized_w
            cropped["resized_h"] = resized_h
            cropped["resized_area"] = resized_w * resized_h
            cropped["token_area"] = token_area
            cropped_boxes.append(cropped)
        if not cropped_boxes:
            return image, boxes, None
        return crop, cropped_boxes, (crop_left, crop_top, crop_right, crop_bottom)

    @staticmethod
    def _all_object_mask(row: Dict[str, object], image_size: Tuple[int, int]) -> Image.Image:
        mask = Image.new("L", image_size, 0)
        draw = ImageDraw.Draw(mask)
        manifest_boxes = row.get("all_object_boxes")
        if isinstance(manifest_boxes, list):
            for box in manifest_boxes:
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    continue
                x1, y1, x2, y2 = [float(value) for value in box]
                x1 = max(0, min(image_size[0], int(math.floor(x1))))
                y1 = max(0, min(image_size[1], int(math.floor(y1))))
                x2 = max(0, min(image_size[0], int(math.ceil(x2))))
                y2 = max(0, min(image_size[1], int(math.ceil(y2))))
                if x2 > x1 and y2 > y1:
                    draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=255)
            return mask
        annotation_path = Path(str(row.get("annotation_path", "")))
        if annotation_path.suffix.lower() == ".json" and annotation_path.exists():
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
            for ann in payload.get("annotations", []):
                poly = [float(value) for value in ann.get("poly", [])]
                if len(poly) < 8:
                    continue
                xs, ys = poly[0::2], poly[1::2]
                draw.rectangle((min(xs), min(ys), max(xs), max(ys)), fill=255)
            return mask
        annotation_spec = str(row.get("annotation_path", ""))
        if "!" in annotation_spec:
            zip_name, member = annotation_spec.split("!", 1)
            zip_path = Path(zip_name)
            if zip_path.exists():
                with zipfile.ZipFile(zip_path) as archive:
                    payload = json.loads(archive.read(member).decode("utf-8"))
                for ann in payload.get("annotations", []):
                    poly = [float(value) for value in ann.get("poly", [])]
                    if len(poly) < 8:
                        continue
                    xs, ys = poly[0::2], poly[1::2]
                    draw.rectangle((min(xs), min(ys), max(xs), max(ys)), fill=255)
            return mask
        if not annotation_path.exists():
            return mask
        with annotation_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 4:
                    continue
                try:
                    x, y, width, height = (float(parts[idx]) for idx in range(4))
                except ValueError:
                    continue
                if width <= 0.0 or height <= 0.0:
                    continue
                x1 = max(0, min(image_size[0], int(math.floor(x))))
                y1 = max(0, min(image_size[1], int(math.floor(y))))
                x2 = max(0, min(image_size[0], int(math.ceil(x + width))))
                y2 = max(0, min(image_size[1], int(math.ceil(y + height))))
                if x2 > x1 and y2 > y1:
                    draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=255)
        return mask

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        row = self.rows[index]
        image_path = Path(str(row["image_path"]))
        width = float(row["width"])
        height = float(row["height"])
        rng = random.Random(self.seed + index)
        boxes = list(row["boxes"])  # type: ignore[arg-type]
        image = Image.open(image_path).convert("RGB")
        all_object_mask = self._all_object_mask(row, image.size)
        source_idx = int(row.get("_det_source_index", 0))
        crop_source_enabled = not self.crop_source_indices or source_idx in self.crop_source_indices
        if self.crop_augment and crop_source_enabled and rng.random() < self.crop_prob:
            image, boxes, crop_box = self._crop_row(image, boxes, rng)
            if crop_box is not None:
                all_object_mask = all_object_mask.crop(crop_box)
            width, height = image.size
        boxes = self._sample_boxes(boxes, rng, width, height)
        padded = torch.zeros((self.max_boxes, 4), dtype=torch.float32)
        valid = torch.zeros((self.max_boxes,), dtype=torch.bool)
        scale_x = self.image_size / max(width, 1.0)
        scale_y = self.image_size / max(height, 1.0)
        for box_idx, box in enumerate(boxes[: self.max_boxes]):
            x1, y1, x2, y2 = [float(value) for value in box["bbox"]]  # type: ignore[index]
            padded[box_idx] = torch.tensor([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y], dtype=torch.float32)
            valid[box_idx] = True
        resized_object_mask = all_object_mask.resize((self.image_size, self.image_size), Image.NEAREST)
        object_mask_tensor = torch.from_numpy((np.asarray(resized_object_mask) > 0).astype(np.float32)).unsqueeze(0)
        source = str(row.get("_det_source", "det"))
        return self.transform(image), padded, valid, object_mask_tensor, f"{source}::{image_path.name}"


class NormalFolderDataset(Dataset):
    def __init__(self, root: Path, image_size: int) -> None:
        self.paths = [
            path
            for path in sorted(root.iterdir() if root.exists() else [])
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        ]
        if not self.paths:
            raise RuntimeError(f"No normal images found in {root}")
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        image_path = self.paths[index]
        return self.transform(Image.open(image_path).convert("RGB")), torch.tensor(0, dtype=torch.long), str(image_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FOD reconstruction with small-object erasing auxiliary loss.")
    parser.add_argument("--fod-root", type=Path, default=PROJECT_ROOT.parent / "datasets" / "FOD_dataset_ad_50_50_grouped")
    parser.add_argument("--normal-root", type=Path, default=None, help="Optional folder of normal images; overrides --fod-root crop loading.")
    parser.add_argument("--layout", choices=["auto", "original", "mvtec"], default="original")
    parser.add_argument("--train-distance", default="05")
    parser.add_argument("--visdrone-manifest", type=Path, default=None, help="Legacy single auxiliary-data manifest.")
    parser.add_argument("--det-manifests", type=Path, nargs="+", default=None, help="One or more auxiliary small-object manifests.")
    parser.add_argument("--det-source-weights", default="", help="Relative image quotas for --det-manifests, for example 3,1.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--external-repo", type=Path, default=PROJECT_ROOT.parent / "Dinomaly")
    parser.add_argument("--inpformer-repo", type=Path, default=PROJECT_ROOT.parent / "INP-Former")
    parser.add_argument("--architecture", choices=["dinomaly", "inpformer"], default="inpformer")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument("--encoder", default="dinov2reg_vit_base_14")
    parser.add_argument("--inp-num", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument(
        "--normal-crop-size",
        type=int,
        default=0,
        help="Physical FOD normal crop size before resizing to --image-size; 0 uses --image-size.",
    )
    parser.add_argument("--patch-full-height", type=int, default=0)
    parser.add_argument("--patch-stride", type=int, default=224)
    parser.add_argument("--ground-roi-mask", type=Path, default=None)
    parser.add_argument("--ground-roi-min-coverage", type=float, default=0.0)
    parser.add_argument(
        "--roi-mask-loss",
        action="store_true",
        help="Restrict every spatial Normal reconstruction loss to the crop Clean ROI.",
    )
    parser.add_argument(
        "--roi-mask-erode-pixels",
        type=int,
        default=0,
        help="Erode the training Clean ROI in physical pre-resize pixels.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--det-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--total-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--lr-schedule", choices=["fixed", "cosine"], default="fixed")
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-warmup-start-factor", type=float, default=0.2)
    parser.add_argument(
        "--lr-decay-start-step",
        type=int,
        default=0,
        help=(
            "For cosine scheduling, keep --lr fixed through this optimizer step, "
            "then decay to --min-lr at --total-steps. Zero preserves the original schedule."
        ),
    )
    parser.add_argument(
        "--lr-decay-end-step",
        type=int,
        default=0,
        help=(
            "Optional absolute step at which cosine decay reaches --min-lr. "
            "Later steps stay at --min-lr; zero uses --total-steps."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-det-images", type=int, default=4096)
    parser.add_argument("--max-boxes-per-image", type=int, default=32)
    parser.add_argument("--det-scale-resample", action="store_true")
    parser.add_argument("--det-scale-bins", default="0:16:0.10,16:32:0.25,32:64:0.45,64:96:0.20")
    parser.add_argument("--det-crop-augment", action="store_true")
    parser.add_argument("--det-crop-source-indices", default="", help="Comma-separated manifest indices to crop; empty crops every source.")
    parser.add_argument("--det-crop-prob", type=float, default=1.0)
    parser.add_argument("--det-crop-target-min-side", type=float, default=32.0)
    parser.add_argument("--det-crop-target-max-side", type=float, default=80.0)
    parser.add_argument("--det-crop-target-bins", default="", help="Weighted model-pixel target-side bins, e.g. 18:24:0.25,24:42:0.35.")
    parser.add_argument("--det-crop-max-box-side", type=float, default=112.0)
    parser.add_argument("--det-square-crop", action="store_true", help="Use a square crop so auxiliary objects keep their aspect ratio.")
    parser.add_argument(
        "--det-target-background-compose",
        action="store_true",
        help=(
            "Paste selected auxiliary-object boxes onto the current target-domain Normal batch before "
            "object erasing. The Normal memory/prototype state is not updated from the composite branch."
        ),
    )
    parser.add_argument(
        "--det-target-background-max-pastes",
        type=int,
        default=24,
        help="Maximum non-overlapping auxiliary boxes pasted onto each target Normal background.",
    )
    parser.add_argument(
        "--det-target-background-feather",
        type=int,
        default=1,
        help="Model-pixel feather width for pasted box boundaries; zero is exact rectangular CutPaste.",
    )
    parser.add_argument(
        "--det-target-background-alpha-shape",
        choices=["box", "ellipse"],
        default="box",
        help="Alpha support for target-background pastes; ellipse removes box-corner source context.",
    )
    parser.add_argument(
        "--det-no-shuffle",
        action="store_true",
        help="Consume the detection manifest in order (used by exact token-prior schedules).",
    )
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=0)
    parser.add_argument("--latest-interval", type=int, default=50, help="Overwrite latest.pth every N steps for resume; 0 disables step-based latest saves.")
    parser.add_argument(
        "--stop-after-step",
        type=int,
        default=0,
        help=(
            "Stop cleanly after this optimizer step while preserving the original "
            "--total-steps schedules. A resumable phase_step_XXXXXX.pth is always saved."
        ),
    )
    parser.add_argument(
        "--final-only-checkpoints",
        action="store_true",
        help="Do not write epoch/interval optimizer checkpoints; keep only lightweight final/validation model weights.",
    )
    parser.add_argument("--resume", type=Path, default=None, help="Resume full training state from a checkpoint produced by this script.")
    parser.add_argument("--auto-resume", action="store_true", help="Resume from output_dir/latest.pth when it exists.")
    parser.add_argument(
        "--reset-resume-rng-seed",
        type=int,
        default=-1,
        help=(
            "After restoring model/optimizer state, replace the checkpoint RNG state "
            "with this seed; negative keeps the exact original continuation order."
        ),
    )
    parser.add_argument(
        "--normal-loss",
        choices=["native", "hard_weighted", "legacy_plain", "inp_soft_mining"],
        default="native",
    )
    parser.add_argument(
        "--normal-loss-weight",
        type=float,
        default=1.0,
        help=(
            "Scale the complete normal branch objective so auxiliary detection exposure can "
            "remain fixed when the normal training set size changes."
        ),
    )
    parser.add_argument(
        "--normal-loss-step-offset",
        type=int,
        default=0,
        help="Continue step-dependent native-loss schedules from a warm-start checkpoint.",
    )
    parser.add_argument("--prototype-loss-weight", type=float, default=0.2)
    parser.add_argument("--hard-quantile", type=float, default=0.9)
    parser.add_argument("--easy-weight", type=float, default=0.1)
    parser.add_argument("--dinomaly-native-p-final", type=float, default=0.9)
    parser.add_argument("--dinomaly-native-warmup-iters", type=int, default=1000)
    parser.add_argument("--dinomaly-native-factor", type=float, default=0.1)
    parser.add_argument("--inpformer-native-y", type=float, default=3.0)
    parser.add_argument("--inp-coherence-loss", choices=["hard", "soft"], default="hard")
    parser.add_argument("--inp-soft-mining-gamma", type=float, default=3.0)
    parser.add_argument("--masked-recon", action="store_true", help="Use the perspective masked reconstruction loss for the normal branch.")
    parser.add_argument("--mask-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--mask-strategy",
        choices=["perspective", "uniform"],
        default="perspective",
        help=(
            "Structural mask family for the Normal branch. The default preserves "
            "the historical perspective-mask behavior."
        ),
    )
    parser.add_argument("--mask-segments", type=int, default=4)
    parser.add_argument("--mask-band-block-sizes", default="1,1,2,2,4,4")
    parser.add_argument("--mask-fill", choices=["visible_mean", "zero"], default="visible_mean")
    parser.add_argument("--mask-prototype-source", choices=["masked", "full"], default="masked")
    parser.add_argument("--mask-band-loss-weights", default="")
    parser.add_argument("--full-anchor-loss-weight", type=float, default=0.0)
    parser.add_argument("--det-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--sequential-branch-backward",
        action="store_true",
        help=(
            "Backpropagate the normal branch before forwarding the detection branch. "
            "This preserves the summed gradient while reducing peak activation memory."
        ),
    )
    parser.add_argument(
        "--gradient-conflict-probe",
        action="store_true",
        help=(
            "Measure normal-vs-detection gradient cosine/norm conflicts during "
            "sequential backward without changing the accumulated gradient."
        ),
    )
    parser.add_argument(
        "--gradient-conflict-probe-every",
        type=int,
        default=1,
        help="Probe every N optimizer steps; requires --sequential-branch-backward.",
    )
    parser.add_argument(
        "--gradient-conflict-probe-output",
        type=Path,
        default=None,
        help="CSV output path; defaults to output_dir/gradient_conflict.csv.",
    )
    parser.add_argument(
        "--gradient-conflict-component-control",
        action="store_true",
        help=(
            "Decompose the sequential detection gradient and scale only its "
            "component opposing the normal gradient."
        ),
    )
    parser.add_argument(
        "--gradient-conflict-component-scale",
        type=float,
        default=1.0,
        help="Multiplier k for the opposing detection-gradient component.",
    )
    parser.add_argument(
        "--gradient-conflict-component-budget-beta",
        type=float,
        default=None,
        help=(
            "Optional per-step anchor descent budget beta in [0,1]. When set, "
            "it replaces the fixed conflict scale while conflict control is active."
        ),
    )
    parser.add_argument(
        "--gradient-conflict-component-anchor",
        choices=["normal", "normal_rank"],
        default="normal",
        help=(
            "Gradient accumulated before legacy OE is projected. normal_rank "
            "requires separate score-rank backward and protects Normal+Local-rank."
        ),
    )
    parser.add_argument(
        "--gradient-conflict-component-start-step",
        type=int,
        default=1,
        help=(
            "Use conflict scale 1 before this step, then switch to "
            "--gradient-conflict-component-scale."
        ),
    )
    parser.add_argument(
        "--gradient-conflict-component-groups",
        default="decoder",
        help="Comma-separated coarse parameter groups to control.",
    )
    parser.add_argument(
        "--gradient-conflict-component-output",
        type=Path,
        default=None,
        help="CSV output path; defaults to output_dir/conflict_component_control.csv.",
    )
    parser.add_argument("--det-loss-warmup-steps", type=int, default=0)
    parser.add_argument("--det-loss-start-frac", type=float, default=0.0)
    parser.add_argument("--det-loss-end-frac", type=float, default=0.0)
    parser.add_argument("--lambda-det-bg", type=float, default=1.0)
    parser.add_argument(
        "--det-background-mode",
        choices=["selected", "none", "clean"],
        default="selected",
        help=(
            "VisDrone background supervision: selected uses the legacy complement of sampled targets; "
            "none disables background reconstruction; clean reconstructs only pixels outside every annotation."
        ),
    )
    parser.add_argument("--det-clean-bg-dilate", type=int, default=1)
    parser.add_argument("--lambda-obj-bg", type=float, default=1.0)
    parser.add_argument("--lambda-obj-sep", type=float, default=0.1)
    parser.add_argument("--lambda-smooth", type=float, default=0.1)
    parser.add_argument("--lambda-score-rank", type=float, default=0.0)
    parser.add_argument(
        "--score-rank-start-step",
        type=int,
        default=1,
        help="Keep score-rank disabled before this optimizer step.",
    )
    parser.add_argument("--lambda-score-cover", type=float, default=0.0)
    parser.add_argument(
        "--score-cover-start-step",
        type=int,
        default=1,
        help="Keep dense object lower-tail coverage disabled before this optimizer step.",
    )
    parser.add_argument(
        "--score-rank-separate-backward",
        action="store_true",
        help=(
            "Apply normal-conflict projection to legacy OE first, then add the "
            "score-rank gradient without projecting it away."
        ),
    )
    parser.add_argument(
        "--score-rank-background-mode",
        choices=["local_ring", "global_clean"],
        default="local_ring",
    )
    parser.add_argument("--score-rank-global-topk-frac", type=float, default=0.01)
    parser.add_argument("--score-rank-object-low-frac", type=float, default=0.25)
    parser.add_argument(
        "--score-rank-gradient-probe-every",
        type=int,
        default=1,
        help="Measure the separately applied weighted rank-gradient norm every N steps.",
    )
    parser.add_argument(
        "--rank-gradient-audit-only",
        action="store_true",
        help="Run rank-gradient diagnostics without optimizer updates or model checkpoints.",
    )
    parser.add_argument("--rank-gradient-audit-steps", type=int, default=32)
    parser.add_argument(
        "--rank-gradient-audit-target-ratio",
        type=float,
        default=0.15,
        help="Target rank/legacy effective decoder-gradient ratio used for weight calibration.",
    )
    parser.add_argument("--rank-gradient-audit-output", type=Path, default=None)
    parser.add_argument("--lambda-normal-fp", type=float, default=0.0)
    parser.add_argument("--normal-texture-preserve-weight", type=float, default=0.0)
    parser.add_argument("--normal-texture-start-frac", type=float, default=0.0)
    parser.add_argument("--normal-texture-end-frac", type=float, default=0.0)
    parser.add_argument("--normal-texture-not-bg-weight", type=float, default=0.2)
    parser.add_argument("--normal-texture-top-frac", type=float, default=0.15)
    parser.add_argument("--normal-texture-margin", type=float, default=0.05)
    parser.add_argument("--normal-prior-recon-weight", type=float, default=0.0)
    parser.add_argument("--normal-prior-start-frac", type=float, default=0.0)
    parser.add_argument("--normal-prior-end-frac", type=float, default=0.0)
    parser.add_argument("--normal-prior-texture-weight", type=float, default=0.7)
    parser.add_argument("--normal-prior-object-weight", type=float, default=0.3)
    parser.add_argument("--normal-prior-power", type=float, default=1.0)
    parser.add_argument("--normal-prior-top-frac", type=float, default=0.30)
    parser.add_argument(
        "--hn-oe-sidecar",
        type=Path,
        default=None,
        help="Frozen train-normal token sidecar containing the OE-raised selector.",
    )
    parser.add_argument(
        "--hn-oe-normal-crops-csv",
        type=Path,
        default=None,
        help="Crop manifest used to map global HN-OE coordinates into normal batches.",
    )
    parser.add_argument("--hn-oe-loss-weight", type=float, default=0.0)
    parser.add_argument("--hn-oe-margin", type=float, default=0.005)
    parser.add_argument("--hn-oe-start-step", type=int, default=91)
    parser.add_argument(
        "--hn-oe-gradient-scope",
        choices=("decoder", "all"),
        default="decoder",
        help=(
            "decoder applies the causal-restore gradient only to bottleneck/decoder; "
            "all additionally lets it update aggregation/prototype parameters."
        ),
    )
    parser.add_argument(
        "--native-bg-teacher-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional frozen Native+OE teacher used only to suppress Adaptive "
            "background score excess on target-composite images."
        ),
    )
    parser.add_argument("--native-bg-excess-weight", type=float, default=0.0)
    parser.add_argument("--native-bg-excess-min-tokens", type=int, default=8)
    parser.add_argument("--native-bg-excess-target-dilate", type=int, default=1)
    parser.add_argument("--det-objectness-weighting", action="store_true")
    parser.add_argument("--det-objectness-min-weight", type=float, default=0.15)
    parser.add_argument("--det-objectness-power", type=float, default=1.0)
    parser.add_argument("--det-size-weighting", action="store_true")
    parser.add_argument("--det-size-focus-min-side", type=float, default=32.0)
    parser.add_argument("--det-size-focus-max-side", type=float, default=64.0)
    parser.add_argument("--det-size-focus-weight", type=float, default=2.0)
    parser.add_argument("--det-core-ring-erasing", action="store_true")
    parser.add_argument("--det-core-frac", type=float, default=0.60)
    parser.add_argument(
        "--det-hard-residual-mining",
        action="store_true",
        help="Focus legacy OE gradients on object tokens that still have the smallest reconstruction residual.",
    )
    parser.add_argument("--det-hard-residual-frac", type=float, default=0.50)
    parser.add_argument("--det-hard-residual-min-weight", type=float, default=0.10)
    parser.add_argument("--det-instance-local-erasing", action="store_true", help="Compute core/ring erasing per box instead of using one union ring per image.")
    parser.add_argument("--lambda-boundary-preserve", type=float, default=0.0)
    parser.add_argument("--objectness-freq-weight", type=float, default=0.25)
    parser.add_argument("--objectness-edge-weight", type=float, default=0.25)
    parser.add_argument("--objectness-dino-weight", type=float, default=0.50)
    parser.add_argument("--objectness-smooth-kernel", type=int, default=3)
    parser.add_argument(
        "--det-prototype-mode",
        choices=["joint", "freeze", "normal_context", "paired_normal_context"],
        default="joint",
        help=(
            "How the VisDrone/object-erasing branch uses INP-Former prototypes. "
            "'joint' keeps the original behavior; 'freeze' stops det gradients into "
            "prototype tokens and aggregation; 'normal_context' decodes det images "
            "with one detached prototype context averaged across the same-step normal batch; "
            "'paired_normal_context' uses the corresponding pre-composition Normal sample's "
            "detached context for each target-background composite."
        ),
    )
    parser.add_argument("--sep-sim-margin", type=float, default=0.30)
    parser.add_argument("--score-rank-margin", type=float, default=0.05)
    parser.add_argument("--score-cover-margin", type=float, default=0.02)
    parser.add_argument("--normal-fp-margin", type=float, default=0.08)
    parser.add_argument("--normal-fp-topk-frac", type=float, default=0.02)
    parser.add_argument("--ring-radius", type=int, default=2)
    add_guided_prototype_args(parser)
    parser.add_argument(
        "--guided-prototype-prior-head-mode",
        choices=["train", "freeze"],
        default="train",
        help="When a trainable guided prior is configured, either update or freeze its inherited head during OE.",
    )
    parser.add_argument(
        "--target-gate-loss-weight",
        type=float,
        default=0.0,
        help="Balanced pasted-target token supervision weight for the familiarity risk calibrator.",
    )
    parser.add_argument(
        "--target-gate-normal-anchor-weight",
        type=float,
        default=0.0,
        help=(
            "Legacy ablation option: MSE weight keeping calibrated clean-Normal risk "
            "near the inherited risk. The reported OAL-FOD objective uses 0."
        ),
    )
    parser.add_argument(
        "--target-gate-lr",
        type=float,
        default=0.0,
        help="Calibrator learning rate; zero uses the main learning rate.",
    )
    parser.add_argument(
        "--target-gate-warmup-steps",
        type=int,
        default=0,
        help="Train only the target-gate module for the first N optimizer steps.",
    )
    parser.add_argument(
        "--target-proto-invariance-weight",
        type=float,
        default=0.0,
        help="Paired clean/composite prototype cosine-invariance loss weight (E1+).",
    )
    parser.add_argument(
        "--target-proto-repulsion-weight",
        type=float,
        default=0.0,
        help="Target-core nearest-Normal-prototype hinge repulsion weight (E2+).",
    )
    parser.add_argument("--target-proto-repulsion-normal-quantile", type=float, default=0.99)
    parser.add_argument("--target-proto-repulsion-margin-delta", type=float, default=0.02)
    parser.add_argument("--target-proto-core-min-occupancy", type=float, default=0.01)
    parser.add_argument(
        "--target-proto-repulsion-margin-scope",
        choices=["global", "effective_mode"],
        default="global",
        help=(
            "Use one global Normal distance tail or one tail per effective adaptive "
            "Normal mode. The latter requires an adaptive Center teacher."
        ),
    )
    parser.add_argument(
        "--target-proto-repulsion-gradient-side",
        choices=["prototype", "target"],
        default="prototype",
        help=(
            "Backpropagate ProtoRep through image-conditioned prototype aggregation "
            "or through trainable pre-decoder bottleneck target representations. "
            "The opposite side is stop-gradient."
        ),
    )
    parser.add_argument(
        "--target-proto-repulsion-min-normal-tokens-per-mode",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--target-proto-repulsion-mode-budget",
        choices=["none", "clean_target_ratio"],
        default="none",
        help=(
            "Optionally downweight per-effective-mode ProtoRep by detached "
            "min(1, clean-Normal token share / target-core token share)."
        ),
    )
    parser.add_argument(
        "--target-aggregation-attention-weight",
        type=float,
        default=0.0,
        help="Target-density suppression plus clean-background aggregation-attention anchor weight (E3+).",
    )
    parser.add_argument("--target-aggregation-attention-ratio", type=float, default=0.25)
    parser.add_argument("--target-aggregation-background-anchor-weight", type=float, default=1.0)
    parser.add_argument(
        "--target-read-attention-weight",
        type=float,
        default=0.0,
        help="Target decoder-read suppression and clean-background anchor weight (E4).",
    )
    parser.add_argument("--target-read-attention-ratio", type=float, default=0.25)
    parser.add_argument("--target-read-background-anchor-weight", type=float, default=1.0)
    inline_val = parser.add_argument_group("inline FOD validation")
    inline_val.add_argument("--inline-fod-val-every-epochs", type=int, default=0)
    inline_val.add_argument("--inline-fod-val-start-epoch", type=int, default=0)
    inline_val.add_argument("--inline-fod-val-output-dir", type=Path, default=None)
    inline_val.add_argument(
        "--inline-fod-val-distances",
        nargs="+",
        default=["05", "10", "15", "20", "25", "30"],
    )
    inline_val.add_argument("--inline-fod-val-stride", type=int, default=0)
    inline_val.add_argument("--inline-fod-val-batch-size", type=int, default=8)
    inline_val.add_argument("--inline-fod-val-progress-every", type=int, default=16)
    inline_val.add_argument("--inline-fod-val-min-pred-area", type=int, default=16)
    inline_val.add_argument("--inline-fod-val-iou-threshold", type=float, default=0.1)
    inline_val.add_argument("--inline-fod-val-seed", type=int, default=20260715)
    inline_val.add_argument("--inline-fod-val-keep-cache", action="store_true")
    return parser.parse_args()


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = [item.cpu() for item in state["cuda"]]
        visible_devices = torch.cuda.device_count()
        if cuda_states and visible_devices > 0:
            # A checkpoint may have been written with more visible GPUs than a
            # single-GPU continuation process.  Logical cuda:0 should restore
            # the original primary-device state on every physical GPU branch.
            if len(cuda_states) < visible_devices:
                cuda_states.extend(
                    cuda_states[-1].clone()
                    for _ in range(visible_devices - len(cuda_states))
                )
            torch.cuda.set_rng_state_all(cuda_states[:visible_devices])


def resolve_resume_path(args: argparse.Namespace) -> Path | None:
    if args.resume is not None:
        return args.resume
    if args.auto_resume:
        latest = args.output_dir / "latest.pth"
        if latest.exists():
            return latest
    return None


def merged_feature(features: List[torch.Tensor], detach: bool) -> torch.Tensor:
    normalized = [F.normalize(item.detach() if detach else item, dim=1) for item in features]
    return F.normalize(torch.stack(normalized, dim=0).mean(dim=0), dim=1)


def boxes_to_token_mask(boxes: torch.Tensor, valid: torch.Tensor, token_h: int, token_w: int, image_size: int) -> torch.Tensor:
    mask = torch.zeros((boxes.shape[0], 1, token_h, token_w), dtype=torch.bool, device=boxes.device)
    token_w_px = float(image_size) / float(token_w)
    token_h_px = float(image_size) / float(token_h)
    for b in range(boxes.shape[0]):
        for box_idx in torch.nonzero(valid[b], as_tuple=False).flatten().tolist():
            x1, y1, x2, y2 = boxes[b, box_idx].tolist()
            tx1 = max(0, min(token_w, int(np.floor(x1 / token_w_px))))
            ty1 = max(0, min(token_h, int(np.floor(y1 / token_h_px))))
            tx2 = max(0, min(token_w, int(np.ceil(x2 / token_w_px))))
            ty2 = max(0, min(token_h, int(np.ceil(y2 / token_h_px))))
            if tx2 > tx1 and ty2 > ty1:
                mask[b, :, ty1:ty2, tx1:tx2] = True
    return mask


def boxes_to_size_weight_map(boxes: torch.Tensor, valid: torch.Tensor, token_h: int, token_w: int, args: argparse.Namespace) -> torch.Tensor:
    weights = torch.ones((boxes.shape[0], 1, token_h, token_w), dtype=torch.float32, device=boxes.device)
    if not args.det_size_weighting:
        return weights
    token_w_px = float(args.image_size) / float(token_w)
    token_h_px = float(args.image_size) / float(token_h)
    focus_min = float(args.det_size_focus_min_side)
    focus_max = float(args.det_size_focus_max_side)
    focus_weight = max(1.0, float(args.det_size_focus_weight))
    for b in range(boxes.shape[0]):
        for box_idx in torch.nonzero(valid[b], as_tuple=False).flatten().tolist():
            x1, y1, x2, y2 = boxes[b, box_idx].tolist()
            side = max(x2 - x1, y2 - y1)
            if not (focus_min <= side <= focus_max):
                continue
            tx1 = max(0, min(token_w, int(np.floor(x1 / token_w_px))))
            ty1 = max(0, min(token_h, int(np.floor(y1 / token_h_px))))
            tx2 = max(0, min(token_w, int(np.ceil(x2 / token_w_px))))
            ty2 = max(0, min(token_h, int(np.ceil(y2 / token_h_px))))
            if tx2 > tx1 and ty2 > ty1:
                weights[b, :, ty1:ty2, tx1:tx2] = torch.maximum(
                    weights[b, :, ty1:ty2, tx1:tx2],
                    weights.new_full((1, ty2 - ty1, tx2 - tx1), focus_weight),
                )
    return weights


def masked_cosine_distance(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    dist = (1.0 - F.cosine_similarity(pred, target.detach(), dim=1, eps=1e-6)).unsqueeze(1).clamp_min(0.0)
    weight = mask.float()
    return (dist * weight).sum() / weight.sum().clamp_min(1.0)


def ring_from_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    kernel = max(1, radius * 2 + 1)
    dilated = F.max_pool2d(mask.float(), kernel_size=kernel, stride=1, padding=radius) > 0
    return dilated & ~mask


def quantile_loss_value(values: torch.Tensor, q: float) -> torch.Tensor:
    q = max(0.0, min(1.0, float(q)))
    if values.numel() <= 1:
        return values.mean()
    return torch.quantile(values.float(), q).to(values.dtype)


def scheduled_weight(
    max_weight: float,
    step: int,
    total_steps: int,
    start_frac: float,
    end_frac: float,
    warmup_steps: int = 0,
) -> float:
    max_weight = float(max_weight)
    if max_weight <= 0.0:
        return 0.0
    start_frac = max(0.0, min(1.0, float(start_frac)))
    end_frac = max(0.0, min(1.0, float(end_frac)))
    if end_frac > start_frac:
        start_step = max(1, int(round(total_steps * start_frac)))
        end_step = max(start_step + 1, int(round(total_steps * end_frac)))
        if step <= start_step:
            return 0.0
        if step >= end_step:
            return max_weight
        progress = (step - start_step) / float(max(end_step - start_step, 1))
        return max_weight * progress * progress
    if warmup_steps > 0:
        return max_weight * min(1.0, step / float(warmup_steps))
    return max_weight


def scheduled_learning_rate(
    base_lr: float,
    min_lr: float,
    step: int,
    total_steps: int,
    schedule: str,
    warmup_steps: int = 0,
    warmup_start_factor: float = 0.2,
    decay_start_step: int = 0,
    decay_end_step: int = 0,
) -> float:
    base_lr = float(base_lr)
    min_lr = max(0.0, min(float(min_lr), base_lr))
    warmup_steps = max(0, min(int(warmup_steps), int(total_steps)))
    warmup_start_factor = max(0.0, min(1.0, float(warmup_start_factor)))
    if warmup_steps > 0 and step <= warmup_steps:
        progress = step / float(warmup_steps)
        factor = warmup_start_factor + (1.0 - warmup_start_factor) * progress
        return base_lr * factor
    if schedule == "fixed":
        return base_lr
    if schedule != "cosine":
        raise ValueError(f"Unsupported lr schedule: {schedule}")
    decay_start_step = max(int(decay_start_step), warmup_steps, 0)
    if step <= decay_start_step:
        return base_lr
    decay_end_step = int(decay_end_step)
    if decay_end_step <= 0:
        decay_end_step = int(total_steps)
    decay_end_step = max(decay_end_step, decay_start_step + 1)
    decay_steps = max(decay_end_step - decay_start_step, 1)
    progress = max(0.0, min(1.0, (step - decay_start_step) / float(decay_steps)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def activated_step_value(
    active_value: float,
    step: int,
    start_step: int,
    *,
    inactive_value: float,
) -> float:
    """Switch a scalar training control on at an exact optimizer step."""

    if start_step <= 0:
        raise ValueError("start_step must be positive.")
    return float(active_value) if int(step) >= int(start_step) else float(inactive_value)


def residual_score(enc_tokens: torch.Tensor, dec_tokens: torch.Tensor) -> torch.Tensor:
    enc_tokens = F.normalize(enc_tokens.detach(), dim=1)
    dec_tokens = F.normalize(dec_tokens, dim=1)
    return (1.0 - (enc_tokens * dec_tokens).sum(dim=1)).clamp_min(0.0)


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def robust_norm_map(values: torch.Tensor, low_q: float = 0.05, high_q: float = 0.95) -> torch.Tensor:
    values = values.float()
    flat = values.flatten(1)
    low = torch.quantile(flat, low_q, dim=1).view(-1, 1, 1, 1)
    high = torch.quantile(flat, high_q, dim=1).view(-1, 1, 1, 1)
    return ((values - low) / (high - low).clamp_min(1e-6)).clamp(0.0, 1.0)


def local_ring_mean(values: torch.Tensor, radius: int) -> torch.Tensor:
    kernel = max(3, radius * 2 + 1)
    padding = kernel // 2
    area = float(kernel * kernel)
    pooled_sum = F.avg_pool2d(values, kernel_size=kernel, stride=1, padding=padding) * area
    count = F.avg_pool2d(torch.ones_like(values[:, :1]), kernel_size=kernel, stride=1, padding=padding) * area
    if values.shape[1] != 1:
        count = count.expand(-1, values.shape[1], -1, -1)
    return (pooled_sum - values) / (count - 1.0).clamp_min(1.0)


def normalized_luminance(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    rgb = (images.float() * std + mean).clamp(0.0, 1.0)
    weights = images.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (rgb * weights).sum(dim=1, keepdim=True)


def token_low_level_maps(images: torch.Tensor, token_h: int, token_w: int) -> tuple[torch.Tensor, torch.Tensor]:
    lum = normalized_luminance(images)
    blur = F.avg_pool2d(lum, kernel_size=7, stride=1, padding=3)
    # Local high-pass energy is a stable spatial-domain proxy for high-frequency spectral energy.
    freq = (lum - blur).square()

    sobel_x = lum.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    sobel_y = lum.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    grad_x = F.conv2d(lum, sobel_x, padding=1)
    grad_y = F.conv2d(lum, sobel_y, padding=1)
    edge = torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)
    return (
        F.adaptive_avg_pool2d(freq, (token_h, token_w)),
        F.adaptive_avg_pool2d(edge, (token_h, token_w)),
    )


def dino_ring_contrast(feature: torch.Tensor, radius: int) -> torch.Tensor:
    feature = F.normalize(feature.detach().float(), dim=1)
    ring = F.normalize(local_ring_mean(feature, radius), dim=1)
    return (1.0 - (feature * ring).sum(dim=1, keepdim=True)).clamp_min(0.0)


def guidance_components(images: torch.Tensor, enc_feature: torch.Tensor, args: argparse.Namespace) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        _, _, token_h, token_w = enc_feature.shape
        freq, edge = token_low_level_maps(images, token_h, token_w)
        freq_contrast = (freq - local_ring_mean(freq, args.ring_radius)).abs()
        edge_contrast = (edge - local_ring_mean(edge, args.ring_radius)).abs()
        dino_contrast = dino_ring_contrast(enc_feature, args.ring_radius)
        return {
            "freq": robust_norm_map(freq_contrast),
            "edge": robust_norm_map(edge_contrast),
            "dino": robust_norm_map(dino_contrast),
        }


def combined_guidance_map(images: torch.Tensor, enc_feature: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    comps = guidance_components(images, enc_feature, args)
    total = max(
        float(args.objectness_freq_weight + args.objectness_edge_weight + args.objectness_dino_weight),
        1e-6,
    )
    raw = (
        args.objectness_freq_weight * comps["freq"]
        + args.objectness_edge_weight * comps["edge"]
        + args.objectness_dino_weight * comps["dino"]
    ) / total
    kernel = int(args.objectness_smooth_kernel)
    if kernel > 1:
        if kernel % 2 == 0:
            kernel += 1
        raw = F.avg_pool2d(raw, kernel_size=kernel, stride=1, padding=kernel // 2)
    return robust_norm_map(raw).detach()


def top_fraction_weights(weights: torch.Tensor, frac: float) -> torch.Tensor:
    frac = max(0.0, min(1.0, float(frac)))
    if frac <= 0.0:
        return torch.zeros_like(weights)
    if frac >= 1.0:
        return weights
    flat = weights.flatten(1)
    threshold = torch.quantile(flat, 1.0 - frac, dim=1).view(-1, 1, 1, 1)
    kept = torch.where(weights >= threshold, weights, torch.zeros_like(weights))
    return kept / kept.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)


def normal_texture_preservation_loss(
    images: torch.Tensor,
    enc_feature: torch.Tensor,
    dec_feature: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, Dict[str, float]]:
    texture = top_fraction_weights(
        combined_guidance_map(images, enc_feature.detach(), args),
        args.normal_texture_top_frac,
    )
    enc_norm = F.normalize(enc_feature.detach(), dim=1)
    dec_norm = F.normalize(dec_feature, dim=1)
    keep = weighted_mean((1.0 - (dec_norm * enc_norm).sum(dim=1, keepdim=True)).clamp_min(0.0), texture)

    ring_enc = F.normalize(local_ring_mean(enc_feature.detach(), args.ring_radius), dim=1)
    sim_enc = (dec_norm * enc_norm).sum(dim=1, keepdim=True)
    sim_ring = (dec_norm * ring_enc).sum(dim=1, keepdim=True)
    not_bg = weighted_mean(F.relu(sim_ring - sim_enc + args.normal_texture_margin).square(), texture)
    loss = keep + args.normal_texture_not_bg_weight * not_bg
    return loss, {
        "l_texture_keep": float(keep.detach().cpu()),
        "l_texture_not_bg": float(not_bg.detach().cpu()),
        "texture_weight_mean": float(texture.mean().detach().cpu()),
    }


def normal_prior_reconstruction_loss(
    images: torch.Tensor,
    enc_feature: torch.Tensor,
    dec_feature: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, Dict[str, float]]:
    comps = guidance_components(images, enc_feature.detach(), args)
    texture = robust_norm_map(0.5 * comps["freq"] + 0.5 * comps["edge"])
    objectness = comps["dino"]
    total = max(float(args.normal_prior_texture_weight + args.normal_prior_object_weight), 1e-6)
    prior = (
        args.normal_prior_texture_weight * texture
        + args.normal_prior_object_weight * objectness
    ) / total
    prior = robust_norm_map(prior).pow(max(float(args.normal_prior_power), 0.1))
    if args.normal_prior_top_frac > 0.0:
        prior = top_fraction_weights(prior, args.normal_prior_top_frac)
    enc_norm = F.normalize(enc_feature.detach(), dim=1)
    dec_norm = F.normalize(dec_feature, dim=1)
    residual = (1.0 - (enc_norm * dec_norm).sum(dim=1, keepdim=True)).clamp_min(0.0)
    loss = weighted_mean(residual, prior)
    return loss, {
        "l_normal_prior": float(loss.detach().cpu()),
        "normal_prior_mean": float(prior.mean().detach().cpu()),
        "normal_prior_texture_mean": float(texture.mean().detach().cpu()),
        "normal_prior_object_mean": float(objectness.mean().detach().cpu()),
    }


def _token_maps(tokens: List[torch.Tensor], batch: int, side: int) -> List[torch.Tensor]:
    return [item.permute(0, 2, 1).reshape([batch, -1, side, side]).contiguous() for item in tokens]


@contextmanager
def temporary_requires_grad(params: Iterable[torch.nn.Parameter], requires_grad: bool) -> Iterator[None]:
    params = list(params)
    old = [param.requires_grad for param in params]
    for param in params:
        param.requires_grad_(requires_grad)
    try:
        yield
    finally:
        for param, value in zip(params, old):
            param.requires_grad_(value)


def inpformer_prototype_params(model: nn.Module) -> List[torch.nn.Parameter]:
    params: List[torch.nn.Parameter] = []
    prototype_token = getattr(model, "prototype_token", None)
    if isinstance(prototype_token, torch.nn.Parameter):
        params.append(prototype_token)
    elif hasattr(prototype_token, "parameters"):
        params.extend(list(prototype_token.parameters()))
    aggregation = getattr(model, "aggregation", None)
    if hasattr(aggregation, "parameters"):
        params.extend(list(aggregation.parameters()))
    return params


def inpformer_aggregate_prototypes(
    model: nn.Module,
    target_tokens: torch.Tensor,
    images: torch.Tensor | None = None,
) -> torch.Tensor:
    """Aggregate image-conditioned prototypes for guided or native INP-Former.

    The native path deliberately uses only ``prototype_token`` and
    ``aggregation``.  This lets E1/E2 supervise the original INP-Former
    prototypes without constructing any guided-prototype modules.
    """

    if hasattr(model, "aggregate_guided_prototypes"):
        if images is None:
            raise ValueError("Guided prototype aggregation requires the source images.")
        return model.aggregate_guided_prototypes(target_tokens, images)

    prototype_token = getattr(model, "prototype_token", None)
    aggregation = getattr(model, "aggregation", None)
    if prototype_token is None or aggregation is None:
        raise ValueError("Native prototype aggregation requires prototype_token and aggregation.")
    current_prototype = prototype_token.unsqueeze(0).repeat((target_tokens.shape[0], 1, 1))
    for block in aggregation:
        current_prototype = block(current_prototype, target_tokens)
    return current_prototype


def inpformer_forward_with_agg_prototype(
    model: nn.Module,
    images: torch.Tensor,
    agg_prototype: torch.Tensor | None = None,
    return_agg_prototype: bool = False,
    return_target_tokens: bool = False,
    return_bottleneck_tokens: bool = False,
) -> tuple:
    """INP-Former forward with an optional detached prototype context.

    This mirrors INP-Former's native forward. When agg_prototype is supplied, the
    detector branch skips image-conditioned prototype aggregation and only trains
    bottleneck/decoder against that fixed normal context.
    """

    required = ("encoder", "target_layers", "fuse_feature", "bottleneck", "decoder")
    if not all(hasattr(model, name) for name in required):
        raise ValueError("--det-prototype-mode normal_context is only implemented for INP-Former-like models.")

    x = model.encoder.prepare_tokens(images)
    en_list: List[torch.Tensor] = []
    for i, blk in enumerate(model.encoder.blocks):
        if i <= model.target_layers[-1]:
            if i in model.encoder_require_grad_layer:
                x = blk(x)
            else:
                with torch.no_grad():
                    x = blk(x)
        else:
            continue
        if i in model.target_layers:
            en_list.append(x)
    if not en_list:
        raise RuntimeError("No INP-Former encoder features were captured.")

    original_start = 1 + model.encoder.num_register_tokens
    side = int(math.sqrt(en_list[0].shape[1] - original_start))
    batch = images.shape[0]
    if model.remove_class_token:
        en_list = [item[:, original_start:, :] for item in en_list]

    target_tokens = model.fuse_feature(en_list)
    if agg_prototype is None:
        current_prototype = inpformer_aggregate_prototypes(
            model,
            target_tokens,
            images=images,
        )
        g_loss = model.gather_loss(target_tokens, current_prototype)
    else:
        current_prototype = agg_prototype.detach()
        if current_prototype.shape[0] == 1 and batch != 1:
            current_prototype = current_prototype.expand(batch, -1, -1)
        elif current_prototype.shape[0] != batch:
            raise ValueError(
                f"Prototype batch mismatch: got {current_prototype.shape[0]} prototypes for batch={batch}."
            )
        g_loss = target_tokens.new_tensor(0.0)

    x = target_tokens
    for blk in model.bottleneck:
        x = blk(x)
    bottleneck_tokens = x

    de_list: List[torch.Tensor] = []
    for blk in model.decoder:
        x = blk(x, current_prototype)
        de_list.append(x)
    de_list = de_list[::-1]

    en = [model.fuse_feature([en_list[idx] for idx in idxs]) for idxs in model.fuse_layer_encoder]
    de = [model.fuse_feature([de_list[idx] for idx in idxs]) for idxs in model.fuse_layer_decoder]

    if not model.remove_class_token:
        en = [item[:, original_start:, :] for item in en]
        de = [item[:, original_start:, :] for item in de]

    out = (_token_maps(en, batch, side), _token_maps(de, batch, side), g_loss)
    if return_agg_prototype:
        out = (*out, current_prototype)
    if return_target_tokens:
        out = (*out, target_tokens)
    if return_bottleneck_tokens:
        out = (*out, bottleneck_tokens)
    return out


def paired_normal_prototype_context(
    normal_agg_prototype: torch.Tensor,
    target_batch: int,
) -> torch.Tensor:
    """Map each target composite to its pre-composition Normal context.

    ``compose_target_background_batch`` maps composite ``i`` to Normal sample
    ``i % normal_batch``. Reusing the same mapping here prevents pasted target
    tokens from entering image-conditioned prototype aggregation while keeping
    the clean scene context paired at sample level.
    """

    if normal_agg_prototype.ndim != 3 or normal_agg_prototype.shape[0] <= 0:
        raise ValueError(
            "normal_agg_prototype must have shape [normal_batch, prototypes, channels]"
        )
    if target_batch <= 0:
        raise ValueError("target_batch must be positive")
    indices = torch.arange(
        int(target_batch), device=normal_agg_prototype.device, dtype=torch.long
    ) % int(normal_agg_prototype.shape[0])
    return normal_agg_prototype.detach().index_select(0, indices)


def normal_fp_score_loss(enc_feature: torch.Tensor, dec_feature: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    score = (1.0 - F.cosine_similarity(enc_feature.detach(), dec_feature, dim=1, eps=1e-6)).clamp_min(0.0)
    flat = score.flatten(1)
    k = max(1, int(math.ceil(flat.shape[1] * max(0.0, min(1.0, args.normal_fp_topk_frac)))))
    top = torch.topk(flat, k=k, dim=1, largest=True).values
    return F.relu(top - args.normal_fp_margin).square().mean()


def native_background_excess_loss(
    adaptive_score: torch.Tensor,
    native_score: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    minimum_tokens: int = 8,
    target_dilate: int = 1,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Penalize only Adaptive-over-Native residuals outside pasted targets."""

    if adaptive_score.ndim != 4 or adaptive_score.shape[1] != 1:
        raise ValueError("Adaptive score must be [B,1,H,W].")
    if native_score.shape != adaptive_score.shape:
        raise ValueError("Native and Adaptive score maps must have identical shapes.")
    if target_mask.ndim != 4 or target_mask.shape[0] != adaptive_score.shape[0]:
        raise ValueError("Target mask must be [B,1,H,W] and match score batch.")
    if minimum_tokens <= 0:
        raise ValueError("Native background minimum tokens must be positive.")
    if target_dilate < 0:
        raise ValueError("Native background target dilation must be non-negative.")

    pooled_target = F.adaptive_max_pool2d(
        target_mask.float(), adaptive_score.shape[-2:]
    )
    if target_dilate > 0:
        kernel = 2 * int(target_dilate) + 1
        pooled_target = F.max_pool2d(
            pooled_target, kernel_size=kernel, stride=1, padding=target_dilate
        )
    target_tokens = pooled_target > 0.05
    background = ~target_tokens
    excess = F.relu(adaptive_score - native_score.detach())

    selected = []
    selected_tokens = 0
    background_tokens = 0
    for batch_index in range(int(adaptive_score.shape[0])):
        values = excess[batch_index][background[batch_index]]
        background_tokens += int(values.numel())
        if values.numel() == 0:
            continue
        target_count = int(target_tokens[batch_index].sum().item())
        keep = min(
            int(values.numel()),
            max(int(minimum_tokens), target_count),
        )
        selected.append(values.topk(keep).values)
        selected_tokens += keep
    if not selected:
        raise RuntimeError("Native background excess loss found no background tokens.")
    hard_excess = torch.cat(selected)
    loss = hard_excess.mean()
    active = excess[background] > 0.0
    diagnostics = {
        "l_native_bg_excess": float(loss.detach().cpu()),
        "native_bg_score_mean": float(
            native_score.detach()[background].mean().cpu()
        ),
        "adaptive_bg_score_mean": float(
            adaptive_score.detach()[background].mean().cpu()
        ),
        "native_bg_excess_mean": float(excess.detach()[background].mean().cpu()),
        "native_bg_excess_active_fraction": float(active.float().mean().cpu()),
        "native_bg_excess_selected_tokens": float(selected_tokens),
        "native_bg_background_tokens": float(background_tokens),
    }
    return loss, diagnostics


def object_erasing_loss_terms(
    enc_feature: torch.Tensor,
    dec_feature: torch.Tensor,
    boxes: torch.Tensor,
    valid: torch.Tensor,
    args: argparse.Namespace,
    objectness_weight: torch.Tensor | None = None,
    all_object_mask: torch.Tensor | None = None,
    score_map: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    _, _, token_h, token_w = enc_feature.shape
    obj_mask = boxes_to_token_mask(boxes, valid, token_h, token_w, args.image_size)
    if all_object_mask is None:
        all_obj_mask = obj_mask
    else:
        all_obj_mask = F.adaptive_max_pool2d(all_object_mask.float(), (token_h, token_w)) > 0
        dilate = max(0, int(args.det_clean_bg_dilate))
        if dilate > 0:
            all_obj_mask = F.max_pool2d(
                all_obj_mask.float(),
                kernel_size=2 * dilate + 1,
                stride=1,
                padding=dilate,
            ) > 0
    size_weight_map = boxes_to_size_weight_map(boxes, valid, token_h, token_w, args)
    ring_mask = ring_from_mask(obj_mask, args.ring_radius)
    if args.det_background_mode in {"none", "clean"}:
        ring_mask = ring_mask & ~all_obj_mask
    if objectness_weight is not None:
        objectness_weight = objectness_weight.detach().clamp(0.0, 1.0)
        if objectness_weight.shape[-2:] != (token_h, token_w):
            objectness_weight = F.interpolate(objectness_weight, size=(token_h, token_w), mode="bilinear", align_corners=False)
        objectness_weight = (
            args.det_objectness_min_weight
            + (1.0 - args.det_objectness_min_weight) * objectness_weight.pow(args.det_objectness_power)
        ).clamp(0.0, 1.0)
    if args.det_background_mode == "none" or args.lambda_det_bg <= 0.0:
        l_bg = enc_feature.new_tensor(0.0)
        bg_mask = torch.zeros_like(obj_mask)
    elif args.det_background_mode == "clean":
        bg_mask = ~all_obj_mask
        l_bg = masked_cosine_distance(dec_feature, enc_feature, bg_mask)
    else:
        bg_mask = ~obj_mask
        l_bg = masked_cosine_distance(dec_feature, enc_feature, bg_mask)
    obj_losses: List[torch.Tensor] = []
    sep_losses: List[torch.Tensor] = []
    smooth_losses: List[torch.Tensor] = []
    rank_losses: List[torch.Tensor] = []
    cover_losses: List[torch.Tensor] = []
    rank_bg_values: List[torch.Tensor] = []
    rank_obj_values: List[torch.Tensor] = []
    rank_violations: List[torch.Tensor] = []
    obj_weight_means: List[torch.Tensor] = []
    size_weight_means: List[torch.Tensor] = []
    core_weight_means: List[torch.Tensor] = []
    hard_residual_weight_means: List[torch.Tensor] = []
    hard_residual_score_means: List[torch.Tensor] = []
    boundary_losses: List[torch.Tensor] = []
    regions: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
    if args.det_instance_local_erasing:
        for b in range(enc_feature.shape[0]):
            for box_idx in torch.nonzero(valid[b], as_tuple=False).flatten().tolist():
                one_boxes = boxes[b : b + 1, box_idx : box_idx + 1]
                one_valid = valid[b : b + 1, box_idx : box_idx + 1]
                one_obj = boxes_to_token_mask(one_boxes, one_valid, token_h, token_w, args.image_size)[0, 0]
                one_ring = ring_from_mask(one_obj[None, None], args.ring_radius)[0, 0]
                if args.det_background_mode in {"none", "clean"}:
                    one_ring = one_ring & ~all_obj_mask[b, 0]
                regions.append((b, one_obj, one_ring))
    else:
        regions = [(b, obj_mask[b, 0], ring_mask[b, 0]) for b in range(enc_feature.shape[0])]

    skipped_regions = 0
    for b, obj, ring in regions:
        if int(obj.sum()) <= 0 or int(ring.sum()) <= 0:
            skipped_regions += 1
            continue
        obj_dec = F.normalize(dec_feature[b, :, obj].T, dim=1)
        obj_enc = F.normalize(enc_feature[b, :, obj].detach().T, dim=1)
        ring_dec_tokens = dec_feature[b, :, ring].T
        ring_enc_tokens = enc_feature[b, :, ring].T
        ring_enc = F.normalize(enc_feature[b, :, ring].mean(dim=1), dim=0)
        ring_dec = F.normalize(dec_feature[b, :, ring].detach().mean(dim=1), dim=0)
        size_token_weight = size_weight_map[b, 0, obj].to(dtype=obj_dec.dtype)
        size_weight_means.append(size_token_weight.mean())
        if objectness_weight is None:
            token_weight = size_token_weight
        else:
            token_weight = objectness_weight[b, 0, obj].to(dtype=obj_dec.dtype) * size_token_weight
            obj_weight_means.append(token_weight.mean())
        erase_weight = token_weight
        boundary_weight = None
        if args.det_core_ring_erasing and token_weight.numel() > 1:
            core_frac = max(0.0, min(1.0, float(args.det_core_frac)))
            keep = max(1, min(token_weight.numel(), int(math.ceil(token_weight.numel() * core_frac))))
            thresh = torch.topk(token_weight.detach().reshape(-1), k=keep, largest=True).values[-1]
            core_mask = token_weight.detach() >= thresh
            erase_weight = torch.where(core_mask, token_weight, torch.zeros_like(token_weight))
            boundary_weight = torch.where(core_mask, torch.zeros_like(token_weight), token_weight)
        if getattr(args, "det_hard_residual_mining", False) and erase_weight.numel() > 1:
            residual = (1.0 - (obj_dec * obj_enc).sum(dim=1)).clamp_min(0.0).detach()
            fraction = float(getattr(args, "det_hard_residual_frac", 0.50))
            keep = max(1, min(residual.numel(), int(math.ceil(residual.numel() * fraction))))
            threshold = torch.topk(residual, k=keep, largest=False).values[-1]
            selected = residual <= threshold
            minimum = float(getattr(args, "det_hard_residual_min_weight", 0.10))
            hard_weight = minimum + (1.0 - minimum) * selected.to(residual.dtype)
            erase_weight = erase_weight * hard_weight.to(erase_weight)
            hard_residual_weight_means.append(hard_weight.mean())
            hard_residual_score_means.append(residual.mean())
        core_weight_means.append(erase_weight.mean())
        obj_losses.append(weighted_mean((1.0 - (obj_dec * ring_enc.unsqueeze(0)).sum(dim=1)).clamp_min(0.0), erase_weight))
        sep_losses.append(weighted_mean(F.relu((obj_dec * obj_enc).sum(dim=1) - args.sep_sim_margin).square(), erase_weight))
        smooth_losses.append(weighted_mean((1.0 - (obj_dec * ring_dec.unsqueeze(0)).sum(dim=1)).clamp_min(0.0), erase_weight))
        if args.lambda_boundary_preserve > 0.0 and boundary_weight is not None and float(boundary_weight.sum().detach().cpu()) > 0.0:
            boundary_losses.append(weighted_mean((1.0 - (obj_dec * obj_enc).sum(dim=1)).clamp_min(0.0), boundary_weight))
        if args.lambda_score_rank > 0.0 or args.lambda_score_cover > 0.0:
            rank_mode = getattr(args, "score_rank_background_mode", "local_ring")
            if rank_mode == "global_clean":
                if score_map is None:
                    raise ValueError("global_clean score rank requires an inference-aligned score_map.")
                current_score_map = score_map[b, 0]
                if current_score_map.shape != obj.shape:
                    current_score_map = F.interpolate(
                        current_score_map[None, None],
                        size=obj.shape,
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
                bg_scores = current_score_map[bg_mask[b, 0]].detach()
                obj_scores = current_score_map[obj]
                core_selector = erase_weight.detach() > 0
                if int(core_selector.sum()) > 0:
                    obj_scores = obj_scores[core_selector]
                if bg_scores.numel() <= 0 or obj_scores.numel() <= 0:
                    skipped_regions += 1
                    continue
                bg_frac = max(
                    1.0 / float(bg_scores.numel()),
                    min(1.0, float(getattr(args, "score_rank_global_topk_frac", 0.01))),
                )
                obj_frac = max(
                    1.0 / float(obj_scores.numel()),
                    min(1.0, float(getattr(args, "score_rank_object_low_frac", 0.25))),
                )
                bg_keep = max(1, int(math.ceil(bg_scores.numel() * bg_frac)))
                obj_keep = max(1, int(math.ceil(obj_scores.numel() * obj_frac)))
                rank_bg = torch.topk(bg_scores, k=bg_keep, largest=True).values.mean()
                rank_obj = torch.topk(obj_scores, k=obj_keep, largest=False).values.mean()
                cover_bg = rank_bg
                cover_obj = rank_obj
            else:
                obj_score = residual_score(
                    enc_feature[b, :, obj].detach().T,
                    dec_feature[b, :, obj].T,
                )
                ring_score = residual_score(ring_enc_tokens, ring_dec_tokens).detach()
                rank_bg = quantile_loss_value(ring_score, 0.90)
                cover_bg = quantile_loss_value(ring_score, 0.75)
                rank_obj = quantile_loss_value(obj_score, 0.50)
                cover_obj = quantile_loss_value(obj_score, 0.25)
            rank_violation = args.score_rank_margin + rank_bg - rank_obj
            cover_violation = args.score_cover_margin + cover_bg - cover_obj
            rank_losses.append(F.relu(rank_violation).square())
            cover_losses.append(F.relu(cover_violation).square())
            rank_bg_values.append(rank_bg.detach())
            rank_obj_values.append(rank_obj.detach())
            rank_violations.append((rank_violation.detach() > 0).to(torch.float32))
    zero = enc_feature.new_tensor(0.0)
    l_obj_bg = torch.stack(obj_losses).mean() if obj_losses else zero
    l_obj_sep = torch.stack(sep_losses).mean() if sep_losses else zero
    l_smooth = torch.stack(smooth_losses).mean() if smooth_losses else zero
    l_score_rank = torch.stack(rank_losses).mean() if rank_losses else zero
    l_score_cover = torch.stack(cover_losses).mean() if cover_losses else zero
    l_boundary_preserve = torch.stack(boundary_losses).mean() if boundary_losses else zero
    obj_weight_mean = torch.stack(obj_weight_means).mean() if obj_weight_means else zero
    size_weight_mean = torch.stack(size_weight_means).mean() if size_weight_means else zero
    core_weight_mean = torch.stack(core_weight_means).mean() if core_weight_means else zero
    hard_residual_weight_mean = (
        torch.stack(hard_residual_weight_means).mean() if hard_residual_weight_means else zero
    )
    hard_residual_score_mean = (
        torch.stack(hard_residual_score_means).mean() if hard_residual_score_means else zero
    )
    legacy_loss = (
        args.lambda_det_bg * l_bg
        + args.lambda_obj_bg * l_obj_bg
        + args.lambda_obj_sep * l_obj_sep
        + args.lambda_smooth * l_smooth
        + args.lambda_boundary_preserve * l_boundary_preserve
    )
    rank_loss = args.lambda_score_rank * l_score_rank
    cover_loss = args.lambda_score_cover * l_score_cover
    rank_bg_mean = torch.stack(rank_bg_values).mean() if rank_bg_values else zero
    rank_obj_mean = torch.stack(rank_obj_values).mean() if rank_obj_values else zero
    rank_active_fraction = torch.stack(rank_violations).mean() if rank_violations else zero
    return legacy_loss, rank_loss, cover_loss, {
        "l_det_bg": float(l_bg.detach().cpu()),
        "l_obj_bg": float(l_obj_bg.detach().cpu()),
        "l_obj_sep": float(l_obj_sep.detach().cpu()),
        "l_obj_smooth": float(l_smooth.detach().cpu()),
        "l_score_rank": float(l_score_rank.detach().cpu()),
        "l_score_cover": float(l_score_cover.detach().cpu()),
        "score_rank_bg_mean": float(rank_bg_mean.detach().cpu()),
        "score_rank_obj_mean": float(rank_obj_mean.detach().cpu()),
        "score_rank_gap_mean": float((rank_obj_mean - rank_bg_mean).detach().cpu()),
        "score_rank_active_fraction": float(rank_active_fraction.detach().cpu()),
        "score_rank_regions": float(len(rank_losses)),
        "l_boundary_preserve": float(l_boundary_preserve.detach().cpu()),
        "obj_mask_frac": float(obj_mask.float().mean().detach().cpu()),
        "all_obj_mask_frac": float(all_obj_mask.float().mean().detach().cpu()),
        "det_bg_mask_frac": float(bg_mask.float().mean().detach().cpu()),
        "obj_weight_mean": float(obj_weight_mean.detach().cpu()),
        "size_weight_mean": float(size_weight_mean.detach().cpu()),
        "core_weight_mean": float(core_weight_mean.detach().cpu()),
        "hard_residual_weight_mean": float(hard_residual_weight_mean.detach().cpu()),
        "hard_residual_score_mean": float(hard_residual_score_mean.detach().cpu()),
        "det_regions": float(len(regions)),
        "det_regions_skipped": float(skipped_regions),
    }


def object_erasing_losses(
    enc_feature: torch.Tensor,
    dec_feature: torch.Tensor,
    boxes: torch.Tensor,
    valid: torch.Tensor,
    args: argparse.Namespace,
    objectness_weight: torch.Tensor | None = None,
    all_object_mask: torch.Tensor | None = None,
    score_map: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    legacy_loss, rank_loss, cover_loss, diag = object_erasing_loss_terms(
        enc_feature,
        dec_feature,
        boxes,
        valid,
        args,
        objectness_weight=objectness_weight,
        all_object_mask=all_object_mask,
        score_map=score_map,
    )
    return legacy_loss + rank_loss + cover_loss, diag


def write_history(rows: List[Dict[str, float]], output_dir: Path) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with (output_dir / "train_history.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})
    (output_dir / "train_history.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def write_diagnostic_rows(rows: List[Dict[str, float]], csv_path: Path) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    csv_path.with_suffix(".json").write_text(
        json.dumps(rows, indent=2, allow_nan=True), encoding="utf-8"
    )


def write_config(args: argparse.Namespace, total_steps: int, normal_crops: int, steps_per_epoch: int, det_images: int) -> None:
    payload = {
        key: ([str(item) for item in value] if isinstance(value, list) and value and isinstance(value[0], Path) else str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    normal_term = f"{args.normal_loss}_masked_normal_reconstruction" if args.masked_recon else f"{args.normal_loss}_normal_reconstruction"
    loss_terms = [normal_term, "object_erasing_auxiliary"]
    if args.masked_recon and args.full_anchor_loss_weight > 0.0:
        loss_terms.append("full_anchor_normal_reconstruction")
    if args.det_prototype_mode != "joint":
        loss_terms.append(f"det_prototype_mode={args.det_prototype_mode}")
    loss_terms.append(f"det_background_mode={args.det_background_mode}")
    if args.guided_prototype and args.guided_prototype_trainable_prior:
        loss_terms.append(f"guided_prior_head={args.guided_prototype_prior_head_mode}")
    if args.target_gate_loss_weight > 0.0:
        loss_terms.append("target_supervised_familiarity_gate_calibration")
    if args.target_proto_invariance_weight > 0.0:
        loss_terms.append("paired_target_prototype_invariance")
    if args.target_proto_repulsion_weight > 0.0:
        loss_terms.append("target_core_prototype_repulsion")
    if args.target_aggregation_attention_weight > 0.0:
        loss_terms.append("target_excluded_aggregation_attention")
    if args.target_read_attention_weight > 0.0:
        loss_terms.append("target_suppressed_decoder_read_attention")
    if args.normal_texture_preserve_weight > 0.0:
        loss_terms.append("normal_texture_preservation")
    if args.normal_prior_recon_weight > 0.0:
        loss_terms.append("normal_prior_weighted_reconstruction")
    if args.hn_oe_loss_weight > 0.0:
        loss_terms.append(f"hn_oe_causal_restore[{args.hn_oe_gradient_scope}]")
    if args.native_bg_excess_weight > 0.0:
        loss_terms.append("native_teacher_background_excess[bottleneck_decoder]")
    if args.det_objectness_weighting:
        loss_terms.append("abstract_objectness_weighted_det")
    if args.det_scale_resample:
        loss_terms.append("det_scale_resampled")
    if args.det_crop_augment:
        loss_terms.append("det_object_center_crop")
    if args.det_size_weighting:
        loss_terms.append("det_size_weighted")
    if args.det_core_ring_erasing:
        loss_terms.append("core_ring_erasing")
    if args.det_hard_residual_mining:
        loss_terms.append("hard_residual_mining")
    if args.det_instance_local_erasing:
        loss_terms.append("instance_local_erasing")
    if args.lambda_score_rank > 0.0:
        loss_terms.append("dense_score_rank")
    if args.lambda_score_cover > 0.0:
        loss_terms.append("dense_score_cover")
    if args.lambda_normal_fp > 0.0:
        loss_terms.append("normal_topk_fp_suppression")
    payload.update(
        {
            "resolved_total_steps": total_steps,
            "normal_crops": normal_crops,
            "steps_per_epoch": steps_per_epoch,
            "det_images": det_images,
            "loss": " + ".join(loss_terms),
        }
    )
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    if args.det_manifests is None:
        if args.visdrone_manifest is None:
            raise ValueError("Provide --det-manifests or the legacy --visdrone-manifest.")
        args.det_manifests = [args.visdrone_manifest]
    det_source_weights = parse_float_list(args.det_source_weights) if args.det_source_weights else ()
    det_crop_source_indices = tuple(
        int(part) for part in args.det_crop_source_indices.replace(",", " ").split() if part.strip()
    )
    if args.normal_loss_step_offset < 0:
        raise ValueError("--normal-loss-step-offset must be non-negative.")
    if args.normal_loss_weight < 0.0:
        raise ValueError("--normal-loss-weight must be non-negative.")
    if args.target_gate_loss_weight < 0.0:
        raise ValueError("--target-gate-loss-weight must be non-negative.")
    if args.target_gate_normal_anchor_weight < 0.0:
        raise ValueError("--target-gate-normal-anchor-weight must be non-negative.")
    if args.target_gate_lr < 0.0:
        raise ValueError("--target-gate-lr must be non-negative.")
    if args.target_gate_warmup_steps < 0:
        raise ValueError("--target-gate-warmup-steps must be non-negative.")
    if args.target_gate_loss_weight > 0.0:
        if not (
            args.guided_prototype_target_gate_calibration
            or args.guided_prototype_target_reject_gate
            or args.guided_prototype_target_mode_gate
        ):
            raise ValueError(
                "Target-gate loss requires scalar, reject, or per-mode calibration."
            )
        if not args.det_target_background_compose:
            raise ValueError(
                "Target-gate loss requires --det-target-background-compose."
            )
    target_aux_weights = (
        args.target_proto_invariance_weight,
        args.target_proto_repulsion_weight,
        args.target_aggregation_attention_weight,
        args.target_read_attention_weight,
    )
    if any(weight < 0.0 for weight in target_aux_weights):
        raise ValueError("Target prototype/attention loss weights must be non-negative.")
    if any(weight > 0.0 for weight in target_aux_weights):
        if args.architecture != "inpformer":
            raise ValueError("Target prototype/attention losses require INP-Former.")
        if not args.det_target_background_compose:
            raise ValueError("Target prototype/attention losses require target-background composition.")
        if args.det_prototype_mode != "freeze":
            raise ValueError("Controlled target prototype ablations require det_prototype_mode=freeze.")
    if args.target_aggregation_attention_weight > 0.0 and not args.guided_prototype:
        raise ValueError("Target aggregation-attention loss currently requires guided INP-Former.")
    if not 0.0 < args.target_proto_repulsion_normal_quantile <= 1.0:
        raise ValueError("Target prototype Normal quantile must lie in (0,1].")
    if args.target_proto_repulsion_margin_delta < 0.0:
        raise ValueError("Target prototype margin delta must be non-negative.")
    if not 0.0 <= args.target_proto_core_min_occupancy <= 1.0:
        raise ValueError("Target prototype core occupancy must lie in [0,1].")
    if args.target_proto_repulsion_min_normal_tokens_per_mode <= 0:
        raise ValueError("Target prototype per-mode support must be positive.")
    if (
        args.target_proto_repulsion_margin_scope == "effective_mode"
        and not args.guided_prototype_center6_balanced
    ):
        raise ValueError(
            "Effective-mode target prototype margins require an adaptive Center teacher."
        )
    if (
        args.target_proto_repulsion_mode_budget != "none"
        and args.target_proto_repulsion_margin_scope != "effective_mode"
    ):
        raise ValueError(
            "Target prototype mode budgeting requires effective-mode margin scope."
        )
    if not 0.0 <= args.target_read_attention_ratio <= 1.0:
        raise ValueError("Target read attention ratio must lie in [0,1].")
    if args.target_read_background_anchor_weight < 0.0:
        raise ValueError("Target read background anchor weight must be non-negative.")
    if not 0.0 <= args.target_aggregation_attention_ratio <= 1.0:
        raise ValueError("Target aggregation attention ratio must lie in [0,1].")
    if args.target_aggregation_background_anchor_weight < 0.0:
        raise ValueError("Target aggregation background anchor weight must be non-negative.")
    if args.hn_oe_loss_weight < 0.0:
        raise ValueError("--hn-oe-loss-weight must be non-negative.")
    if args.hn_oe_margin < 0.0:
        raise ValueError("--hn-oe-margin must be non-negative.")
    if args.hn_oe_start_step <= 0:
        raise ValueError("--hn-oe-start-step must be positive.")
    if args.hn_oe_loss_weight > 0.0:
        if args.hn_oe_sidecar is None or args.hn_oe_normal_crops_csv is None:
            raise ValueError(
                "HN-OE replay requires --hn-oe-sidecar and --hn-oe-normal-crops-csv."
            )
        if args.normal_root is not None:
            raise ValueError("HN-OE crop-coordinate replay requires the FOD crop dataset.")
        if args.hn_oe_gradient_scope == "decoder" and not args.sequential_branch_backward:
            raise ValueError(
                "Decoder-only HN-OE gradients require --sequential-branch-backward."
            )
    if args.native_bg_excess_weight < 0.0:
        raise ValueError("--native-bg-excess-weight must be non-negative.")
    if args.native_bg_excess_min_tokens <= 0:
        raise ValueError("--native-bg-excess-min-tokens must be positive.")
    if args.native_bg_excess_target_dilate < 0:
        raise ValueError("--native-bg-excess-target-dilate must be non-negative.")
    if args.native_bg_excess_weight > 0.0:
        if args.native_bg_teacher_checkpoint is None:
            raise ValueError(
                "--native-bg-excess-weight requires --native-bg-teacher-checkpoint."
            )
        if args.architecture != "inpformer":
            raise ValueError("Native background teaching currently requires INP-Former.")
        if args.det_prototype_mode != "freeze":
            raise ValueError(
                "Native background teaching requires --det-prototype-mode freeze."
            )
        if not args.det_target_background_compose:
            raise ValueError(
                "Native background teaching requires --det-target-background-compose."
            )
        if args.rank_gradient_audit_only:
            raise ValueError(
                "Native background teaching cannot be combined with rank-gradient audit."
            )
    if not 0.0 < args.det_hard_residual_frac <= 1.0:
        raise ValueError("--det-hard-residual-frac must be in (0,1].")
    if not 0.0 <= args.det_hard_residual_min_weight <= 1.0:
        raise ValueError("--det-hard-residual-min-weight must be in [0,1].")
    if args.lr <= 0.0:
        raise ValueError("--lr must be positive.")
    if args.min_lr < 0.0 or args.min_lr > args.lr:
        raise ValueError("--min-lr must be in [0, --lr].")
    if args.lr_warmup_steps < 0:
        raise ValueError("--lr-warmup-steps must be non-negative.")
    if args.lr_decay_start_step < 0:
        raise ValueError("--lr-decay-start-step must be non-negative.")
    if args.lr_decay_end_step < 0:
        raise ValueError("--lr-decay-end-step must be non-negative.")
    if 0 < args.lr_decay_end_step <= args.lr_decay_start_step:
        raise ValueError("--lr-decay-end-step must exceed --lr-decay-start-step.")
    if not 0.0 <= args.lr_warmup_start_factor <= 1.0:
        raise ValueError("--lr-warmup-start-factor must be in [0,1].")
    if args.gradient_conflict_probe and not args.sequential_branch_backward:
        raise ValueError(
            "--gradient-conflict-probe requires --sequential-branch-backward."
        )
    if args.gradient_conflict_probe_every <= 0:
        raise ValueError("--gradient-conflict-probe-every must be positive.")
    if args.gradient_conflict_component_control and not args.sequential_branch_backward:
        raise ValueError(
            "--gradient-conflict-component-control requires "
            "--sequential-branch-backward."
        )
    if (
        args.gradient_conflict_component_scale < 0.0
        or not math.isfinite(args.gradient_conflict_component_scale)
    ):
        raise ValueError(
            "--gradient-conflict-component-scale must be finite and non-negative."
        )
    if args.gradient_conflict_component_budget_beta is not None and (
        args.gradient_conflict_component_budget_beta < 0.0
        or args.gradient_conflict_component_budget_beta > 1.0
        or not math.isfinite(args.gradient_conflict_component_budget_beta)
    ):
        raise ValueError(
            "--gradient-conflict-component-budget-beta must be finite and in [0,1]."
        )
    if args.gradient_conflict_component_start_step <= 0:
        raise ValueError("--gradient-conflict-component-start-step must be positive.")
    if args.score_rank_start_step <= 0:
        raise ValueError("--score-rank-start-step must be positive.")
    if args.score_cover_start_step <= 0:
        raise ValueError("--score-cover-start-step must be positive.")
    if args.score_rank_separate_backward and not args.sequential_branch_backward:
        raise ValueError("--score-rank-separate-backward requires --sequential-branch-backward.")
    if args.score_rank_separate_backward and args.lambda_score_rank <= 0.0:
        raise ValueError("--score-rank-separate-backward requires --lambda-score-rank > 0.")
    if args.gradient_conflict_component_anchor == "normal_rank":
        if not args.gradient_conflict_component_control:
            raise ValueError(
                "--gradient-conflict-component-anchor normal_rank requires "
                "--gradient-conflict-component-control."
            )
        if not args.score_rank_separate_backward:
            raise ValueError(
                "--gradient-conflict-component-anchor normal_rank requires "
                "--score-rank-separate-backward."
            )
    if not 0.0 < args.score_rank_global_topk_frac <= 1.0:
        raise ValueError("--score-rank-global-topk-frac must be in (0,1].")
    if not 0.0 < args.score_rank_object_low_frac <= 1.0:
        raise ValueError("--score-rank-object-low-frac must be in (0,1].")
    if args.score_rank_gradient_probe_every <= 0:
        raise ValueError("--score-rank-gradient-probe-every must be positive.")
    if args.rank_gradient_audit_only:
        if not args.sequential_branch_backward:
            raise ValueError("--rank-gradient-audit-only requires --sequential-branch-backward.")
        if args.lambda_score_rank <= 0.0:
            raise ValueError("--rank-gradient-audit-only requires --lambda-score-rank > 0.")
        if args.rank_gradient_audit_steps <= 0:
            raise ValueError("--rank-gradient-audit-steps must be positive.")
        if args.inline_fod_val_every_epochs > 0:
            raise ValueError("Rank-gradient audit must not enable inline validation.")
    if (
        args.rank_gradient_audit_target_ratio <= 0.0
        or not math.isfinite(args.rank_gradient_audit_target_ratio)
    ):
        raise ValueError("--rank-gradient-audit-target-ratio must be finite and positive.")
    setup_seed(args.seed)
    if not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"
    if args.normal_root is None:
        args.train_distance = normalize_distance(args.train_distance)
    if args.roi_mask_erode_pixels < 0:
        raise ValueError("--roi-mask-erode-pixels must be non-negative.")
    if args.roi_mask_loss and (
        args.normal_root is not None or args.ground_roi_mask is None
    ):
        raise ValueError(
            "--roi-mask-loss requires FOD crop Normal data and --ground-roi-mask."
        )
    if args.roi_mask_loss and args.architecture == "inpformer":
        if args.guided_prototype_distill_weight > 0.0:
            raise ValueError(
                "Guided distillation is not part of the ROI-masked native-coherence "
                "path; set --guided-prototype-distill-weight 0."
            )
        if args.guided_prototype_semantic_coverage_weight > 0.0:
            raise ValueError(
                "Semantic prototype coverage is not part of the ROI-masked "
                "native-coherence path; set its weight to 0."
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mask_segments <= 0:
        raise ValueError("--mask-segments must be positive.")
    if args.inp_coherence_loss != "hard" and args.architecture != "inpformer":
        raise ValueError("--inp-coherence-loss is only available for INP-Former.")
    if args.inp_coherence_loss == "soft" and getattr(args, "guided_prototype", False):
        raise ValueError("Soft INP coherence and Guided Prototype define different prototype objectives; enable only one.")
    if args.normal_loss == "inp_soft_mining":
        if args.architecture != "inpformer":
            raise ValueError("--normal-loss inp_soft_mining is only available for INP-Former.")
        if args.masked_recon:
            raise ValueError("INP-Former++ soft mining currently requires full-token normal training.")

    if args.normal_root is not None:
        normal_dataset = NormalFolderDataset(args.normal_root, image_size=args.image_size)
    else:
        normal_crop_size = args.normal_crop_size if args.normal_crop_size > 0 else args.image_size
        normal_dataset = FODCropTrainDataset(
            args.fod_root,
            distance=args.train_distance,
            full_height=args.patch_full_height,
            crop_size=normal_crop_size,
            output_size=args.image_size,
            stride=args.patch_stride,
            layout=args.layout,
            roi_mask_path=args.ground_roi_mask,
            min_roi_coverage=args.ground_roi_min_coverage,
            return_roi_mask=args.roi_mask_loss,
            roi_erode_pixels=args.roi_mask_erode_pixels,
        )
    hn_replay = (
        HardNormalReplay.load(
            args.hn_oe_sidecar,
            args.hn_oe_normal_crops_csv,
            selector="oe_raised",
        )
        if args.hn_oe_loss_weight > 0.0
        else None
    )
    if hn_replay is not None:
        print(
            "[hn-oe] "
            f"tokens={hn_replay.unique_tokens} crop_occurrences={hn_replay.crop_occurrences} "
            f"weight={args.hn_oe_loss_weight:g} margin={args.hn_oe_margin:g} "
            f"scope={args.hn_oe_gradient_scope} start_step={args.hn_oe_start_step}",
            flush=True,
        )
    det_dataset = SmallObjectManifestDataset(
        args.det_manifests,
        image_size=args.image_size,
        max_images=args.max_det_images,
        max_boxes=args.max_boxes_per_image,
        seed=args.seed,
        source_weights=det_source_weights,
        scale_resample=args.det_scale_resample,
        scale_bins=args.det_scale_bins,
        crop_augment=args.det_crop_augment,
        crop_source_indices=det_crop_source_indices,
        crop_prob=args.det_crop_prob,
        crop_target_min_side=args.det_crop_target_min_side,
        crop_target_max_side=args.det_crop_target_max_side,
        crop_target_bins=args.det_crop_target_bins,
        crop_max_box_side=args.det_crop_max_box_side,
        square_crop=args.det_square_crop,
        preserve_row_order=args.det_no_shuffle,
    )
    steps_per_epoch = int(math.ceil(len(normal_dataset) / float(args.batch_size)))
    total_steps = int(args.total_steps) if args.total_steps > 0 else int(args.epochs) * steps_per_epoch
    if args.stop_after_step < 0 or args.stop_after_step > total_steps:
        raise ValueError("--stop-after-step must be zero or lie within the resolved total steps.")
    if args.lr_decay_start_step > total_steps:
        raise ValueError("--lr-decay-start-step cannot exceed the resolved total steps.")
    if args.lr_decay_end_step > total_steps:
        raise ValueError("--lr-decay-end-step cannot exceed the resolved total steps.")
    if args.gradient_conflict_component_start_step > total_steps:
        raise ValueError(
            "--gradient-conflict-component-start-step cannot exceed the resolved total steps."
        )
    if args.score_rank_start_step > total_steps:
        raise ValueError("--score-rank-start-step cannot exceed the resolved total steps.")
    if args.score_cover_start_step > total_steps:
        raise ValueError("--score-cover-start-step cannot exceed the resolved total steps.")
    if args.det_target_background_max_pastes <= 0:
        raise ValueError("--det-target-background-max-pastes must be positive.")
    if args.det_target_background_feather < 0:
        raise ValueError("--det-target-background-feather must be non-negative.")
    if args.inline_fod_val_every_epochs < 0:
        raise ValueError("--inline-fod-val-every-epochs must be non-negative.")
    if args.inline_fod_val_every_epochs > 0 and total_steps % steps_per_epoch != 0:
        raise ValueError(
            "Inline validation requires --total-steps to end on a complete normal epoch."
        )
    if args.inline_fod_val_start_epoch < 0:
        raise ValueError("--inline-fod-val-start-epoch must be non-negative.")
    if not 0.0 <= args.inline_fod_val_iou_threshold <= 1.0:
        raise ValueError("--inline-fod-val-iou-threshold must be in [0,1].")
    # InlineFODValidator consumes the common reconstruction-training names.
    args.dataset = "fod"
    args.crop_size = int(args.normal_crop_size if args.normal_crop_size > 0 else args.image_size)
    args.patch_output_size = int(args.image_size)
    args.inp_local_mask_recon = False
    if args.inline_fod_val_output_dir is None:
        args.inline_fod_val_output_dir = args.output_dir
    write_config(args, total_steps, len(normal_dataset), steps_per_epoch, len(det_dataset))

    normal_loader = DataLoader(
        normal_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    det_loader = DataLoader(
        det_dataset,
        batch_size=args.det_batch_size,
        shuffle=not args.det_no_shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    print(
        f"[config] architecture={args.architecture} normal_crops={len(normal_dataset)} "
        f"steps_per_epoch={steps_per_epoch} total_steps={total_steps} det_images={len(det_dataset)} "
        f"det_sources={dict(det_dataset.source_counts)} device={device}",
        flush=True,
    )

    model, trainable = build_reconstruction_model(
        architecture=args.architecture,
        dinomaly_repo=args.external_repo,
        inpformer_repo=args.inpformer_repo,
        encoder=args.encoder,
        device=device,
        inp_num=args.inp_num,
    )
    if args.architecture == "inpformer":
        configure_inp_coherence(model, args.inp_coherence_loss)
    configure_guided_prototypes(model, args, args.architecture)
    if args.checkpoint is not None:
        load_checkpoint(
            model,
            args.checkpoint,
            device,
            strict=args.strict_load,
            allowed_missing_prefixes=(
                tuple(
                    prefix
                    for enabled, prefix in (
                        (
                            args.guided_prototype_target_gate_calibration,
                            "guided_target_gate_calibrator.",
                        ),
                        (
                            args.guided_prototype_target_reject_gate,
                            "guided_target_reject_gate.",
                        ),
                        (
                            args.guided_prototype_target_mode_gate,
                            "guided_target_mode_gate.",
                        ),
                    )
                    if enabled
                )
            ),
        )
        print(f"[load] checkpoint={args.checkpoint}", flush=True)
    native_bg_teacher = None
    if args.native_bg_excess_weight > 0.0:
        assert args.native_bg_teacher_checkpoint is not None
        native_bg_teacher, _ = build_reconstruction_model(
            architecture=args.architecture,
            dinomaly_repo=args.external_repo,
            inpformer_repo=args.inpformer_repo,
            encoder=args.encoder,
            device=device,
            inp_num=args.inp_num,
        )
        load_checkpoint(
            native_bg_teacher,
            args.native_bg_teacher_checkpoint,
            device,
            strict=True,
        )
        native_bg_teacher.eval()
        for parameter in native_bg_teacher.parameters():
            parameter.requires_grad_(False)
        print(
            "[native-bg-teacher] "
            f"checkpoint={args.native_bg_teacher_checkpoint} "
            f"weight={args.native_bg_excess_weight:g} "
            f"min_tokens={args.native_bg_excess_min_tokens} "
            f"target_dilate={args.native_bg_excess_target_dilate}",
            flush=True,
        )
    guided_modules = guided_prototype_trainable_modules(model)
    for module in guided_modules:
        is_prior_head = module is getattr(model, "guided_prior_head", None)
        if is_prior_head and args.guided_prototype_prior_head_mode == "freeze":
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)
        else:
            trainable.append(module)
    model.train()
    if hasattr(model, "encoder"):
        model.encoder.eval()
    mask_config = PerspectiveMaskConfig(
        strategy=args.mask_strategy,
        band_block_sizes=parse_block_sizes(args.mask_band_block_sizes),
        fill=args.mask_fill,
        prototype_source=args.mask_prototype_source,
    )
    mask_band_loss_weights = parse_float_list(args.mask_band_loss_weights) if args.mask_band_loss_weights else None

    target_gate_module = getattr(model, "guided_target_mode_gate", None)
    if not isinstance(target_gate_module, nn.Module):
        target_gate_module = getattr(model, "guided_target_reject_gate", None)
    if not isinstance(target_gate_module, nn.Module):
        target_gate_module = getattr(model, "guided_target_gate_calibrator", None)
    target_gate_params = (
        list(target_gate_module.parameters())
        if isinstance(target_gate_module, nn.Module)
        else []
    )
    target_gate_param_ids = {id(parameter) for parameter in target_gate_params}
    base_params = [
        parameter
        for parameter in trainable.parameters()
        if id(parameter) not in target_gate_param_ids
    ]
    optimizer_groups = [{"params": base_params, "lr_scale": 1.0}]
    if target_gate_params:
        gate_lr = float(args.target_gate_lr or args.lr)
        optimizer_groups.append(
            {
                "params": target_gate_params,
                "lr": gate_lr,
                "lr_scale": gate_lr / float(args.lr),
                "weight_decay": 0.0,
            }
        )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history: List[Dict[str, float]] = []
    epoch_losses: List[float] = []
    best_epoch_loss = float("inf")
    start_step = 0
    resume_path = resolve_resume_path(args)
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        state = torch.load(resume_path, map_location=device)
        model_state = state["model"] if isinstance(state, dict) and "model" in state else state
        model.load_state_dict(model_state, strict=True)
        if not isinstance(state, dict) or "optimizer" not in state:
            raise RuntimeError(f"Checkpoint {resume_path} does not contain optimizer state; use it as a model checkpoint, not --resume.")
        optimizer.load_state_dict(state["optimizer"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        metadata = state.get("metadata", {})
        training_state = state.get("training_state", {})
        start_step = int(metadata.get("iter", training_state.get("step", 0)))
        best_epoch_loss = float(training_state.get("best_epoch_loss", metadata.get("best_loss", float("inf"))))
        epoch_losses = [float(value) for value in training_state.get("epoch_losses", [])]
        history = list(training_state.get("history", []))
        restore_rng_state(training_state.get("rng_state"))
        if args.reset_resume_rng_seed >= 0:
            setup_seed(args.reset_resume_rng_seed)
            print(
                f"[resume] reset_rng_seed={args.reset_resume_rng_seed}",
                flush=True,
            )
        print(f"[resume] checkpoint={resume_path} start_step={start_step} best_loss={best_epoch_loss}", flush=True)
        del model_state, metadata, training_state, state
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    trainable_ids = {
        id(parameter)
        for parameter in trainable.parameters()
        if parameter.requires_grad
    }
    hn_decoder_parameters: List[nn.Parameter] = []
    if hn_replay is not None and args.hn_oe_gradient_scope == "decoder":
        seen_hn_parameters: set[int] = set()
        for name, parameter in model.named_parameters():
            if (
                id(parameter) in trainable_ids
                and id(parameter) not in seen_hn_parameters
                and coarse_parameter_group(name) in {"bottleneck", "decoder"}
            ):
                seen_hn_parameters.add(id(parameter))
                hn_decoder_parameters.append(parameter)
        if not hn_decoder_parameters:
            raise RuntimeError("No bottleneck/decoder parameters found for HN-OE replay.")
    native_bg_parameters: List[nn.Parameter] = []
    if native_bg_teacher is not None:
        seen_native_bg_parameters: set[int] = set()
        for name, parameter in model.named_parameters():
            if (
                id(parameter) in trainable_ids
                and id(parameter) not in seen_native_bg_parameters
                and coarse_parameter_group(name) in {"bottleneck", "decoder"}
            ):
                seen_native_bg_parameters.add(id(parameter))
                native_bg_parameters.append(parameter)
        if not native_bg_parameters:
            raise RuntimeError(
                "No trainable bottleneck/decoder parameters found for Native "
                "background teaching."
            )
    rank_probe_parameters: List[nn.Parameter] = []
    seen_rank_probe_parameters: set[int] = set()
    for name, parameter in model.named_parameters():
        if (
            id(parameter) in trainable_ids
            and id(parameter) not in seen_rank_probe_parameters
            and coarse_parameter_group(name) == "decoder"
        ):
            seen_rank_probe_parameters.add(id(parameter))
            rank_probe_parameters.append(parameter)
    if (args.rank_gradient_audit_only or args.score_rank_separate_backward) and not rank_probe_parameters:
        raise RuntimeError("No trainable decoder parameters were found for rank-gradient diagnostics.")
    rank_gradient_audit_rows: List[Dict[str, float]] = []
    rank_gradient_probe_rows: List[Dict[str, float]] = []
    rank_gradient_audit_path = (
        args.rank_gradient_audit_output
        if args.rank_gradient_audit_output is not None
        else args.output_dir / "rank_gradient_audit.csv"
    )
    gradient_probe = None
    if args.gradient_conflict_probe:
        probe_path = (
            args.gradient_conflict_probe_output
            if args.gradient_conflict_probe_output is not None
            else args.output_dir / "gradient_conflict.csv"
        )
        gradient_probe = GradientConflictProbe(
            model,
            trainable.parameters(),
            probe_path,
            append=start_step > 0,
        )
        print(
            f"[gradient-probe] output={probe_path} every={args.gradient_conflict_probe_every} "
            f"groups={gradient_probe.groups}",
            flush=True,
        )
    conflict_component_scaler = None
    if args.gradient_conflict_component_control:
        component_groups = tuple(
            part.strip()
            for part in args.gradient_conflict_component_groups.split(",")
            if part.strip()
        )
        component_output = (
            args.gradient_conflict_component_output
            if args.gradient_conflict_component_output is not None
            else args.output_dir / "conflict_component_control.csv"
        )
        conflict_component_scaler = GradientConflictComponentScaler(
            model,
            trainable.parameters(),
            conflict_scale=args.gradient_conflict_component_scale,
            budget_beta=None,
            anchor_label=args.gradient_conflict_component_anchor,
            groups=component_groups,
            output_path=component_output,
            append=start_step > 0,
        )
        print(
            "[gradient-conflict-control] "
            f"scale={args.gradient_conflict_component_scale:g} "
            f"budget_beta={args.gradient_conflict_component_budget_beta} "
            f"anchor={args.gradient_conflict_component_anchor} "
            f"groups={component_groups} output={component_output}",
            flush=True,
        )
    normal_iter = repeat_loader(normal_loader)
    det_iter = repeat_loader(det_loader)
    descriptor_source_manifest = getattr(
        args, "guided_prototype_descriptor_source_manifest", None
    )
    source_group_resolver = SourceGroupResolver(
        [descriptor_source_manifest] if descriptor_source_manifest is not None else []
    )
    guided_config = getattr(model, "_guided_prototype_config", None)
    context_transport_enabled = bool(
        args.guided_prototype and getattr(guided_config, "context_transport", False)
    )
    if context_transport_enabled and not context_familiarity_transport_memory_ready(model):
        config = model._guided_prototype_config
        memory_diag = fit_context_familiarity_transport_memory(
            model,
            normal_loader,
            device=device,
            source_group_resolver=source_group_resolver,
            candidates_per_image=config.context_candidates_per_image,
            candidates_per_group=config.context_candidates_per_group,
            memory_build_batches=config.context_memory_build_batches,
        )
        print(
            "[guided_context_transport_memory] "
            + " ".join(f"{key}={value:g}" for key, value in memory_diag.items()),
            flush=True,
        )
    descriptor_enabled = bool(
        args.guided_prototype
        and getattr(guided_config, "descriptor_variant", "off") != "off"
    )
    if descriptor_enabled and not normal_descriptor_memory_ready(model):
        config = model._guided_prototype_config
        memory_diag = fit_normal_descriptor_memory(
            model,
            normal_loader,
            device=device,
            source_group_resolver=source_group_resolver,
            candidates_per_image=config.descriptor_candidates_per_image,
            candidates_per_group=config.descriptor_candidates_per_group,
            memory_build_batches=config.descriptor_memory_build_batches,
        )
        print(
            "[guided_normal_descriptor_memory] "
            + " ".join(f"{key}={value:g}" for key, value in memory_diag.items()),
            flush=True,
        )
    use_source_groups = bool(
        args.guided_prototype
        and (
            context_transport_enabled
            or descriptor_enabled
            or (
                args.guided_prototype_familiarity_gate
                and args.guided_prototype_familiarity_calibration == "cross_group"
            )
        )
    )
    started = time.time()

    def save_checkpoint(path: Path, step: int, include_training_state: bool = False, **metadata) -> None:
        payload = checkpoint_payload(
            model,
            iter=step,
            architecture=args.architecture,
            encoder=args.encoder,
            inp_num=args.inp_num,
            normal_loss=args.normal_loss,
            inp_coherence_loss=args.inp_coherence_loss,
            inp_soft_mining_gamma=args.inp_soft_mining_gamma,
            train_distance=args.train_distance,
            det_manifest=[str(path) for path in args.det_manifests],
            **metadata,
        )
        if include_training_state:
            payload["optimizer"] = optimizer.state_dict()
            payload["scaler"] = scaler.state_dict()
            payload["training_state"] = {
                "step": int(step),
                "best_epoch_loss": float(best_epoch_loss),
                "epoch_losses": [float(value) for value in epoch_losses],
                "history": history,
                "rng_state": capture_rng_state(),
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

    inline_validator = None
    if args.inline_fod_val_every_epochs > 0:
        inline_validator = InlineFODValidator(
            args,
            InlineValidationSchedule(
                origin_step=0,
                epoch_iters=steps_per_epoch,
                every_epochs=args.inline_fod_val_every_epochs,
                include_origin=False,
                start_epoch=args.inline_fod_val_start_epoch,
            ),
        )

    def run_inline_validation(step: int) -> None:
        if inline_validator is None or not inline_validator.should_validate(step):
            return

        def save_for_validation(path: Path, validation_step: int) -> None:
            save_checkpoint(
                path,
                validation_step,
                include_training_state=not args.final_only_checkpoints,
                checkpoint_kind="inline_val",
                epoch=validation_step // steps_per_epoch,
                best_loss=best_epoch_loss,
            )

        inline_validator.validate(model, step, save_for_validation)

    loop_end_step = (
        start_step + int(args.rank_gradient_audit_steps)
        if args.rank_gradient_audit_only
        else total_steps
    )
    if not args.rank_gradient_audit_only and args.stop_after_step > 0:
        loop_end_step = min(loop_end_step, int(args.stop_after_step))
    if not args.rank_gradient_audit_only and start_step >= total_steps:
        print(f"[resume] start_step={start_step} already reaches total_steps={total_steps}", flush=True)

    configured_score_rank_weight = float(args.lambda_score_rank)
    configured_score_cover_weight = float(args.lambda_score_cover)
    configured_conflict_component_scale = float(args.gradient_conflict_component_scale)
    configured_conflict_budget_beta = args.gradient_conflict_component_budget_beta
    previous_phase: tuple[bool, bool, bool] | None = None

    for step in range(start_step + 1, loop_end_step + 1):
        target_gate_warmup_active = bool(
            args.target_gate_warmup_steps > 0
            and step <= int(args.target_gate_warmup_steps)
        )
        for parameter in base_params:
            parameter.requires_grad_(True)
        current_score_rank_weight = activated_step_value(
            configured_score_rank_weight,
            step,
            args.score_rank_start_step,
            inactive_value=0.0,
        )
        rank_active = current_score_rank_weight > 0.0
        current_score_cover_weight = activated_step_value(
            configured_score_cover_weight,
            step,
            args.score_cover_start_step,
            inactive_value=0.0,
        )
        cover_active = current_score_cover_weight > 0.0
        conflict_control_active = (
            conflict_component_scaler is not None
            and step >= int(args.gradient_conflict_component_start_step)
        )
        args.lambda_score_rank = current_score_rank_weight
        args.lambda_score_cover = current_score_cover_weight
        if conflict_component_scaler is not None:
            conflict_component_scaler.conflict_scale = activated_step_value(
                configured_conflict_component_scale,
                step,
                args.gradient_conflict_component_start_step,
                inactive_value=1.0,
            )
            conflict_component_scaler.budget_beta = (
                configured_conflict_budget_beta
                if conflict_control_active
                else None
            )
        phase = (rank_active, cover_active, conflict_control_active)
        if phase != previous_phase:
            print(
                "[oe-phase] "
                f"step={step} rank_active={int(rank_active)} "
                f"rank_weight={args.lambda_score_rank:g} "
                f"cover_active={int(cover_active)} "
                f"cover_weight={args.lambda_score_cover:g} "
                f"conflict_control_active={int(conflict_control_active)} "
                f"conflict_scale={getattr(conflict_component_scaler, 'conflict_scale', 1.0):g} "
                f"conflict_budget_beta={getattr(conflict_component_scaler, 'budget_beta', None)} "
                f"conflict_anchor={args.gradient_conflict_component_anchor}",
                flush=True,
            )
            previous_phase = phase
        current_lr = scheduled_learning_rate(
            args.lr,
            args.min_lr,
            step,
            max(total_steps, loop_end_step),
            args.lr_schedule,
            warmup_steps=args.lr_warmup_steps,
            warmup_start_factor=args.lr_warmup_start_factor,
            decay_start_step=args.lr_decay_start_step,
            decay_end_step=args.lr_decay_end_step,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr * float(param_group.get("lr_scale", 1.0))
        normal_loss_step = step + int(args.normal_loss_step_offset)
        normal_batch = next(normal_iter)
        if args.roi_mask_loss:
            normal_images, _, normal_paths, normal_roi_mask = normal_batch
            normal_roi_mask = normal_roi_mask.to(device, non_blocking=True)
        else:
            normal_images, _, normal_paths = normal_batch
            normal_roi_mask = None
        det_images, det_boxes, det_valid, det_all_object_mask, _ = next(det_iter)
        normal_images = normal_images.to(device, non_blocking=True)
        normal_group_ids = source_group_resolver.ids(normal_paths, device=device) if use_source_groups else None
        det_images = det_images.to(device, non_blocking=True)
        det_boxes = det_boxes.to(device, non_blocking=True)
        det_valid = det_valid.to(device, non_blocking=True)
        det_all_object_mask = det_all_object_mask.to(device, non_blocking=True)
        target_background_diag = {
            "target_bg_pastes": 0.0,
            "target_bg_pastes_per_image": 0.0,
            "target_bg_mask_fraction": 0.0,
            "target_bg_samples_with_paste": 0.0,
        }
        if args.det_target_background_compose:
            det_images, det_boxes, det_valid, det_all_object_mask, target_background_diag = (
                compose_target_background_batch(
                    det_images,
                    det_boxes,
                    det_valid,
                    normal_images,
                    seed=int(args.seed) + 1000003 * int(step),
                    max_pastes=int(args.det_target_background_max_pastes),
                    feather=int(args.det_target_background_feather),
                    alpha_shape=str(args.det_target_background_alpha_shape),
                )
            )
        det_weight = scheduled_weight(
            args.det_loss_weight,
            step,
            total_steps,
            args.det_loss_start_frac,
            args.det_loss_end_frac,
            warmup_steps=args.det_loss_warmup_steps,
        )
        texture_weight = scheduled_weight(
            args.normal_texture_preserve_weight,
            step,
            total_steps,
            args.normal_texture_start_frac,
            args.normal_texture_end_frac,
        )
        normal_prior_weight = scheduled_weight(
            args.normal_prior_recon_weight,
            step,
            total_steps,
            args.normal_prior_start_frac,
            args.normal_prior_end_frac,
        )

        optimizer.zero_grad(set_to_none=True)
        target_proto_aux_active = bool(
            args.target_proto_invariance_weight > 0.0
            or args.target_proto_repulsion_weight > 0.0
            or args.target_aggregation_attention_weight > 0.0
        )
        model._guided_capture_aggregation_attention = bool(
            args.target_aggregation_attention_weight > 0.0
        )
        read_capture = None
        if args.target_read_attention_weight > 0.0:
            read_capture = DecoderAttentionCapture(model.decoder)
            read_capture.start()
        with torch.cuda.amp.autocast(enabled=use_amp):
            set_guided_prototype_source_groups(model, normal_group_ids)
            set_guided_prototype_valid_roi(model, normal_roi_mask)
            set_guided_prototype_image(model, normal_images, update_prior_stats=True)
            normal_out_full = None
            normal_loss_diag: Dict[str, float] = {}
            target_side_repulsion = bool(
                args.target_proto_repulsion_weight > 0.0
                and args.target_proto_repulsion_gradient_side == "target"
            )
            if args.masked_recon:
                pattern = (step - 1) % max(1, int(args.mask_segments))
                normal_en, normal_de, normal_mask = forward_masked_reconstruction(
                    model,
                    normal_images,
                    pattern=pattern,
                    config=mask_config,
                )
                masked_gather_loss = getattr(model, "_masked_gather_loss", None)
                l_masked_normal, _ = compute_masked_normal_loss(
                    normal_en,
                    normal_de,
                    mask=normal_mask,
                    architecture=args.architecture,
                    step=normal_loss_step,
                    normal_loss=args.normal_loss,
                    prototype_loss_weight=args.prototype_loss_weight,
                    gather_loss=masked_gather_loss if args.architecture == "inpformer" else None,
                    hard_quantile=args.hard_quantile,
                    easy_weight=args.easy_weight,
                    band_weights=mask_band_loss_weights,
                    dinomaly_p_final=args.dinomaly_native_p_final,
                    dinomaly_warmup_iters=args.dinomaly_native_warmup_iters,
                    dinomaly_factor=args.dinomaly_native_factor,
                    inpformer_y=args.inpformer_native_y,
                    inpformer_soft_mining_gamma=args.inp_soft_mining_gamma,
                )
                l_normal = args.mask_loss_weight * l_masked_normal
                if args.full_anchor_loss_weight > 0.0:
                    full_out = model(normal_images)
                    full_loss, _ = compute_normal_loss(
                        full_out,
                        architecture=args.architecture,
                        step=normal_loss_step,
                        normal_loss=args.normal_loss,
                        prototype_loss_weight=args.prototype_loss_weight,
                        hard_quantile=args.hard_quantile,
                        easy_weight=args.easy_weight,
                        dinomaly_p_final=args.dinomaly_native_p_final,
                        dinomaly_warmup_iters=args.dinomaly_native_warmup_iters,
                        dinomaly_factor=args.dinomaly_native_factor,
                        inpformer_y=args.inpformer_native_y,
                        inpformer_soft_mining_gamma=args.inp_soft_mining_gamma,
                    )
                    l_normal = l_normal + args.full_anchor_loss_weight * full_loss
                normal_agg_prototype = None
            elif (
                args.det_prototype_mode in {"normal_context", "paired_normal_context"}
                or (target_proto_aux_active and args.architecture == "inpformer")
            ):
                if args.architecture != "inpformer":
                    raise ValueError(
                        "--det-prototype-mode normal_context/paired_normal_context "
                        "is only supported for --architecture inpformer."
                    )
                normal_out_full = inpformer_forward_with_agg_prototype(
                    model,
                    normal_images,
                    return_agg_prototype=True,
                    return_target_tokens=True,
                    return_bottleneck_tokens=target_side_repulsion,
                )
                normal_out = normal_out_full[:3]
                if args.det_prototype_mode == "normal_context":
                    normal_agg_prototype = normal_out_full[3].detach().mean(dim=0, keepdim=True)
                elif args.det_prototype_mode == "paired_normal_context":
                    normal_agg_prototype = normal_out_full[3].detach()
                else:
                    normal_agg_prototype = None
                normal_en, normal_de = normal_out[:2]
                if normal_roi_mask is None:
                    l_normal, normal_loss_diag = compute_normal_loss(
                        normal_out,
                        architecture=args.architecture,
                        step=normal_loss_step,
                        normal_loss=args.normal_loss,
                        prototype_loss_weight=args.prototype_loss_weight,
                        hard_quantile=args.hard_quantile,
                        easy_weight=args.easy_weight,
                        dinomaly_p_final=args.dinomaly_native_p_final,
                        dinomaly_warmup_iters=args.dinomaly_native_warmup_iters,
                        dinomaly_factor=args.dinomaly_native_factor,
                        inpformer_y=args.inpformer_native_y,
                        inpformer_soft_mining_gamma=args.inp_soft_mining_gamma,
                    )
                else:
                    l_normal, normal_loss_diag = compute_roi_masked_normal_loss(
                        normal_out,
                        normal_roi_mask,
                        architecture=args.architecture,
                        step=normal_loss_step,
                        normal_loss=args.normal_loss,
                        prototype_loss_weight=args.prototype_loss_weight,
                        gather_distance=getattr(model, "distance", None),
                        hard_quantile=args.hard_quantile,
                        easy_weight=args.easy_weight,
                        dinomaly_p_final=args.dinomaly_native_p_final,
                        dinomaly_warmup_iters=args.dinomaly_native_warmup_iters,
                        dinomaly_factor=args.dinomaly_native_factor,
                        inpformer_y=args.inpformer_native_y,
                    )
            else:
                normal_out = model(normal_images)
                normal_agg_prototype = None
                normal_en, normal_de = normal_out[:2]
                if normal_roi_mask is None:
                    l_normal, normal_loss_diag = compute_normal_loss(
                        normal_out,
                        architecture=args.architecture,
                        step=normal_loss_step,
                        normal_loss=args.normal_loss,
                        prototype_loss_weight=args.prototype_loss_weight,
                        hard_quantile=args.hard_quantile,
                        easy_weight=args.easy_weight,
                        dinomaly_p_final=args.dinomaly_native_p_final,
                        dinomaly_warmup_iters=args.dinomaly_native_warmup_iters,
                        dinomaly_factor=args.dinomaly_native_factor,
                        inpformer_y=args.inpformer_native_y,
                        inpformer_soft_mining_gamma=args.inp_soft_mining_gamma,
                    )
                else:
                    l_normal, normal_loss_diag = compute_roi_masked_normal_loss(
                        normal_out,
                        normal_roi_mask,
                        architecture=args.architecture,
                        step=normal_loss_step,
                        normal_loss=args.normal_loss,
                        prototype_loss_weight=args.prototype_loss_weight,
                        gather_distance=getattr(model, "distance", None),
                        hard_quantile=args.hard_quantile,
                        easy_weight=args.easy_weight,
                        dinomaly_p_final=args.dinomaly_native_p_final,
                        dinomaly_warmup_iters=args.dinomaly_native_warmup_iters,
                        dinomaly_factor=args.dinomaly_native_factor,
                        inpformer_y=args.inpformer_native_y,
                    )
            normal_guided_diag = get_guided_prototype_diag(model)
            normal_aux_prototype = None
            normal_aux_tokens = None
            normal_aux_bottleneck_tokens = None
            if target_proto_aux_active:
                if normal_out_full is not None and len(normal_out_full) >= 5:
                    normal_aux_prototype = normal_out_full[3]
                    normal_aux_tokens = normal_out_full[4]
                    if target_side_repulsion:
                        normal_aux_bottleneck_tokens = normal_out_full[5]
                else:
                    normal_aux_prototype = getattr(
                        model, "_guided_last_aggregate_prototype", None
                    )
                    normal_aux_tokens = getattr(
                        model, "_guided_decoder_target_tokens", None
                    )
            normal_aggregation_attention = tuple(
                attention.detach()
                for attention in getattr(model, "_guided_last_aggregation_attention", ())
            )
            if normal_aux_prototype is not None:
                normal_aux_prototype = normal_aux_prototype.detach()
            if normal_aux_tokens is not None:
                normal_aux_tokens = normal_aux_tokens.detach()
            if normal_aux_bottleneck_tokens is not None:
                normal_aux_bottleneck_tokens = normal_aux_bottleneck_tokens.detach()
            normal_read_attention = ()
            if read_capture is not None:
                decoder_layers = len(model.decoder)
                if len(read_capture.outputs) < decoder_layers:
                    raise RuntimeError("Normal forward did not expose all decoder attention layers.")
                normal_read_attention = tuple(
                    attention.detach()
                    for attention in read_capture.recompute_last(decoder_layers)
                )
                read_capture.clear()
            if (
                args.target_gate_loss_weight > 0.0
                and args.target_gate_normal_anchor_weight > 0.0
            ):
                l_target_gate_normal_anchor, target_gate_normal_diag = (
                    guided_target_gate_normal_anchor_loss(
                        model, valid_roi_mask=normal_roi_mask
                    )
                )
            else:
                l_target_gate_normal_anchor = normal_en[0].new_tensor(0.0)
                target_gate_normal_diag = {
                    "l_target_gate_normal_anchor": 0.0,
                    "target_gate_normal_raw_mean": 0.0,
                    "target_gate_normal_calibrated_mean": 0.0,
                }
            l_guided_extra, guided_extra_diag = guided_prototype_extra_loss(
                model,
                normal_en,
                normal_de,
            )
            l_normal_fp = (
                normal_fp_score_loss(
                    merged_feature(normal_en, detach=True),
                    merged_feature(normal_de, detach=False),
                    args,
                )
                if args.lambda_normal_fp > 0.0
                else normal_en[0].new_tensor(0.0)
            )
            normal_enc_feature = merged_feature(normal_en, detach=True)
            normal_dec_feature = merged_feature(normal_de, detach=False)
            if texture_weight > 0.0:
                l_texture, texture_diag = normal_texture_preservation_loss(
                    normal_images,
                    normal_enc_feature,
                    normal_dec_feature,
                    args,
                )
            else:
                l_texture = normal_en[0].new_tensor(0.0)
                texture_diag = {
                    "l_texture_keep": 0.0,
                    "l_texture_not_bg": 0.0,
                    "texture_weight_mean": 0.0,
                }
            if normal_prior_weight > 0.0:
                l_normal_prior, normal_prior_diag = normal_prior_reconstruction_loss(
                    normal_images,
                    normal_enc_feature,
                    normal_dec_feature,
                    args,
                )
            else:
                l_normal_prior = normal_en[0].new_tensor(0.0)
                normal_prior_diag = {
                    "l_normal_prior": 0.0,
                    "normal_prior_mean": 0.0,
                    "normal_prior_texture_mean": 0.0,
                    "normal_prior_object_mean": 0.0,
                }

            hn_restore_active = bool(
                hn_replay is not None and step >= int(args.hn_oe_start_step)
            )
            if hn_restore_active:
                hn_score = reconstruction_score(model, normal_en, normal_de)
                hn_side = int(hn_score.shape[-1])
                hn_weights, hn_targets = hn_replay.batch_targets(
                    normal_paths,
                    token_side=hn_side,
                    physical_crop_size=int(args.crop_size),
                    device=device,
                )
                l_hn_restore, hn_diag = causal_restore_loss(
                    hn_score,
                    hn_weights,
                    hn_targets,
                    margin=float(args.hn_oe_margin),
                )
            else:
                l_hn_restore = normal_en[0].new_tensor(0.0)
                hn_diag = {
                    "l_hn_restore": 0.0,
                    "hn_score_mean": 0.0,
                    "hn_p2_target_mean": 0.0,
                    "hn_active_fraction": 0.0,
                    "hn_batch_occurrences": 0.0,
                    "hn_batch_weight": 0.0,
                }
            hn_objective = float(args.hn_oe_loss_weight) * l_hn_restore

            normal_base_objective = float(args.normal_loss_weight) * (
                l_normal
                + l_guided_extra
                + args.lambda_normal_fp * l_normal_fp
                + texture_weight * l_texture
                + normal_prior_weight * l_normal_prior
            )
            normal_base_objective = normal_base_objective + (
                float(args.target_gate_loss_weight)
                * float(args.target_gate_normal_anchor_weight)
                * l_target_gate_normal_anchor
            )
            normal_objective = normal_base_objective + (
                hn_objective
                if hn_restore_active and args.hn_oe_gradient_scope == "all"
                else 0.0
            )
            if args.sequential_branch_backward:
                decoder_only_hn = bool(
                    hn_restore_active and args.hn_oe_gradient_scope == "decoder"
                )
                scaler.scale(normal_objective).backward(retain_graph=decoder_only_hn)
                if decoder_only_hn:
                    hn_gradients = torch.autograd.grad(
                        scaler.scale(hn_objective),
                        hn_decoder_parameters,
                        retain_graph=False,
                        allow_unused=True,
                    )
                    for parameter, gradient in zip(hn_decoder_parameters, hn_gradients):
                        if gradient is None:
                            continue
                        if parameter.grad is None:
                            parameter.grad = gradient
                        else:
                            parameter.grad.add_(gradient)
                    del hn_gradients

            set_guided_prototype_source_groups(model, None)
            set_guided_prototype_valid_roi(model, None)
            det_aux_tokens = None
            det_aux_bottleneck_tokens = None
            det_aux_prototype = None
            if args.det_prototype_mode == "joint" or args.architecture != "inpformer":
                set_guided_prototype_image(model, det_images, update_prior_stats=False)
                det_en, det_de = model(det_images)[:2]
            elif args.det_prototype_mode == "freeze":
                set_guided_prototype_image(model, det_images, update_prior_stats=False)
                with temporary_requires_grad(inpformer_prototype_params(model), False):
                    if target_proto_aux_active:
                        det_out_full = inpformer_forward_with_agg_prototype(
                            model,
                            det_images,
                            return_agg_prototype=True,
                            return_target_tokens=True,
                            return_bottleneck_tokens=target_side_repulsion,
                        )
                        det_en, det_de = det_out_full[:2]
                        det_aux_prototype = det_out_full[3]
                        det_aux_tokens = det_out_full[4]
                        if target_side_repulsion:
                            det_aux_bottleneck_tokens = det_out_full[5]
                    else:
                        det_en, det_de = model(det_images)[:2]
            elif args.det_prototype_mode == "normal_context":
                assert normal_agg_prototype is not None
                det_en, det_de = inpformer_forward_with_agg_prototype(
                    model,
                    det_images,
                    agg_prototype=normal_agg_prototype,
                )[:2]
            elif args.det_prototype_mode == "paired_normal_context":
                assert normal_agg_prototype is not None
                det_context = paired_normal_prototype_context(
                    normal_agg_prototype,
                    int(det_images.shape[0]),
                )
                det_en, det_de = inpformer_forward_with_agg_prototype(
                    model,
                    det_images,
                    agg_prototype=det_context,
                )[:2]
            else:
                raise ValueError(f"Unsupported det_prototype_mode={args.det_prototype_mode!r}.")
            adaptive_native_bg_score = None
            if native_bg_teacher is not None:
                adaptive_native_bg_score = reconstruction_score(model, det_en, det_de)
                with torch.no_grad():
                    native_bg_out = native_bg_teacher(det_images)
                    native_bg_score = reconstruction_score(
                        native_bg_teacher,
                        native_bg_out[0],
                        native_bg_out[1],
                    )
                l_native_bg_excess, native_bg_diag = native_background_excess_loss(
                    adaptive_native_bg_score,
                    native_bg_score,
                    det_all_object_mask,
                    minimum_tokens=int(args.native_bg_excess_min_tokens),
                    target_dilate=int(args.native_bg_excess_target_dilate),
                )
            else:
                l_native_bg_excess = det_en[0].new_tensor(0.0)
                native_bg_diag = {
                    "l_native_bg_excess": 0.0,
                    "native_bg_score_mean": 0.0,
                    "adaptive_bg_score_mean": 0.0,
                    "native_bg_excess_mean": 0.0,
                    "native_bg_excess_active_fraction": 0.0,
                    "native_bg_excess_selected_tokens": 0.0,
                    "native_bg_background_tokens": 0.0,
                }
            native_bg_objective = (
                float(args.native_bg_excess_weight) * l_native_bg_excess
            )
            det_read_attention = ()
            if read_capture is not None:
                decoder_layers = len(model.decoder)
                if len(read_capture.outputs) < decoder_layers:
                    raise RuntimeError("Target composite forward did not expose all decoder attention layers.")
                det_read_attention = read_capture.recompute_last(decoder_layers)
                read_capture.stop()
            if args.target_gate_loss_weight > 0.0:
                l_target_gate, target_gate_diag = guided_target_gate_supervision_loss(
                    model,
                    det_all_object_mask,
                )
            else:
                l_target_gate = det_en[0].new_tensor(0.0)
                target_gate_diag = {
                    "l_target_gate": 0.0,
                    "target_gate_positive_fraction": 0.0,
                    "target_gate_positive_risk": 0.0,
                    "target_gate_background_risk": 0.0,
                    "target_gate_balanced_accuracy": 0.0,
                }
            target_aux_diag = {
                "l_target_proto_invariance": 0.0,
                "target_proto_cosine": 0.0,
                "target_proto_drift": 0.0,
                "target_proto_worst_drift": 0.0,
                "l_target_proto_repulsion": 0.0,
                "target_proto_repulsion_margin": 0.0,
                "target_proto_repulsion_margin_min": 0.0,
                "target_proto_repulsion_margin_max": 0.0,
                "target_proto_effective_mode_count": 0.0,
                "target_proto_margin_fallback_modes": 0.0,
                "target_proto_distance_mean": 0.0,
                "target_proto_repulsion_active": 0.0,
                "normal_proto_distance_mean": 0.0,
                "normal_proto_distance_q95": 0.0,
                "normal_proto_distance_q99": 0.0,
                "target_proto_core_fraction": 0.0,
                "target_proto_occupancy_mean": 0.0,
                "target_proto_gradient_side_target": 0.0,
                "l_target_aggregation_attention": 0.0,
                "l_target_aggregation_suppress": 0.0,
                "l_target_aggregation_background_anchor": 0.0,
                "target_aggregation_attention_mass": 0.0,
                "target_aggregation_attention_density_ratio": 0.0,
                "target_aggregation_attention_layers": 0.0,
                "l_target_read_attention": 0.0,
                "l_target_read_suppress": 0.0,
                "l_target_read_background_anchor": 0.0,
                "target_read_attention_mean": 0.0,
                "background_read_attention_mean": 0.0,
                "target_read_attention_ratio": 0.0,
                "target_read_attention_layers": 0.0,
            }
            zero_aux = det_en[0].new_tensor(0.0)
            l_target_proto_invariance = zero_aux
            l_target_proto_repulsion = zero_aux
            l_target_aggregation_attention = zero_aux
            l_target_read_attention = zero_aux
            if target_proto_aux_active:
                if normal_aux_prototype is None:
                    raise RuntimeError("Target prototype loss did not receive clean Normal prototypes.")
                target_tokens = det_aux_tokens
                if target_tokens is None:
                    target_tokens = getattr(model, "_guided_decoder_target_tokens", None)
                if target_tokens is None:
                    raise RuntimeError("Target prototype loss did not receive composite tokens.")
                prototype_to_mode = None
                if args.target_proto_repulsion_margin_scope == "effective_mode":
                    teacher = getattr(model, "guided_center6_teacher", None)
                    prototype_to_mode = getattr(teacher, "slot_to_mode", None)
                    if prototype_to_mode is None:
                        raise RuntimeError(
                            "Effective-mode target prototype loss did not receive "
                            "a prototype-to-mode mapping."
                        )
                if target_side_repulsion:
                    if det_aux_prototype is None or det_aux_bottleneck_tokens is None:
                        raise RuntimeError(
                            "Target-side prototype repulsion did not receive "
                            "detached adaptive prototypes and bottleneck tokens."
                        )
                    composite_prototype = det_aux_prototype.detach()
                else:
                    with temporary_requires_grad(target_gate_params, False):
                        composite_prototype = inpformer_aggregate_prototypes(
                            model,
                            target_tokens,
                            images=det_images,
                        )
                if args.target_proto_invariance_weight > 0.0:
                    l_target_proto_invariance, diag = prototype_invariance_loss(
                        composite_prototype,
                        normal_aux_prototype,
                        prototype_to_mode=prototype_to_mode,
                    )
                    target_aux_diag.update(diag)
                if args.target_proto_repulsion_weight > 0.0:
                    repulsion_target_tokens = (
                        det_aux_bottleneck_tokens
                        if target_side_repulsion
                        else target_tokens
                    )
                    repulsion_normal_tokens = (
                        normal_aux_bottleneck_tokens
                        if target_side_repulsion
                        else normal_aux_tokens
                    )
                    if repulsion_target_tokens is None or repulsion_normal_tokens is None:
                        raise RuntimeError("Target prototype repulsion did not receive Normal tokens.")
                    l_target_proto_repulsion, diag = prototype_repulsion_loss(
                        repulsion_target_tokens,
                        composite_prototype,
                        det_all_object_mask,
                        repulsion_normal_tokens,
                        normal_aux_prototype,
                        normal_quantile=args.target_proto_repulsion_normal_quantile,
                        margin_delta=args.target_proto_repulsion_margin_delta,
                        minimum_occupancy=args.target_proto_core_min_occupancy,
                        prototype_to_mode=prototype_to_mode,
                        minimum_normal_tokens_per_mode=(
                            args.target_proto_repulsion_min_normal_tokens_per_mode
                        ),
                        gradient_side=args.target_proto_repulsion_gradient_side,
                        mode_budget=args.target_proto_repulsion_mode_budget,
                        clean_valid_mask=normal_roi_mask,
                    )
                    target_aux_diag.update(diag)
                if args.target_aggregation_attention_weight > 0.0:
                    composite_attention = getattr(
                        model, "_guided_last_aggregation_attention", ()
                    )
                    l_target_aggregation_attention, diag = (
                        aggregation_attention_exclusion_loss(
                            composite_attention,
                            normal_aggregation_attention,
                            det_all_object_mask,
                            target_to_background_ratio=args.target_aggregation_attention_ratio,
                            background_anchor_weight=args.target_aggregation_background_anchor_weight,
                        )
                    )
                    target_aux_diag.update(diag)
            if args.target_read_attention_weight > 0.0:
                l_target_read_attention, diag = decoder_read_attention_exclusion_loss(
                    det_read_attention,
                    normal_read_attention,
                    det_all_object_mask,
                    target_to_background_ratio=args.target_read_attention_ratio,
                    background_anchor_weight=args.target_read_background_anchor_weight,
                )
                target_aux_diag.update(diag)
            l_target_prototype_objective = (
                float(args.target_proto_invariance_weight) * l_target_proto_invariance
                + float(args.target_proto_repulsion_weight) * l_target_proto_repulsion
                + float(args.target_aggregation_attention_weight)
                * l_target_aggregation_attention
                + float(args.target_read_attention_weight) * l_target_read_attention
            )
            det_enc_feature = merged_feature(det_en, detach=True)
            det_dec_feature = merged_feature(det_de, detach=False)
            det_objectness = (
                combined_guidance_map(det_images, det_enc_feature, args)
                if args.det_objectness_weighting
                else None
            )
            if (
                args.score_rank_background_mode == "global_clean"
                and (args.lambda_score_rank > 0.0 or args.lambda_score_cover > 0.0)
            ):
                det_score_map = (
                    adaptive_native_bg_score
                    if adaptive_native_bg_score is not None
                    else reconstruction_score(model, det_en, det_de)
                )
            else:
                det_score_map = None
            l_det_legacy, l_det_rank_weighted, l_det_cover_weighted, det_diag = object_erasing_loss_terms(
                det_enc_feature,
                det_dec_feature,
                det_boxes,
                det_valid,
                args,
                objectness_weight=det_objectness,
                all_object_mask=det_all_object_mask,
                score_map=det_score_map,
            )
            l_det = l_det_legacy + l_det_rank_weighted + l_det_cover_weighted
            det_objective = (
                det_weight * l_det
                + float(args.target_gate_loss_weight) * l_target_gate
                + l_target_prototype_objective
            )
            det_legacy_objective = det_weight * (l_det_legacy + l_det_cover_weighted)
            det_rank_objective = det_weight * l_det_rank_weighted
            det_rank_raw_objective = det_weight * (
                l_det_rank_weighted / float(args.lambda_score_rank)
                if args.lambda_score_rank > 0.0
                else l_det_rank_weighted
            )
            if args.sequential_branch_backward:
                logged_normal_objective = normal_objective + (
                    hn_objective
                    if hn_restore_active and args.hn_oe_gradient_scope == "decoder"
                    else 0.0
                )
                loss = (
                    logged_normal_objective.detach()
                    + det_objective
                    + native_bg_objective.detach()
                )
            else:
                loss = normal_objective + det_objective + native_bg_objective.detach()

        native_bg_gradients = []
        if native_bg_teacher is not None:
            native_bg_gradients = torch.autograd.grad(
                scaler.scale(native_bg_objective),
                native_bg_parameters,
                retain_graph=True,
                allow_unused=True,
            )
        if args.rank_gradient_audit_only:
            normal_gradients = [
                parameter.grad.detach() if parameter.grad is not None else None
                for parameter in rank_probe_parameters
            ]
            legacy_gradients = torch.autograd.grad(
                scaler.scale(det_legacy_objective),
                rank_probe_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            rank_gradients = torch.autograd.grad(
                scaler.scale(det_rank_raw_objective),
                rank_probe_parameters,
                retain_graph=False,
                allow_unused=True,
            )
            audit_diag = gradient_triplet_diagnostics(
                normal_gradients,
                legacy_gradients,
                rank_gradients,
                target_rank_to_legacy_ratio=args.rank_gradient_audit_target_ratio,
            )
            audit_row: Dict[str, float] = {
                "step": float(step),
                "audit_batch": float(step - start_step),
                "l_normal": float(normal_objective.detach().cpu()),
                "l_det_legacy": float(l_det_legacy.detach().cpu()),
                "l_score_rank": float(det_diag["l_score_rank"]),
                "score_rank_bg_mean": float(det_diag["score_rank_bg_mean"]),
                "score_rank_obj_mean": float(det_diag["score_rank_obj_mean"]),
                "score_rank_gap_mean": float(det_diag["score_rank_gap_mean"]),
                "score_rank_active_fraction": float(det_diag["score_rank_active_fraction"]),
                "global_clean_mode": float(args.score_rank_background_mode == "global_clean"),
                **target_aux_diag,
                **audit_diag,
            }
            rank_gradient_audit_rows.append(audit_row)
            if (
                step == start_step + 1
                or (step - start_step) % max(1, args.progress_every) == 0
                or step == loop_end_step
            ):
                print(
                    "[rank-gradient-audit] "
                    f"batch={step - start_step}/{args.rank_gradient_audit_steps} "
                    f"rank_normal_cos={audit_row['rank_normal_cosine']:.6f} "
                    f"joint_survival={audit_row['rank_effective_joint_k0_ratio']:.6f} "
                    f"raw_ratio={audit_row['rank_to_legacy_effective_ratio_unweighted']:.6f} "
                    f"suggested_w={audit_row['suggested_rank_weight']:.6f} "
                    f"active={audit_row['score_rank_active_fraction']:.3f}",
                    flush=True,
                )
            optimizer.zero_grad(set_to_none=True)
            normal_images = det_images = None
            det_boxes = det_valid = det_all_object_mask = None
            normal_en = normal_de = det_en = det_de = None
            normal_out = normal_out_full = normal_agg_prototype = None
            normal_enc_feature = normal_dec_feature = None
            det_enc_feature = det_dec_feature = det_objectness = det_score_map = None
            normal_objective = det_objective = det_legacy_objective = None
            det_rank_objective = det_rank_raw_objective = loss = None
            l_normal = l_guided_extra = l_normal_fp = None
            l_texture = l_normal_prior = l_det = l_det_legacy = None
            l_det_rank_weighted = l_det_cover_weighted = None
            del normal_gradients, legacy_gradients, rank_gradients
            torch.cuda.empty_cache()
            continue

        probe_this_step = bool(
            gradient_probe is not None
            and step % int(args.gradient_conflict_probe_every) == 0
        )
        if args.sequential_branch_backward:
            separate_rank_active = bool(
                args.score_rank_separate_backward and rank_active
            )
            anchor_includes_rank = bool(
                separate_rank_active
                and args.gradient_conflict_component_anchor == "normal_rank"
            )
            if anchor_includes_rank:
                scaler.scale(det_rank_objective).backward(retain_graph=True)
            if probe_this_step:
                gradient_probe.begin_det()
            if conflict_component_scaler is not None:
                conflict_component_scaler.begin_det()
            projected_det_objective = (
                det_legacy_objective
                if separate_rank_active
                else det_objective
            )
            scaler.scale(projected_det_objective).backward(
                retain_graph=(
                    separate_rank_active and not anchor_includes_rank
                )
            )
            if probe_this_step:
                probe_rows = gradient_probe.end_det(
                    {
                        "step": step,
                        "epoch": step / max(steps_per_epoch, 1),
                        "normal_loss": float(normal_objective.detach().cpu()),
                        "det_loss": float(projected_det_objective.detach().cpu()),
                        "det_weight": float(det_weight),
                        "learning_rate": float(current_lr),
                    }
                )
                if step % steps_per_epoch == 0:
                    overall_probe = probe_rows[0]
                    print(
                        "[gradient-probe] "
                        f"epoch={step // steps_per_epoch} "
                        f"cos_shared={float(overall_probe['cosine_shared']):.6f} "
                        f"cos_full={float(overall_probe['cosine_full']):.6f} "
                        f"det_normal_ratio={float(overall_probe['det_to_normal_shared_ratio']):.6f} "
                        f"tensor_conflict={float(overall_probe['tensor_conflict_fraction']):.6f}",
                        flush=True,
                    )
            conflict_component_diag = {}
            if conflict_component_scaler is not None:
                conflict_component_diag = conflict_component_scaler.end_det(
                    {
                        "step": step,
                        "epoch": step / max(steps_per_epoch, 1),
                    }
                )
                if step % steps_per_epoch == 0:
                    print(
                        "[gradient-conflict-control] "
                        f"epoch={step // steps_per_epoch} "
                        f"scale={args.gradient_conflict_component_scale:g} "
                        f"raw_cos={float(conflict_component_diag['raw_cosine']):.6f} "
                        f"effective_cos={float(conflict_component_diag['effective_cosine']):.6f} "
                        f"applied={int(conflict_component_diag['applied'])}",
                        flush=True,
                    )
            if separate_rank_active:
                rank_probe_this_step = bool(
                    step % int(args.score_rank_gradient_probe_every) == 0
                )
                if rank_probe_this_step and not anchor_includes_rank:
                    weighted_rank_gradients = torch.autograd.grad(
                        scaler.scale(det_rank_objective),
                        rank_probe_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    weighted_rank_sq = sum(
                        float(gradient.detach().float().square().sum().cpu())
                        for gradient in weighted_rank_gradients
                        if gradient is not None
                    )
                    weighted_rank_norm = math.sqrt(max(weighted_rank_sq, 0.0))
                    legacy_effective_norm = float(
                        conflict_component_diag.get("effective_det_norm", float("nan"))
                    )
                    rank_gradient_probe_rows.append(
                        {
                            "step": float(step),
                            "epoch": float(step / max(steps_per_epoch, 1)),
                            "rank_weight": float(args.lambda_score_rank),
                            "weighted_rank_norm": weighted_rank_norm,
                            "legacy_effective_norm": legacy_effective_norm,
                            "rank_to_legacy_effective_ratio": (
                                weighted_rank_norm / legacy_effective_norm
                                if legacy_effective_norm > 0.0
                                else float("nan")
                            ),
                            "l_score_rank": float(det_diag["l_score_rank"]),
                            "score_rank_gap_mean": float(det_diag["score_rank_gap_mean"]),
                            "score_rank_active_fraction": float(det_diag["score_rank_active_fraction"]),
                        }
                    )
                    del weighted_rank_gradients
                if not anchor_includes_rank:
                    scaler.scale(det_rank_objective).backward()
        else:
            conflict_component_diag = {}
            scaler.scale(loss).backward()
        if native_bg_gradients:
            for parameter, gradient in zip(
                native_bg_parameters, native_bg_gradients
            ):
                if gradient is None:
                    continue
                if parameter.grad is None:
                    parameter.grad = gradient
                else:
                    parameter.grad.add_(gradient)
        if target_gate_warmup_active:
            for parameter in base_params:
                parameter.grad = None
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
        scaler.step(optimizer)
        scaler.update()
        model._guided_last_aggregate_prototype = None
        model._guided_last_aggregation_attention = ()

        row = {
            "step": float(step),
            "target_gate_warmup_active": float(target_gate_warmup_active),
            "loss": float(loss.detach().cpu()),
            "l_normal": float(l_normal.detach().cpu()),
            "l_normal_fp": float(l_normal_fp.detach().cpu()),
            "l_texture": float(l_texture.detach().cpu()),
            "texture_weight": float(texture_weight),
            "normal_prior_weight": float(normal_prior_weight),
            "hn_restore_weight": float(args.hn_oe_loss_weight),
            "native_bg_excess_weight": float(args.native_bg_excess_weight),
            "l_det": float(l_det.detach().cpu()),
            "target_gate_loss_weight": float(args.target_gate_loss_weight),
            "target_proto_invariance_weight": float(args.target_proto_invariance_weight),
            "target_proto_repulsion_weight": float(args.target_proto_repulsion_weight),
            "target_aggregation_attention_weight": float(args.target_aggregation_attention_weight),
            "target_read_attention_weight": float(args.target_read_attention_weight),
            "target_gate_lr": float(
                optimizer.param_groups[1]["lr"]
                if len(optimizer.param_groups) > 1
                else optimizer.param_groups[0]["lr"]
            ),
            "det_weight": float(det_weight),
            "score_rank_weight": float(args.lambda_score_rank),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": float(time.time() - started),
            **texture_diag,
            **normal_prior_diag,
            **hn_diag,
            **native_bg_diag,
            **target_background_diag,
            **det_diag,
            **target_gate_diag,
            **target_gate_normal_diag,
            **target_aux_diag,
            **normal_guided_diag,
            **normal_loss_diag,
            **guided_extra_diag,
            **{
                f"conflict_control_{key}": value
                for key, value in conflict_component_diag.items()
                if key not in {"step", "epoch", "groups"}
            },
        }
        history.append(row)
        epoch_losses.append(row["loss"])
        if args.progress_every > 0 and (step == 1 or step % args.progress_every == 0 or step == total_steps):
            print(
                f"[train] step={step}/{total_steps} loss={row['loss']:.5f} "
                f"normal={row['l_normal']:.5f} normal_fp={row['l_normal_fp']:.5f} "
                f"texture={row['l_texture']:.5f} tex_w={row['texture_weight']:.3f} "
                f"prior={row['l_normal_prior']:.5f} prior_w={row['normal_prior_weight']:.3f} "
                f"hn={row['l_hn_restore']:.5f} hn_active={row['hn_active_fraction']:.3f} "
                f"native_bg={row['l_native_bg_excess']:.5f} "
                f"native_bg_active={row['native_bg_excess_active_fraction']:.3f} "
                f"det={row['l_det']:.5f} det_w={row['det_weight']:.3f} "
                f"obj_bg={row['l_obj_bg']:.5f} rank={row['l_score_rank']:.5f} cover={row['l_score_cover']:.5f} "
                f"target_paste={row['target_bg_pastes_per_image']:.2f} "
                f"proto_drift={row['target_proto_drift']:.5f} "
                f"proto_dist={row['target_proto_distance_mean']:.5f} "
                f"agg_ratio={row['target_aggregation_attention_density_ratio']:.3f} "
                f"read_ratio={row['target_read_attention_ratio']:.3f} "
                f"elapsed={row['seconds']:.1f}s",
                flush=True,
            )
        if steps_per_epoch > 0 and step % steps_per_epoch == 0:
            epoch = step // steps_per_epoch
            epoch_loss = float(np.mean(epoch_losses))
            if epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                best_path = args.output_dir / "best.pth"
                best_metrics = {
                    "checkpoint": None if args.final_only_checkpoints else str(best_path),
                    "step": step,
                    "epoch": epoch,
                    "epoch_loss": epoch_loss,
                    "best_loss": best_epoch_loss,
                }
                (args.output_dir / "best_metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
                if args.final_only_checkpoints:
                    print(f"[best-loss] epoch={epoch} step={step} loss={epoch_loss:.6f} checkpoint_skipped", flush=True)
                else:
                    save_checkpoint(
                        best_path,
                        step,
                        include_training_state=True,
                        checkpoint_kind="best",
                        epoch=epoch,
                        epoch_loss=epoch_loss,
                        best_loss=best_epoch_loss,
                    )
                    print(f"[best] epoch={epoch} step={step} loss={epoch_loss:.6f} saved {best_path}", flush=True)
            else:
                print(
                    f"[epoch] epoch={epoch} step={step} loss={epoch_loss:.6f} best={best_epoch_loss:.6f}",
                    flush=True,
                )
            epoch_losses = []
            if not args.final_only_checkpoints:
                save_checkpoint(args.output_dir / "latest.pth", step, include_training_state=True, checkpoint_kind="latest", epoch=epoch, best_loss=best_epoch_loss)
        if not args.final_only_checkpoints and args.save_interval > 0 and step % args.save_interval == 0:
            ckpt_path = args.output_dir / f"iter_{step:06d}.pth"
            save_checkpoint(ckpt_path, step, include_training_state=True, checkpoint_kind="interval", best_loss=best_epoch_loss)
            print(f"[save] {ckpt_path}", flush=True)
        if not args.final_only_checkpoints and args.latest_interval > 0 and step % args.latest_interval == 0:
            latest_path = args.output_dir / "latest.pth"
            save_checkpoint(latest_path, step, include_training_state=True, checkpoint_kind="latest", best_loss=best_epoch_loss)
            print(f"[latest] {latest_path}", flush=True)
        if args.sequential_branch_backward:
            # The next normal forward is the peak-memory point. Drop references to
            # both completed branch graphs before it starts, then release cached
            # blocks so the large DINO attention allocation stays contiguous.
            normal_images = det_images = None
            det_boxes = det_valid = det_all_object_mask = None
            normal_en = normal_de = det_en = det_de = None
            normal_out = normal_out_full = normal_agg_prototype = None
            normal_enc_feature = normal_dec_feature = None
            det_enc_feature = det_dec_feature = det_objectness = det_score_map = None
            normal_objective = det_objective = det_legacy_objective = None
            det_rank_objective = det_rank_raw_objective = loss = None
            l_normal = l_guided_extra = l_normal_fp = None
            l_texture = l_normal_prior = l_det = l_det_legacy = None
            l_native_bg_excess = native_bg_objective = None
            l_det_rank_weighted = l_det_cover_weighted = None
            torch.cuda.empty_cache()
        run_inline_validation(step)

    if args.rank_gradient_audit_only:
        write_diagnostic_rows(rank_gradient_audit_rows, rank_gradient_audit_path)
        suggested_weights = np.asarray(
            [row["suggested_rank_weight"] for row in rank_gradient_audit_rows],
            dtype=np.float64,
        )
        finite_suggested = suggested_weights[np.isfinite(suggested_weights)]
        summary = {
            "checkpoint": str(resume_path),
            "audit_steps": len(rank_gradient_audit_rows),
            "background_mode": args.score_rank_background_mode,
            "target_rank_to_legacy_ratio": args.rank_gradient_audit_target_ratio,
            "suggested_rank_weight_median": (
                float(np.median(finite_suggested))
                if finite_suggested.size
                else float("nan")
            ),
            "suggested_rank_weight_q25": (
                float(np.quantile(finite_suggested, 0.25))
                if finite_suggested.size
                else float("nan")
            ),
            "suggested_rank_weight_q75": (
                float(np.quantile(finite_suggested, 0.75))
                if finite_suggested.size
                else float("nan")
            ),
        }
        for key in (
            "rank_normal_cosine",
            "rank_legacy_cosine",
            "rank_effective_joint_k0_ratio",
            "rank_removed_joint_k0_fraction",
            "rank_to_legacy_effective_ratio_unweighted",
            "score_rank_active_fraction",
            "score_rank_gap_mean",
        ):
            values = np.asarray([row[key] for row in rank_gradient_audit_rows], dtype=np.float64)
            finite_values = values[np.isfinite(values)]
            summary[f"{key}_mean"] = (
                float(finite_values.mean()) if finite_values.size else float("nan")
            )
        (args.output_dir / "rank_gradient_audit_summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
        )
        if gradient_probe is not None:
            gradient_probe.close()
        if conflict_component_scaler is not None:
            conflict_component_scaler.close()
        (args.output_dir / "ALL_DONE").touch()
        print(
            "[rank-gradient-audit] done "
            f"steps={len(rank_gradient_audit_rows)} "
            f"suggested_weight_median={summary['suggested_rank_weight_median']:.6f} "
            f"output={rank_gradient_audit_path}",
            flush=True,
        )
        return

    if loop_end_step < total_steps:
        phase_path = args.output_dir / f"phase_step_{loop_end_step:06d}.pth"
        save_checkpoint(
            phase_path,
            loop_end_step,
            include_training_state=True,
            checkpoint_kind="phase_boundary",
            epoch=loop_end_step // steps_per_epoch,
            best_loss=best_epoch_loss,
            total_steps=total_steps,
        )
        write_history(history, args.output_dir)
        (args.output_dir / "PHASE_DONE").touch()
        print(
            f"[phase-stop] step={loop_end_step}/{total_steps} saved {phase_path}",
            flush=True,
        )
        return

    best_path = args.output_dir / "best.pth"
    if not args.final_only_checkpoints and not best_path.exists() and epoch_losses:
        # Short OE fine-tunes may finish before the first full normal-data epoch.
        # In that case the final partial epoch is the only available best candidate.
        partial_epoch_loss = float(np.mean(epoch_losses))
        best_epoch_loss = partial_epoch_loss
        partial_epoch = total_steps // steps_per_epoch + 1 if steps_per_epoch > 0 else 1
        save_checkpoint(
            best_path,
            total_steps,
            include_training_state=True,
            checkpoint_kind="best_partial_epoch",
            epoch=partial_epoch,
            epoch_complete=False,
            steps_in_epoch=len(epoch_losses),
            epoch_loss=partial_epoch_loss,
            best_loss=best_epoch_loss,
        )
        best_metrics = {
            "checkpoint": str(best_path),
            "step": total_steps,
            "epoch": partial_epoch,
            "epoch_complete": False,
            "steps_in_epoch": len(epoch_losses),
            "epoch_loss": partial_epoch_loss,
            "best_loss": best_epoch_loss,
        }
        (args.output_dir / "best_metrics.json").write_text(
            json.dumps(best_metrics, indent=2), encoding="utf-8"
        )
        print(
            f"[best-partial] epoch={partial_epoch} step={total_steps} "
            f"steps_in_epoch={len(epoch_losses)} loss={partial_epoch_loss:.6f} saved {best_path}",
            flush=True,
        )

    final_path = args.output_dir / "model.pth"
    save_checkpoint(
        final_path,
        total_steps,
        include_training_state=not args.final_only_checkpoints,
        checkpoint_kind="final",
        best_loss=best_epoch_loss,
    )
    if not args.final_only_checkpoints:
        save_checkpoint(args.output_dir / "latest.pth", total_steps, include_training_state=True, checkpoint_kind="latest", best_loss=best_epoch_loss)
    if gradient_probe is not None:
        gradient_probe.close()
    if conflict_component_scaler is not None:
        conflict_component_scaler.close()
    if rank_gradient_probe_rows:
        write_diagnostic_rows(
            rank_gradient_probe_rows,
            args.output_dir / "rank_gradient_bypass.csv",
        )
    write_history(history, args.output_dir)
    (args.output_dir / "ALL_DONE").touch()
    print(f"[done] saved {final_path}", flush=True)


if __name__ == "__main__":
    main()
