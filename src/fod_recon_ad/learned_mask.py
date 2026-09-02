from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class NormalFolderDataset(Dataset):
    def __init__(self, root: Path, image_size: int) -> None:
        self.paths = [
            path
            for path in sorted(root.iterdir() if root.exists() else [])
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS and not path.name.startswith(".")
        ]
        if not self.paths:
            raise RuntimeError(f"No normal images found in {root}")
        self.transform = image_transform(image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        image_path = self.paths[index]
        return self.transform(Image.open(image_path).convert("RGB")), str(image_path)


class SmallObjectManifestDataset(Dataset):
    def __init__(self, manifest_path: Path, image_size: int, max_images: int, max_boxes: int, seed: int) -> None:
        rows: List[Dict[str, object]] = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if not rows:
            raise RuntimeError(f"No rows found in {manifest_path}")
        rng = random.Random(seed)
        if max_images > 0 and len(rows) > max_images:
            rows = sorted(rng.sample(rows, max_images), key=lambda row: str(row["image_path"]))
        self.rows = rows
        self.image_size = int(image_size)
        self.max_boxes = int(max_boxes)
        self.seed = int(seed)
        self.transform = image_transform(image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        row = self.rows[index]
        image_path = Path(str(row["image_path"]))
        width = float(row["width"])
        height = float(row["height"])
        boxes = list(row["boxes"])  # type: ignore[arg-type]
        if self.max_boxes > 0 and len(boxes) > self.max_boxes:
            boxes = random.Random(self.seed + index).sample(boxes, self.max_boxes)
        padded = torch.zeros((self.max_boxes, 4), dtype=torch.float32)
        valid = torch.zeros((self.max_boxes,), dtype=torch.bool)
        scale_x = self.image_size / max(width, 1.0)
        scale_y = self.image_size / max(height, 1.0)
        for box_idx, box in enumerate(boxes[: self.max_boxes]):
            x1, y1, x2, y2 = [float(value) for value in box["bbox"]]  # type: ignore[index]
            padded[box_idx] = torch.tensor([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y])
            valid[box_idx] = True
        return self.transform(Image.open(image_path).convert("RGB")), padded, valid, str(image_path)


def image_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def freeze_module(module: torch.nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def extract_fused_spatial_tokens(model: torch.nn.Module, images: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Return fused encoder spatial tokens in the same layout used by adaptive masking."""

    required = ("encoder", "target_layers", "fuse_feature")
    if not all(hasattr(model, name) for name in required):
        raise ValueError("Learned mask proposal currently expects a ViT reconstruction model.")
    with torch.no_grad():
        x = model.encoder.prepare_tokens(images)
        en_list: List[torch.Tensor] = []
        for i, blk in enumerate(model.encoder.blocks):
            if i <= model.target_layers[-1]:
                x = blk(x)
            else:
                continue
            if i in model.target_layers:
                en_list.append(x)
        if not en_list:
            raise RuntimeError("No encoder target features were captured.")
        start = 1 + model.encoder.num_register_tokens
        side = int(math.sqrt(en_list[0].shape[1] - start))
        if side * side != en_list[0].shape[1] - start:
            raise RuntimeError(f"Cannot infer square token grid from shape {tuple(en_list[0].shape)}.")
        if getattr(model, "remove_class_token", False):
            en_list = [item[:, start:, :] for item in en_list]
            start = 0
        fused = model.fuse_feature(en_list)
        spatial_tokens = fused[:, start:, :].detach()
    return spatial_tokens, side


def tokens_to_map(tokens: torch.Tensor, side: int) -> torch.Tensor:
    batch, token_count, channels = tokens.shape
    if token_count != side * side:
        raise RuntimeError(f"Expected {side * side} tokens, got {token_count}.")
    return tokens.transpose(1, 2).reshape(batch, channels, side, side).contiguous()


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


def token_low_level_maps(images: torch.Tensor, token_h: int, token_w: int) -> Tuple[torch.Tensor, torch.Tensor]:
    lum = normalized_luminance(images)
    blur = F.avg_pool2d(lum, kernel_size=7, stride=1, padding=3)
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


def abstract_objectness_target(
    images: torch.Tensor,
    spatial_tokens: torch.Tensor,
    side: int,
    freq_weight: float = 0.25,
    edge_weight: float = 0.25,
    feature_weight: float = 0.50,
    ring_radius: int = 2,
    smooth_kernel: int = 3,
) -> torch.Tensor:
    fmap = F.normalize(tokens_to_map(spatial_tokens, side).detach().float(), dim=1)
    ring = F.normalize(local_ring_mean(fmap, ring_radius), dim=1)
    dino_contrast = (1.0 - (fmap * ring).sum(dim=1, keepdim=True)).clamp_min(0.0)
    freq, edge = token_low_level_maps(images, side, side)
    freq_contrast = (freq - local_ring_mean(freq, ring_radius)).abs()
    edge_contrast = (edge - local_ring_mean(edge, ring_radius)).abs()
    total = max(float(freq_weight + edge_weight + feature_weight), 1e-6)
    target = (
        freq_weight * robust_norm_map(freq_contrast)
        + edge_weight * robust_norm_map(edge_contrast)
        + feature_weight * robust_norm_map(dino_contrast)
    ) / total
    kernel = int(smooth_kernel)
    if kernel > 1:
        if kernel % 2 == 0:
            kernel += 1
        target = F.avg_pool2d(target, kernel_size=kernel, stride=1, padding=kernel // 2)
    return robust_norm_map(target).detach()


def boxes_to_token_target(
    boxes: torch.Tensor,
    valid: torch.Tensor,
    token_h: int,
    token_w: int,
    image_size: int,
    dilation: int = 0,
) -> torch.Tensor:
    mask = torch.zeros((boxes.shape[0], 1, token_h, token_w), dtype=torch.float32, device=boxes.device)
    token_w_px = float(image_size) / float(token_w)
    token_h_px = float(image_size) / float(token_h)
    for b in range(boxes.shape[0]):
        for box_idx in torch.nonzero(valid[b], as_tuple=False).flatten().tolist():
            x1, y1, x2, y2 = boxes[b, box_idx].tolist()
            tx1 = max(0, min(token_w, int(math.floor(x1 / token_w_px))))
            ty1 = max(0, min(token_h, int(math.floor(y1 / token_h_px))))
            tx2 = max(0, min(token_w, int(math.ceil(x2 / token_w_px))))
            ty2 = max(0, min(token_h, int(math.ceil(y2 / token_h_px))))
            if tx2 > tx1 and ty2 > ty1:
                mask[b, :, ty1:ty2, tx1:tx2] = 1.0
    if dilation > 0:
        kernel = dilation * 2 + 1
        mask = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=dilation)
    return mask.clamp(0.0, 1.0)


def focal_bce_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.75, gamma: float = 2.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    pt = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (alpha_t * (1.0 - pt).pow(gamma) * bce).mean()


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    flat_prob = prob.flatten(1)
    flat_target = target.flatten(1)
    inter = (flat_prob * flat_target).sum(dim=1)
    denom = flat_prob.sum(dim=1) + flat_target.sum(dim=1)
    return (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()


def budget_loss(probs: torch.Tensor, target_ratio: float, mode: str = "upper") -> torch.Tensor:
    ratio = probs.flatten(1).mean(dim=1)
    target = probs.new_full(ratio.shape, float(target_ratio))
    if mode == "match":
        return (ratio - target).pow(2).mean()
    if mode == "upper":
        return F.relu(ratio - target).pow(2).mean()
    raise ValueError(f"Unsupported budget mode: {mode}")


def topk_mask_from_logits(logits: torch.Tensor, ratio: float, dilation: int = 0) -> torch.Tensor:
    batch, _, height, width = logits.shape
    flat = logits.flatten(1)
    total = flat.shape[1]
    k = max(1, int(round(total * float(ratio))))
    selected = torch.topk(flat, k=k, dim=1, largest=True).indices
    mask = torch.zeros((batch, total), dtype=torch.bool, device=logits.device)
    mask.scatter_(1, selected, True)
    mask = mask.view(batch, 1, height, width)
    if dilation > 0:
        kernel = int(dilation) * 2 + 1
        mask = F.max_pool2d(mask.float(), kernel_size=kernel, stride=1, padding=int(dilation)) > 0
    return mask


def score_separation_stats(
    logits: torch.Tensor,
    target: torch.Tensor,
    topk_ratio: float,
    topk_dilation: int = 0,
    prefix: str = "",
) -> Dict[str, float]:
    target_bool = target > 0.5
    bg_bool = ~target_bool
    probs = torch.sigmoid(logits)
    selected = topk_mask_from_logits(logits, topk_ratio, dilation=topk_dilation)
    selected_f = selected.float()
    target_f = target_bool.float()
    overlap = (selected_f * target_f).sum()
    target_count = target_f.sum().clamp_min(1.0)
    selected_count = selected_f.sum().clamp_min(1.0)
    out: Dict[str, float] = {
        f"{prefix}selected_ratio": float(selected_f.mean().detach().cpu()),
        f"{prefix}topk_target_recall": float((overlap / target_count).detach().cpu()),
        f"{prefix}topk_precision": float((overlap / selected_count).detach().cpu()),
    }
    if int(target_bool.sum().detach().cpu()) > 0:
        obj_logits = logits[target_bool]
        obj_probs = probs[target_bool]
        out[f"{prefix}obj_logit_mean"] = float(obj_logits.mean().detach().cpu())
        out[f"{prefix}obj_prob_mean"] = float(obj_probs.mean().detach().cpu())
    else:
        out[f"{prefix}obj_logit_mean"] = 0.0
        out[f"{prefix}obj_prob_mean"] = 0.0
    if int(bg_bool.sum().detach().cpu()) > 0:
        bg_logits = logits[bg_bool]
        bg_probs = probs[bg_bool]
        out[f"{prefix}bg_logit_mean"] = float(bg_logits.mean().detach().cpu())
        out[f"{prefix}bg_prob_mean"] = float(bg_probs.mean().detach().cpu())
    else:
        out[f"{prefix}bg_logit_mean"] = 0.0
        out[f"{prefix}bg_prob_mean"] = 0.0
    out[f"{prefix}logit_margin"] = out[f"{prefix}obj_logit_mean"] - out[f"{prefix}bg_logit_mean"]
    out[f"{prefix}prob_margin"] = out[f"{prefix}obj_prob_mean"] - out[f"{prefix}bg_prob_mean"]
    return out


def normal_topk_stats(logits: torch.Tensor, topk_ratio: float, topk_dilation: int = 0, prefix: str = "") -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    selected = topk_mask_from_logits(logits, topk_ratio, dilation=topk_dilation)
    return {
        f"{prefix}logit_mean": float(logits.detach().mean().cpu()),
        f"{prefix}logit_max": float(logits.detach().amax().cpu()),
        f"{prefix}prob_mean": float(probs.detach().mean().cpu()),
        f"{prefix}prob_max": float(probs.detach().amax().cpu()),
        f"{prefix}topk_logit_mean": float(logits[selected].detach().mean().cpu()),
        f"{prefix}topk_prob_mean": float(probs[selected].detach().mean().cpu()),
        f"{prefix}selected_ratio": float(selected.float().mean().detach().cpu()),
    }


def tv_loss(probs: torch.Tensor) -> torch.Tensor:
    loss = probs.new_tensor(0.0)
    if probs.shape[-1] > 1:
        loss = loss + (probs[:, :, :, 1:] - probs[:, :, :, :-1]).abs().mean()
    if probs.shape[-2] > 1:
        loss = loss + (probs[:, :, 1:, :] - probs[:, :, :-1, :]).abs().mean()
    return loss


def planner_normal_fp_loss(probs: torch.Tensor, raw_residual: torch.Tensor) -> torch.Tensor:
    residual = robust_norm_map(raw_residual.detach())
    return (probs * residual).flatten(1).mean(dim=1).mean()


def reconstruction_residual_map(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        en, de = model(images)[:2]
        residuals = [
            (1.0 - F.cosine_similarity(target.detach(), pred, dim=1, eps=1e-6).unsqueeze(1)).clamp_min(0.0)
            for target, pred in zip(en, de)
        ]
    return torch.stack(residuals, dim=0).mean(dim=0)


def save_planner_checkpoint(path: Path, planner: torch.nn.Module, **metadata: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"planner": planner.state_dict(), "metadata": metadata}, path)


def load_planner_state(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location=device)
    if isinstance(state, dict):
        if "planner" in state:
            return state["planner"]
        if "adaptive_mask_planner" in state:
            return state["adaptive_mask_planner"]
        if "model" in state:
            model_state = state["model"]
            prefix = "adaptive_mask_planner."
            return {key[len(prefix) :]: value for key, value in model_state.items() if key.startswith(prefix)}
    return state
