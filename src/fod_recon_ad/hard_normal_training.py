from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class HardNormalReplay:
    """Map globally mined train-normal tokens back into deterministic crops."""

    image_name_to_view: dict[str, int]
    coordinates_by_view: dict[int, np.ndarray]
    p2_targets_by_view: dict[int, np.ndarray]
    weights_by_view: dict[int, np.ndarray]
    unique_tokens: int
    crop_occurrences: int

    @classmethod
    def load(
        cls,
        sidecar: Path | str,
        normal_crops_csv: Path | str,
        *,
        selector: str = "oe_raised",
    ) -> "HardNormalReplay":
        with Path(normal_crops_csv).open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        unique_images = sorted({str(row["image"]) for row in rows})
        image_name_to_view: dict[str, int] = {}
        for view, image in enumerate(unique_images):
            name = Path(image).name
            if name in image_name_to_view:
                raise ValueError(f"Ambiguous normal image basename: {name}")
            image_name_to_view[name] = view
        with np.load(sidecar, allow_pickle=False) as payload:
            if selector not in payload.files:
                raise KeyError(f"Hard-normal sidecar has no selector {selector!r}.")
            selected = np.asarray(payload[selector], dtype=bool)
            views = np.asarray(payload["views"], dtype=np.int64)[selected]
            global_y = np.asarray(payload["global_y"], dtype=np.int64)[selected]
            global_x = np.asarray(payload["global_x"], dtype=np.int64)[selected]
            targets = np.asarray(payload["p2_mean"], dtype=np.float32)[selected]
            support = np.asarray(payload["support"], dtype=np.float32)[selected]
        if not selected.any():
            raise RuntimeError(f"No tokens selected by {selector!r}.")
        coordinates_by_view = {}
        p2_targets_by_view = {}
        weights_by_view = {}
        for view in np.unique(views):
            keep = views == view
            coordinates_by_view[int(view)] = np.stack(
                (global_y[keep], global_x[keep]), axis=1
            ).astype(np.int64)
            p2_targets_by_view[int(view)] = targets[keep].astype(np.float32)
            weights_by_view[int(view)] = (1.0 / np.maximum(support[keep], 1.0)).astype(
                np.float32
            )
        return cls(
            image_name_to_view=image_name_to_view,
            coordinates_by_view=coordinates_by_view,
            p2_targets_by_view=p2_targets_by_view,
            weights_by_view=weights_by_view,
            unique_tokens=int(selected.sum()),
            crop_occurrences=int(support.sum()),
        )

    def batch_targets(
        self,
        crop_specs: Sequence[str],
        *,
        token_side: int,
        physical_crop_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.zeros(
            (len(crop_specs), 1, token_side, token_side),
            dtype=torch.float32,
            device=device,
        )
        targets = torch.zeros_like(weights)
        for batch_index, crop_spec in enumerate(crop_specs):
            image_path, x_text, y_text = str(crop_spec).rsplit(":", 2)
            image_name = Path(image_path).name
            if image_name not in self.image_name_to_view:
                continue
            view = self.image_name_to_view[image_name]
            coordinates = self.coordinates_by_view.get(view)
            if coordinates is None:
                continue
            x_offset = int(round(int(x_text) * token_side / physical_crop_size))
            y_offset = int(round(int(y_text) * token_side / physical_crop_size))
            local_y = coordinates[:, 0] - y_offset
            local_x = coordinates[:, 1] - x_offset
            keep = (
                (local_y >= 0)
                & (local_y < token_side)
                & (local_x >= 0)
                & (local_x < token_side)
            )
            if not keep.any():
                continue
            yy = torch.from_numpy(local_y[keep]).to(device=device, dtype=torch.long)
            xx = torch.from_numpy(local_x[keep]).to(device=device, dtype=torch.long)
            value = torch.from_numpy(self.weights_by_view[view][keep]).to(device=device)
            target = torch.from_numpy(self.p2_targets_by_view[view][keep]).to(device=device)
            weights[batch_index, 0, yy, xx] = value
            targets[batch_index, 0, yy, xx] = target
        return weights, targets


def causal_restore_loss(
    scores: torch.Tensor,
    weights: torch.Tensor,
    p2_targets: torch.Tensor,
    *,
    margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if scores.shape != weights.shape or scores.shape != p2_targets.shape:
        raise ValueError(
            f"HN replay shape mismatch: {scores.shape}, {weights.shape}, {p2_targets.shape}"
        )
    selected = weights > 0
    denominator = weights.sum().clamp_min(1e-8)
    excess = (scores - p2_targets - float(margin)).clamp_min(0.0)
    loss = (weights * excess).sum() / denominator
    if selected.any():
        selected_weight = weights[selected]
        selected_denominator = selected_weight.sum().clamp_min(1e-8)
        score_mean = float(
            (scores[selected] * selected_weight).sum().detach().cpu() / selected_denominator.cpu()
        )
        target_mean = float(
            (p2_targets[selected] * selected_weight).sum().detach().cpu()
            / selected_denominator.cpu()
        )
        active_fraction = float((excess[selected] > 0).float().mean().detach().cpu())
        occurrences = int(selected.sum().detach().cpu())
    else:
        score_mean = 0.0
        target_mean = 0.0
        active_fraction = 0.0
        occurrences = 0
    return loss, {
        "l_hn_restore": float(loss.detach().cpu()),
        "hn_score_mean": score_mean,
        "hn_p2_target_mean": target_mean,
        "hn_active_fraction": active_fraction,
        "hn_batch_occurrences": float(occurrences),
        "hn_batch_weight": float(weights.sum().detach().cpu()),
    }
