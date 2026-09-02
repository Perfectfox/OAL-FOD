from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import torch
import torch.nn as nn


def gradient_triplet_diagnostics(
    normal_gradients: Sequence[torch.Tensor | None],
    legacy_gradients: Sequence[torch.Tensor | None],
    rank_gradients: Sequence[torch.Tensor | None],
    *,
    target_rank_to_legacy_ratio: float = 0.15,
) -> dict[str, float]:
    """Summarize normal, legacy-OE and unweighted rank gradients.

    The function uses global vector geometry without concatenating parameter
    tensors.  ``rank_effective_joint_k0_ratio`` measures how much of an
    unweighted rank update survives when it is merged with legacy OE *before*
    the normal-conflict projection.  ``suggested_rank_weight`` scales the raw
    rank norm to the requested fraction of projected legacy OE.
    """

    if not (
        len(normal_gradients) == len(legacy_gradients) == len(rank_gradients)
    ):
        raise ValueError("Gradient sequences must have equal length.")
    if target_rank_to_legacy_ratio <= 0.0 or not math.isfinite(
        target_rank_to_legacy_ratio
    ):
        raise ValueError("target_rank_to_legacy_ratio must be finite and positive.")

    n_sq = l_sq = r_sq = 0.0
    dot_nl = dot_nr = dot_lr = 0.0
    shared = 0
    for normal_gradient, legacy_gradient, rank_gradient in zip(
        normal_gradients, legacy_gradients, rank_gradients
    ):
        if normal_gradient is None:
            continue
        normal = normal_gradient.detach().float()
        legacy = (
            torch.zeros_like(normal)
            if legacy_gradient is None
            else legacy_gradient.detach().float()
        )
        rank = (
            torch.zeros_like(normal)
            if rank_gradient is None
            else rank_gradient.detach().float()
        )
        n_sq += float(normal.square().sum().cpu())
        l_sq += float(legacy.square().sum().cpu())
        r_sq += float(rank.square().sum().cpu())
        dot_nl += float((normal * legacy).sum().cpu())
        dot_nr += float((normal * rank).sum().cpu())
        dot_lr += float((legacy * rank).sum().cpu())
        shared += 1

    n_norm = math.sqrt(max(n_sq, 0.0))
    l_norm = math.sqrt(max(l_sq, 0.0))
    r_norm = math.sqrt(max(r_sq, 0.0))

    def cosine(dot: float, left: float, right: float) -> float:
        denom = left * right
        return dot / denom if denom > 0.0 else float("nan")

    def projection_coefficient(dot: float) -> float:
        return dot / n_sq if dot < 0.0 and n_sq > 0.0 else 0.0

    legacy_coefficient = projection_coefficient(dot_nl)
    legacy_effective_sq = max(
        l_sq - 2.0 * legacy_coefficient * dot_nl + legacy_coefficient**2 * n_sq,
        0.0,
    )
    legacy_effective_norm = math.sqrt(legacy_effective_sq)

    joint_coefficient = projection_coefficient(dot_nl + dot_nr)
    coefficient_delta = joint_coefficient - legacy_coefficient
    # P_N(L + R) - P_N(L) = R - coefficient_delta * N.
    rank_joint_effective_sq = max(
        r_sq
        - 2.0 * coefficient_delta * dot_nr
        + coefficient_delta**2 * n_sq,
        0.0,
    )
    rank_joint_effective_norm = math.sqrt(rank_joint_effective_sq)
    removed_norm = abs(coefficient_delta) * n_norm
    suggested_weight = (
        target_rank_to_legacy_ratio * legacy_effective_norm / r_norm
        if r_norm > 0.0
        else float("nan")
    )

    return {
        "shared_tensors": float(shared),
        "normal_norm": n_norm,
        "legacy_raw_norm": l_norm,
        "legacy_effective_k0_norm": legacy_effective_norm,
        "rank_raw_norm": r_norm,
        "rank_normal_cosine": cosine(dot_nr, n_norm, r_norm),
        "rank_legacy_cosine": cosine(dot_lr, l_norm, r_norm),
        "legacy_normal_cosine": cosine(dot_nl, n_norm, l_norm),
        "rank_conflict_fraction": max(-cosine(dot_nr, n_norm, r_norm), 0.0)
        if n_norm > 0.0 and r_norm > 0.0
        else float("nan"),
        "rank_effective_joint_k0_norm": rank_joint_effective_norm,
        "rank_effective_joint_k0_ratio": (
            rank_joint_effective_norm / r_norm if r_norm > 0.0 else float("nan")
        ),
        "rank_removed_joint_k0_fraction": (
            removed_norm / r_norm if r_norm > 0.0 else float("nan")
        ),
        "rank_to_legacy_effective_ratio_unweighted": (
            r_norm / legacy_effective_norm
            if legacy_effective_norm > 0.0
            else float("nan")
        ),
        "target_rank_to_legacy_ratio": float(target_rank_to_legacy_ratio),
        "suggested_rank_weight": suggested_weight,
    }


def coarse_parameter_group(name: str) -> str:
    """Map reconstruction parameters to stable, interpretable probe groups."""

    root = name.split(".", 1)[0]
    if root in {"prototype_token", "bottleneck", "aggregation", "decoder"}:
        return root
    if "guided" in name or "prior_head" in name:
        return "guided"
    return root


class GradientConflictProbe:
    """Measure two sequential backward passes without modifying either gradient.

    The probe assumes ``normal.backward()`` has completed. ``begin_det`` records
    normal-gradient norms and enables hooks for the following detection backward.
    A leaf hook observes the incoming detection gradient before it is accumulated
    into ``parameter.grad``, which still contains the normal gradient at that time.
    """

    FIELDNAMES = (
        "step",
        "epoch",
        "group",
        "cosine_shared",
        "cosine_full",
        "normal_full_norm",
        "normal_shared_norm",
        "det_norm",
        "det_to_normal_shared_ratio",
        "det_to_normal_full_ratio",
        "tensor_conflict_fraction",
        "element_conflict_fraction",
        "normal_tensors",
        "det_tensors",
        "shared_tensors",
        "shared_elements",
        "normal_loss",
        "det_loss",
        "det_weight",
        "learning_rate",
    )

    def __init__(
        self,
        model: nn.Module,
        trainable_parameters: Iterable[nn.Parameter],
        output_path: Path,
        *,
        append: bool = False,
    ) -> None:
        trainable_ids = {
            id(parameter)
            for parameter in trainable_parameters
            if parameter.requires_grad
        }
        self.entries: list[tuple[str, str, nn.Parameter]] = []
        seen: set[int] = set()
        for name, parameter in model.named_parameters():
            if id(parameter) not in trainable_ids or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            self.entries.append((name, coarse_parameter_group(name), parameter))
        if len(seen) != len(trainable_ids):
            missing = len(trainable_ids) - len(seen)
            raise RuntimeError(f"Gradient probe could not name {missing} trainable parameters.")

        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append and self.output_path.exists() else "w"
        self._handle = self.output_path.open(mode, encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.FIELDNAMES)
        if mode == "w" or self.output_path.stat().st_size == 0:
            self._writer.writeheader()
            self._handle.flush()

        self._active = False
        self._normal_full: Dict[str, list[torch.Tensor]] = defaultdict(list)
        self._terms: Dict[str, Dict[str, list[torch.Tensor]]] = {}
        self._normal_tensor_counts: Dict[str, int] = defaultdict(int)
        self._det_tensor_counts: Dict[str, int] = defaultdict(int)
        self._shared_tensor_counts: Dict[str, int] = defaultdict(int)
        self._tensor_conflict_counts: Dict[str, list[torch.Tensor]] = defaultdict(list)
        self._shared_element_counts: Dict[str, list[torch.Tensor]] = defaultdict(list)
        self._element_conflict_counts: Dict[str, list[torch.Tensor]] = defaultdict(list)
        self._hooks = [
            parameter.register_hook(self._make_hook(group, parameter))
            for _, group, parameter in self.entries
        ]

    @property
    def groups(self) -> list[str]:
        return sorted({group for _, group, _ in self.entries})

    @staticmethod
    def _empty_terms() -> Dict[str, list[torch.Tensor]]:
        return {"dot": [], "normal_shared_sq": [], "det_sq": []}

    def _make_hook(self, group: str, parameter: nn.Parameter):
        def hook(det_gradient: torch.Tensor) -> torch.Tensor:
            if not self._active:
                return det_gradient
            det = det_gradient.detach().float()
            self._det_tensor_counts[group] += 1
            self._terms[group]["det_sq"].append(det.square().sum())
            normal_gradient = parameter.grad
            if normal_gradient is None:
                return det_gradient
            normal = normal_gradient.detach().float()
            product = normal * det
            self._shared_tensor_counts[group] += 1
            self._terms[group]["dot"].append(product.sum())
            self._terms[group]["normal_shared_sq"].append(normal.square().sum())
            self._tensor_conflict_counts[group].append((product.sum() < 0).to(torch.float32))
            shared = product != 0
            self._shared_element_counts[group].append(shared.sum().to(torch.float32))
            self._element_conflict_counts[group].append(
                ((product < 0) & shared).sum().to(torch.float32)
            )
            return det_gradient

        return hook

    def begin_det(self) -> None:
        if self._active:
            raise RuntimeError("Gradient probe is already active.")
        self._normal_full = defaultdict(list)
        self._terms = {group: self._empty_terms() for group in self.groups}
        self._normal_tensor_counts = defaultdict(int)
        self._det_tensor_counts = defaultdict(int)
        self._shared_tensor_counts = defaultdict(int)
        self._tensor_conflict_counts = defaultdict(list)
        self._shared_element_counts = defaultdict(list)
        self._element_conflict_counts = defaultdict(list)
        for _, group, parameter in self.entries:
            if parameter.grad is None:
                continue
            self._normal_full[group].append(parameter.grad.detach().float().square().sum())
            self._normal_tensor_counts[group] += 1
        self._active = True

    @staticmethod
    def _sum(values: list[torch.Tensor]) -> float:
        if not values:
            return 0.0
        return float(torch.stack(values).sum().detach().cpu())

    def _aggregate(self, groups: Iterable[str]) -> Dict[str, float | int]:
        selected = list(groups)
        normal_full_sq = sum(self._sum(self._normal_full[group]) for group in selected)
        normal_shared_sq = sum(
            self._sum(self._terms[group]["normal_shared_sq"]) for group in selected
        )
        det_sq = sum(self._sum(self._terms[group]["det_sq"]) for group in selected)
        dot = sum(self._sum(self._terms[group]["dot"]) for group in selected)
        normal_full_norm = math.sqrt(max(normal_full_sq, 0.0))
        normal_shared_norm = math.sqrt(max(normal_shared_sq, 0.0))
        det_norm = math.sqrt(max(det_sq, 0.0))
        shared_denom = normal_shared_norm * det_norm
        full_denom = normal_full_norm * det_norm
        tensor_conflicts = sum(
            self._sum(self._tensor_conflict_counts[group]) for group in selected
        )
        shared_tensors = sum(self._shared_tensor_counts[group] for group in selected)
        element_conflicts = sum(
            self._sum(self._element_conflict_counts[group]) for group in selected
        )
        shared_elements = sum(
            self._sum(self._shared_element_counts[group]) for group in selected
        )
        return {
            "cosine_shared": dot / shared_denom if shared_denom > 0.0 else float("nan"),
            "cosine_full": dot / full_denom if full_denom > 0.0 else float("nan"),
            "normal_full_norm": normal_full_norm,
            "normal_shared_norm": normal_shared_norm,
            "det_norm": det_norm,
            "det_to_normal_shared_ratio": det_norm / normal_shared_norm
            if normal_shared_norm > 0.0
            else float("nan"),
            "det_to_normal_full_ratio": det_norm / normal_full_norm
            if normal_full_norm > 0.0
            else float("nan"),
            "tensor_conflict_fraction": tensor_conflicts / shared_tensors
            if shared_tensors > 0
            else float("nan"),
            "element_conflict_fraction": element_conflicts / shared_elements
            if shared_elements > 0.0
            else float("nan"),
            "normal_tensors": sum(self._normal_tensor_counts[group] for group in selected),
            "det_tensors": sum(self._det_tensor_counts[group] for group in selected),
            "shared_tensors": shared_tensors,
            "shared_elements": int(shared_elements),
        }

    def end_det(self, metadata: Mapping[str, float | int]) -> list[dict[str, object]]:
        if not self._active:
            raise RuntimeError("Gradient probe is not active.")
        self._active = False
        rows: list[dict[str, object]] = []
        group_specs = [("overall", self.groups)] + [
            (group, [group]) for group in self.groups
        ]
        for label, groups in group_specs:
            row = {
                "step": int(metadata["step"]),
                "epoch": float(metadata["epoch"]),
                "group": label,
                **self._aggregate(groups),
                "normal_loss": float(metadata["normal_loss"]),
                "det_loss": float(metadata["det_loss"]),
                "det_weight": float(metadata["det_weight"]),
                "learning_rate": float(metadata["learning_rate"]),
            }
            self._writer.writerow(row)
            rows.append(row)
        self._handle.flush()
        return rows

    def close(self) -> None:
        self._active = False
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        if not self._handle.closed:
            self._handle.close()


class GradientConflictComponentScaler:
    """Scale only an auxiliary gradient component opposing an anchor gradient.

    For selected parameter groups, decompose the detection gradient as

    ``g_det = g_orthogonal + g_conflict``

    where ``g_conflict`` is the projection onto the anchor-gradient direction
    when their global dot product is negative.  The accumulated gradient is
    then changed from ``g_anchor + g_aux`` to

    ``g_anchor + g_orthogonal + conflict_scale * g_conflict``.

    With ``budget_beta`` enabled, the scale is resolved per step as

    ``min(1, beta * ||g_anchor||^2 / -<g_anchor, g_aux>)``.

    This guarantees the first-order anchor descent budget
    ``<g_anchor, g_total> >= (1 - beta) * ||g_anchor||^2``.  A beta of zero
    is the legacy full projection, while beta one only prevents the auxiliary
    gradient from reversing the anchor descent direction.

    ``begin_det`` must be called after the normal backward and ``end_det`` after
    the sequential detection backward, before gradient clipping/optimizer step.
    Both gradients may be AMP-scaled; the common scale cancels in the projection.
    """

    FIELDNAMES = (
        "step",
        "epoch",
        "groups",
        "anchor_label",
        "conflict_scale",
        "budget_beta",
        "resolved_conflict_scale",
        "applied",
        "raw_cosine",
        "effective_cosine",
        "normal_norm",
        "raw_det_norm",
        "effective_det_norm",
        "conflict_component_norm",
        "conflict_component_fraction",
        "shared_tensors",
    )

    def __init__(
        self,
        model: nn.Module,
        trainable_parameters: Iterable[nn.Parameter],
        *,
        conflict_scale: float,
        budget_beta: float | None = None,
        anchor_label: str = "normal",
        groups: Sequence[str] = ("decoder",),
        output_path: Path | None = None,
        append: bool = False,
    ) -> None:
        if conflict_scale < 0.0 or not math.isfinite(conflict_scale):
            raise ValueError("conflict_scale must be finite and non-negative.")
        if budget_beta is not None and (
            budget_beta < 0.0
            or budget_beta > 1.0
            or not math.isfinite(budget_beta)
        ):
            raise ValueError("budget_beta must be finite and in [0,1].")
        selected_groups = tuple(dict.fromkeys(str(group) for group in groups))
        if not selected_groups:
            raise ValueError("At least one gradient-conflict group is required.")

        trainable_ids = {
            id(parameter)
            for parameter in trainable_parameters
            if parameter.requires_grad
        }
        self.entries: list[tuple[str, str, nn.Parameter]] = []
        seen: set[int] = set()
        for name, parameter in model.named_parameters():
            group = coarse_parameter_group(name)
            if (
                id(parameter) not in trainable_ids
                or id(parameter) in seen
                or group not in selected_groups
            ):
                continue
            seen.add(id(parameter))
            self.entries.append((name, group, parameter))
        if not self.entries:
            raise RuntimeError(
                "No trainable parameters matched gradient-conflict groups "
                f"{selected_groups}."
            )

        self.conflict_scale = float(conflict_scale)
        self.budget_beta = (
            None if budget_beta is None else float(budget_beta)
        )
        self.anchor_label = str(anchor_label)
        self.groups = selected_groups
        self._normal_gradients: list[tuple[nn.Parameter, torch.Tensor]] = []
        self._active = False
        self._handle = None
        self._writer = None
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append and path.exists() else "w"
            self._handle = path.open(mode, encoding="utf-8", newline="")
            self._writer = csv.DictWriter(self._handle, fieldnames=self.FIELDNAMES)
            if mode == "w" or path.stat().st_size == 0:
                self._writer.writeheader()
                self._handle.flush()

    def begin_det(self) -> None:
        if self._active:
            raise RuntimeError("Gradient conflict component scaler is already active.")
        self._normal_gradients = [
            (parameter, parameter.grad.detach().clone())
            for _, _, parameter in self.entries
            if parameter.grad is not None
        ]
        self._active = True

    def end_det(self, metadata: Mapping[str, float | int]) -> dict[str, object]:
        if not self._active:
            raise RuntimeError("Gradient conflict component scaler is not active.")
        self._active = False
        try:
            terms: list[tuple[nn.Parameter, torch.Tensor]] = []
            normal_sq_terms: list[torch.Tensor] = []
            det_sq_terms: list[torch.Tensor] = []
            dot_terms: list[torch.Tensor] = []
            for parameter, normal_gradient in self._normal_gradients:
                if parameter.grad is None:
                    continue
                det_gradient = parameter.grad.detach() - normal_gradient
                normal = normal_gradient.float()
                det = det_gradient.float()
                terms.append((parameter, normal_gradient))
                normal_sq_terms.append(normal.square().sum())
                det_sq_terms.append(det.square().sum())
                dot_terms.append((normal * det).sum())

            normal_sq = GradientConflictProbe._sum(normal_sq_terms)
            det_sq = GradientConflictProbe._sum(det_sq_terms)
            dot = GradientConflictProbe._sum(dot_terms)
            normal_norm = math.sqrt(max(normal_sq, 0.0))
            raw_det_norm = math.sqrt(max(det_sq, 0.0))
            denom = normal_norm * raw_det_norm
            raw_cosine = dot / denom if denom > 0.0 else float("nan")
            is_conflict = dot < 0.0 and normal_sq > 0.0 and det_sq > 0.0

            resolved_conflict_scale = self.conflict_scale
            if is_conflict and self.budget_beta is not None:
                resolved_conflict_scale = min(
                    1.0,
                    self.budget_beta * normal_sq / max(-dot, torch.finfo(torch.float32).eps),
                )

            conflict_sq = dot * dot / normal_sq if is_conflict else 0.0
            orthogonal_sq = max(det_sq - conflict_sq, 0.0)
            if is_conflict:
                projection_coefficient = dot / normal_sq
                adjustment = (resolved_conflict_scale - 1.0) * projection_coefficient
                if adjustment != 0.0:
                    with torch.no_grad():
                        for parameter, normal_gradient in terms:
                            parameter.grad.add_(normal_gradient, alpha=adjustment)
                effective_dot = resolved_conflict_scale * dot
                effective_det_sq = (
                    orthogonal_sq
                    + resolved_conflict_scale * resolved_conflict_scale * conflict_sq
                )
            else:
                effective_dot = dot
                effective_det_sq = det_sq

            effective_det_norm = math.sqrt(max(effective_det_sq, 0.0))
            effective_denom = normal_norm * effective_det_norm
            effective_cosine = (
                effective_dot / effective_denom
                if effective_denom > 0.0
                else float("nan")
            )
            conflict_norm = math.sqrt(max(conflict_sq, 0.0))
            row: dict[str, object] = {
                "step": int(metadata["step"]),
                "epoch": float(metadata["epoch"]),
                "groups": "+".join(self.groups),
                "anchor_label": self.anchor_label,
                "conflict_scale": self.conflict_scale,
                "budget_beta": (
                    "" if self.budget_beta is None else self.budget_beta
                ),
                "resolved_conflict_scale": resolved_conflict_scale,
                "applied": int(is_conflict and resolved_conflict_scale != 1.0),
                "raw_cosine": raw_cosine,
                "effective_cosine": effective_cosine,
                "normal_norm": normal_norm,
                "raw_det_norm": raw_det_norm,
                "effective_det_norm": effective_det_norm,
                "conflict_component_norm": conflict_norm,
                "conflict_component_fraction": (
                    conflict_norm / raw_det_norm
                    if raw_det_norm > 0.0
                    else float("nan")
                ),
                "shared_tensors": len(terms),
            }
            if self._writer is not None:
                self._writer.writerow(row)
                assert self._handle is not None
                self._handle.flush()
            return row
        finally:
            self._normal_gradients.clear()

    def close(self) -> None:
        self._active = False
        self._normal_gradients.clear()
        if self._handle is not None and not self._handle.closed:
            self._handle.close()
