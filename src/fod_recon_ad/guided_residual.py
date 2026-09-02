from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


RESIDUAL_PROFILES = (
    "raw_mean",
    "calibrated_mean",
    "calibrated_minimum",
    "subspace_mean",
    "subspace_minimum",
)


def normalized_residuals(
    encoder_features: Sequence[torch.Tensor],
    decoder_features: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    if len(encoder_features) != len(decoder_features) or not encoder_features:
        raise ValueError(
            f"Expected equal non-empty feature groups, got {len(encoder_features)} and "
            f"{len(decoder_features)}."
        )
    return [
        F.normalize(target.detach(), dim=1) - F.normalize(pred, dim=1)
        for target, pred in zip(encoder_features, decoder_features)
    ]


def fuse_group_scores(scores: Sequence[torch.Tensor], mode: str) -> torch.Tensor:
    if not scores:
        raise ValueError("At least one residual score group is required.")
    if mode == "mean":
        return torch.stack(list(scores), dim=0).mean(dim=0)
    if mode == "minimum":
        if len(scores) < 2:
            raise ValueError("Cross-scale minimum requires at least two feature groups.")
        return torch.minimum(scores[0], scores[-1])
    raise ValueError(f"Unsupported group score fusion: {mode}")


@dataclass
class GuidedResidualScorer:
    residual_mean: torch.Tensor
    basis: torch.Tensor
    raw_scale: torch.Tensor
    subspace_ranks: tuple[int, ...]
    subspace_scale: torch.Tensor
    profile: str = "raw_mean"
    rank: int = 0
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.profile not in RESIDUAL_PROFILES:
            raise ValueError(f"Unsupported guided residual profile: {self.profile}")
        if self.residual_mean.ndim != 2 or self.basis.ndim != 3:
            raise ValueError("Expected residual_mean [G,C] and basis [G,C,R].")
        groups, channels = self.residual_mean.shape
        if self.basis.shape[:2] != (groups, channels):
            raise ValueError("Residual mean and basis shapes do not match.")
        if self.raw_scale.shape != (groups,):
            raise ValueError("raw_scale must contain one value per feature group.")
        if self.subspace_scale.shape != (len(self.subspace_ranks), groups):
            raise ValueError("subspace_scale must have shape [rank_count, group_count].")
        if self.profile.startswith("subspace") and self.rank not in self.subspace_ranks:
            raise ValueError(
                f"Rank {self.rank} is unavailable; fitted ranks are {self.subspace_ranks}."
            )

    @property
    def device(self) -> torch.device:
        return self.residual_mean.device

    def to(self, device: torch.device) -> "GuidedResidualScorer":
        self.residual_mean = self.residual_mean.to(device)
        self.basis = self.basis.to(device)
        self.raw_scale = self.raw_scale.to(device)
        self.subspace_scale = self.subspace_scale.to(device)
        return self

    def _raw_scores(self, residuals: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        return [0.5 * residual.square().sum(dim=1, keepdim=True) for residual in residuals]

    def _subspace_scores(self, residuals: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        rank_index = self.subspace_ranks.index(self.rank)
        scores = []
        for group_index, residual in enumerate(residuals):
            centered = residual - self.residual_mean[group_index].view(1, -1, 1, 1)
            if self.rank > 0:
                basis = self.basis[group_index, :, : self.rank].to(centered.dtype)
                coefficients = torch.einsum("bchw,cr->brhw", centered, basis)
                parallel = torch.einsum("brhw,cr->bchw", coefficients, basis)
                centered = centered - parallel
            distance = 0.5 * centered.square().sum(dim=1, keepdim=True)
            scale = self.subspace_scale[rank_index, group_index].to(distance.dtype)
            scores.append(distance / scale.clamp_min(self.eps))
        return scores

    @torch.no_grad()
    def score(
        self,
        encoder_features: Sequence[torch.Tensor],
        decoder_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        residuals = normalized_residuals(encoder_features, decoder_features)
        if len(residuals) != self.residual_mean.shape[0]:
            raise ValueError(
                f"Scorer was fitted for {self.residual_mean.shape[0]} groups, "
                f"but model returned {len(residuals)}."
            )
        if self.profile == "raw_mean":
            return fuse_group_scores(self._raw_scores(residuals), "mean")
        if self.profile.startswith("calibrated"):
            scores = [
                score / self.raw_scale[index].to(score.dtype).clamp_min(self.eps)
                for index, score in enumerate(self._raw_scores(residuals))
            ]
        else:
            scores = self._subspace_scores(residuals)
        fusion = "minimum" if self.profile.endswith("minimum") else "mean"
        return fuse_group_scores(scores, fusion)


def save_guided_residual_scorer(
    path: str | Path,
    *,
    residual_mean: np.ndarray,
    basis: np.ndarray,
    raw_scale: np.ndarray,
    subspace_ranks: Sequence[int],
    subspace_scale: np.ndarray,
    metadata: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        residual_mean=np.asarray(residual_mean, dtype=np.float32),
        basis=np.asarray(basis, dtype=np.float32),
        raw_scale=np.asarray(raw_scale, dtype=np.float32),
        subspace_ranks=np.asarray(subspace_ranks, dtype=np.int64),
        subspace_scale=np.asarray(subspace_scale, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def load_guided_residual_scorer(
    path: str | Path,
    *,
    device: torch.device,
    profile: str,
    rank: int,
) -> tuple[GuidedResidualScorer, dict]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        scorer = GuidedResidualScorer(
            residual_mean=torch.from_numpy(payload["residual_mean"].astype(np.float32)),
            basis=torch.from_numpy(payload["basis"].astype(np.float32)),
            raw_scale=torch.from_numpy(payload["raw_scale"].astype(np.float32)),
            subspace_ranks=tuple(int(value) for value in payload["subspace_ranks"].tolist()),
            subspace_scale=torch.from_numpy(payload["subspace_scale"].astype(np.float32)),
            profile=profile,
            rank=int(rank),
        ).to(device)
    return scorer, metadata


def attach_guided_residual_scorer(
    model,
    path: str | Path | None,
    *,
    device: torch.device,
    profile: str,
    rank: int,
) -> dict | None:
    if path is None:
        return None
    scorer, metadata = load_guided_residual_scorer(
        path,
        device=device,
        profile=profile,
        rank=rank,
    )
    model._guided_residual_scorer = scorer
    return metadata


def reconstruction_score(
    model,
    encoder_features: Sequence[torch.Tensor],
    decoder_features: Sequence[torch.Tensor],
) -> torch.Tensor:
    diffusion_scorer = getattr(model, "_residual_diffusion_scorer", None)
    if diffusion_scorer is not None:
        return diffusion_scorer.score(encoder_features, decoder_features)
    scorer = getattr(model, "_guided_residual_scorer", None)
    if scorer is None:
        residuals = normalized_residuals(encoder_features, decoder_features)
        return fuse_group_scores(
            [0.5 * residual.square().sum(dim=1, keepdim=True) for residual in residuals],
            "mean",
        )
    return scorer.score(encoder_features, decoder_features)
