from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import torch


def _strip_crop_coordinates(path: str) -> str:
    parts = path.rsplit(":", 2)
    if len(parts) == 3 and parts[1].lstrip("-").isdigit() and parts[2].lstrip("-").isdigit():
        return parts[0]
    return path


def stable_group_id(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) & ((1 << 63) - 1)


class SourceGroupResolver:
    """Resolve training patches to source-image groups without dataset-specific call sites."""

    def __init__(self, manifests: Sequence[str | Path] = ()) -> None:
        self._index_cache: dict[Path, dict[str, str]] = {}
        self._explicit_sources: dict[str, str] = {}
        for manifest_value in manifests:
            manifest = Path(manifest_value)
            if not manifest.exists():
                raise FileNotFoundError(manifest)
            with manifest.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    source = row.get("source_file", "")
                    patch = row.get("image", "") or row.get("patch_file", "")
                    if source and patch:
                        self._explicit_sources[Path(patch).name] = source

    def _candidate_indexes(self, path: Path) -> Iterable[Path]:
        seen: set[Path] = set()
        for candidate_path in (path, path.resolve()):
            for parent in (candidate_path.parent, candidate_path.parent.parent):
                index = parent / "index.csv"
                if index not in seen:
                    seen.add(index)
                    yield index

    def _load_index(self, path: Path) -> dict[str, str]:
        if path in self._index_cache:
            return self._index_cache[path]
        mapping: dict[str, str] = {}
        if path.exists():
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    patch = row.get("patch_file", "")
                    source = row.get("source_file", "")
                    if patch and source:
                        mapping[Path(patch).name] = source
        self._index_cache[path] = mapping
        return mapping

    def resolve(self, sample_path: str | Path) -> str:
        raw_path = _strip_crop_coordinates(str(sample_path))
        path = Path(raw_path)
        explicit = self._explicit_sources.get(path.name)
        if explicit:
            return explicit
        for index_path in self._candidate_indexes(path):
            source = self._load_index(index_path).get(path.name)
            if source:
                return source
        try:
            return str(path.resolve())
        except OSError:
            return str(path)

    def ids(self, sample_paths: Sequence[str | Path], device: torch.device | None = None) -> torch.Tensor:
        values = [stable_group_id(self.resolve(path)) for path in sample_paths]
        return torch.tensor(values, dtype=torch.long, device=device)


def select_safe_alpha(
    native_q99: Sequence[float],
    candidate_q99: dict[float, Sequence[float]],
    relative_tolerance: float = 0.03,
    absolute_tolerance: float = 0.0,
) -> float:
    """Choose the largest alpha whose normal residual tail stays within the Native envelope."""

    if not native_q99:
        raise ValueError("Native Q99 statistics must not be empty.")
    safe: list[float] = []
    for alpha, values in candidate_q99.items():
        if len(values) != len(native_q99):
            raise ValueError(
                f"Alpha {alpha} has {len(values)} Q99 values, expected {len(native_q99)}."
            )
        limits = [
            float(reference) + max(float(absolute_tolerance), abs(float(reference)) * float(relative_tolerance))
            for reference in native_q99
        ]
        if all(float(value) <= limit for value, limit in zip(values, limits)):
            safe.append(float(alpha))
    return max(safe) if safe else 0.0
