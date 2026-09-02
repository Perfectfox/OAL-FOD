#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fod_recon_ad.familiarity_memory_rebuild import (  # noqa: E402
    adaptive_modes_from_stable_teacher,
    hierarchical_reliability_modes_from_stable_teacher,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package an adaptive Center6 teacher from an existing Normal memory, "
            "stable teacher and its Normal-only diagnostics."
        )
    )
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--stable-teacher", type=Path, required=True)
    parser.add_argument("--memory-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--min-memory-members", type=int, default=2)
    parser.add_argument("--min-memory-views", type=int, default=2)
    parser.add_argument(
        "--one-slot-per-mode",
        action="store_true",
        help=(
            "After Normal-only stable-mode selection, allocate exactly one "
            "physical prototype slot to every effective mode. This changes "
            "student capacity without changing the selected teacher modes or memory."
        ),
    )
    parser.add_argument(
        "--hierarchical-reliability",
        action="store_true",
        help=(
            "Keep all candidate child modes and store a continuous Normal-only "
            "within-group reliability instead of hard-merging failed splits."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.memory, args.stable_teacher, args.memory_summary):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = json.loads(args.memory_summary.read_text(encoding="utf-8"))
    mode_diagnostics = summary.get("stable_mode_teacher")
    if not isinstance(mode_diagnostics, dict):
        raise ValueError("Memory summary does not contain stable_mode_teacher diagnostics.")

    with np.load(args.memory, allow_pickle=False) as payload:
        memory_payload = {key: np.asarray(payload[key]) for key in payload.files}
    required_memory = {"bank", "source_view_ids", "semantic_group_ids"}
    missing = required_memory.difference(memory_payload)
    if missing:
        raise ValueError(f"Memory sidecar is missing arrays: {sorted(missing)}")
    with np.load(args.stable_teacher, allow_pickle=False) as payload:
        required_teacher = {"mode_teacher_centers", "mode_teacher_groups"}
        missing = required_teacher.difference(payload.files)
        if missing:
            raise ValueError(f"Stable teacher is missing arrays: {sorted(missing)}")
        mode_centers = torch.from_numpy(
            np.asarray(payload["mode_teacher_centers"], dtype=np.float32)
        )
        groups = tuple(
            int(value)
            for value in np.asarray(payload["mode_teacher_groups"], dtype=np.int64).tolist()
        )

    common_args = (
        mode_centers,
        groups,
        mode_diagnostics,
        torch.from_numpy(np.asarray(memory_payload["bank"], dtype=np.float32)),
        torch.from_numpy(np.asarray(memory_payload["source_view_ids"], dtype=np.int64)),
        torch.from_numpy(np.asarray(memory_payload["semantic_group_ids"], dtype=np.int64)),
    )
    common_kwargs = {
        "min_memory_members_per_mode": args.min_memory_members,
        "min_memory_views_per_mode": args.min_memory_views,
    }
    adaptive_group_reliability = None
    if args.hierarchical_reliability:
        (
            adaptive_centers,
            adaptive_groups,
            slot_to_mode,
            mode_group_ids,
            adaptive_group_reliability,
            adaptive_diagnostics,
        ) = hierarchical_reliability_modes_from_stable_teacher(
            *common_args, **common_kwargs
        )
    else:
        (
            adaptive_centers,
            adaptive_groups,
            slot_to_mode,
            mode_group_ids,
            adaptive_diagnostics,
        ) = adaptive_modes_from_stable_teacher(*common_args, **common_kwargs)
    slot_groups = groups
    if args.one_slot_per_mode:
        slot_groups = adaptive_groups
        slot_to_mode = torch.arange(
            int(sum(adaptive_groups)),
            dtype=torch.long,
        )
        adaptive_diagnostics = {
            **adaptive_diagnostics,
            "version": "adaptive_stable_modes_one_slot_per_mode_v1",
            "slot_groups": list(slot_groups),
            "prototype_slot_count": int(sum(slot_groups)),
            "slot_to_mode": slot_to_mode.tolist(),
            "capacity_policy": "one_slot_per_effective_mode",
        }
    memory_payload.update(
        adaptive_mode_centers=adaptive_centers.numpy().astype(np.float32),
        adaptive_mode_groups=np.asarray(adaptive_groups, dtype=np.int64),
        adaptive_slot_groups=np.asarray(slot_groups, dtype=np.int64),
        adaptive_slot_to_mode=slot_to_mode.numpy().astype(np.int64),
        adaptive_mode_group_ids=mode_group_ids.numpy().astype(np.int64),
        adaptive_mode_construction_version=np.asarray(adaptive_diagnostics["version"]),
    )
    if adaptive_group_reliability is not None:
        memory_payload["adaptive_group_reliability"] = (
            adaptive_group_reliability.numpy().astype(np.float32)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **memory_payload)
    summary_output = args.summary_output or args.output.with_suffix(".json")
    summary_output.write_text(
        json.dumps(adaptive_diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(adaptive_diagnostics, indent=2), flush=True)
    print(f"[done] {args.output}", flush=True)


if __name__ == "__main__":
    main()
