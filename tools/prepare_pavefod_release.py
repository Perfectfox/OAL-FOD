#!/usr/bin/env python3
"""Build the privacy-filtered, identifier-free PaveFOD public release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--roi-hard", type=Path, required=True)
    parser.add_argument("--roi-soft", type=Path, required=True)
    parser.add_argument("--instance-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_png(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="PNG", optimize=True, compress_level=9)


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    with Image.open(args.roi_hard) as image:
        roi_hard = np.asarray(image.convert("L")) > 0
    with Image.open(args.roi_soft) as image:
        roi_soft = np.asarray(image.convert("L"))

    manifest_path = args.split_root / "split_manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))

    counters: Counter[str] = Counter()
    stem_to_id: dict[str, str] = {}
    metadata: list[dict] = []
    inside_roi_mismatches = 0

    for row in source_rows:
        split = row["split"]
        counters[split] += 1
        sample_id = f"PF_{split}_{counters[split]:04d}"
        label_name = "good" if row["label"] == "0" else "anomaly"
        base = Path(row["distance"]) / split / label_name
        image_rel = base / "rgb" / f"{sample_id}.png"
        binary_rel = base / "gt" / f"{sample_id}.png" if row["output_mask"] else None
        instance_rel = (
            base / "instance_gt" / f"{sample_id}.png"
            if row["output_instance_mask"]
            else None
        )

        source_image_path = args.split_root / row["output_image"]
        with Image.open(source_image_path) as image:
            source_image = np.asarray(image.convert("RGB"))
        if source_image.shape[:2] != roi_hard.shape:
            raise RuntimeError(f"ROI size mismatch for {source_image_path}")
        released_image = source_image.copy()
        released_image[~roi_hard] = 255
        if not np.array_equal(source_image[roi_hard], released_image[roi_hard]):
            inside_roi_mismatches += 1
        save_png(released_image, args.output / image_rel)

        if binary_rel is not None:
            with Image.open(args.split_root / row["output_mask"]) as image:
                binary = np.asarray(image.convert("L"))
            save_png(binary, args.output / binary_rel)
        if instance_rel is not None:
            with Image.open(args.split_root / row["output_instance_mask"]) as image:
                instances = np.asarray(image)
            save_png(instances, args.output / instance_rel)

        source_stem = source_image_path.stem
        if source_stem in stem_to_id:
            raise RuntimeError(f"Duplicate source stem: {source_stem}")
        stem_to_id[source_stem] = sample_id
        metadata.append(
            {
                "sample_id": sample_id,
                "split": split,
                "label": label_name,
                "distance_code": row["distance"],
                "nominal_range": row["distance_group"],
                "instance_count": row["instance_count"],
                "mask_area_px": row["mask_area"],
                "image": image_rel.as_posix(),
                "binary_mask": binary_rel.as_posix() if binary_rel else "",
                "instance_mask": instance_rel.as_posix() if instance_rel else "",
            }
        )

    roi_dir = args.output / "roi"
    save_png((roi_hard.astype(np.uint8) * 255), roi_dir / "ground_roi_hard.png")
    save_png(roi_soft, roi_dir / "ground_roi_soft.png")

    metadata_fields = [
        "sample_id",
        "split",
        "label",
        "distance_code",
        "nominal_range",
        "instance_count",
        "mask_area_px",
        "image",
        "binary_mask",
        "instance_mask",
    ]
    write_csv(args.output / "metadata.csv", metadata, metadata_fields)

    with args.instance_records.open(newline="", encoding="utf-8") as handle:
        source_instances = list(csv.DictReader(handle))
    instances = []
    for row in source_instances:
        instances.append(
            {
                "sample_id": stem_to_id[row["image_stem"]],
                "split": row["split"],
                "instance_id": row["instance_id"],
                "object_type": row["label"],
                "depth_m": row["depth_m"],
                "mask_area_px": row["mask_area_px"],
                "mask_bbox_xyxy": row["mask_bbox"],
            }
        )
    instance_fields = [
        "sample_id",
        "split",
        "instance_id",
        "object_type",
        "depth_m",
        "mask_area_px",
        "mask_bbox_xyxy",
    ]
    write_csv(args.output / "instances.csv", instances, instance_fields)

    split_counts = Counter(row["split"] for row in metadata)
    instance_counts = Counter(row["split"] for row in instances)
    object_counts = Counter(row["object_type"] for row in instances)
    manifest = {
        "name": "PaveFOD",
        "version": "1.0.0",
        "image_resolution": [2560, 1440],
        "images": len(metadata),
        "images_by_split": dict(sorted(split_counts.items())),
        "instances": len(instances),
        "instances_by_split": dict(sorted(instance_counts.items())),
        "instances_by_object_type": dict(sorted(object_counts.items())),
        "privacy_processing": {
            "inside_verified_roi_preserved": inside_roi_mismatches == 0,
            "outside_verified_roi": "replaced with white",
            "camera_ip_and_capture_timestamp_in_public_metadata": False,
        },
    }
    (args.output / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    repository_root = Path(__file__).resolve().parents[1]
    shutil.copyfile(
        repository_root / "docs" / "PAVEFOD_ARCHIVE_README.md",
        args.output / "README.md",
    )
    shutil.copyfile(repository_root / "DATASET_LICENSE.md", args.output / "LICENSE.md")

    checksum_paths = sorted(
        path for path in args.output.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    with (args.output / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in checksum_paths:
            handle.write(f"{sha256(path)}  {path.relative_to(args.output).as_posix()}\n")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
