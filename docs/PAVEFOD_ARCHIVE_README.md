# PaveFOD v1.0.0

PaveFOD is the dataset accompanying **Object-Aware Learning for Open-Set
Foreign Object Debris Detection**.

## Contents

- 87 privacy-filtered RGB images at 2560 x 1440
- 4 object-free training images
- 41 anomalous validation images with 95 object instances
- 42 anomalous test images with 103 object instances
- Binary and instance-level PNG masks
- Eight canonical object types and nominal 5-30 m acquisition ranges
- A fixed split and per-image/per-instance metadata

The six distance codes are `05`, `10`, `15`, `20`, `25`, and `30`; code `30`
is the mixed-range `5-30m` group. Each RGB image and its masks share the same
sample stem. Instance masks use zero for background and positive integers for
individual objects; `instances.csv` maps each integer to its object type,
depth, area, and bounding box.

## Privacy processing

Pixels inside the verified pavement ROI are preserved exactly. Pixels outside
the ROI are replaced with white to remove people and unrelated public-space
content. Original camera IP addresses and capture timestamps are removed from
all public filenames and metadata. The hard and soft ROI masks are included in
`roi/`.

## Integrity

From this directory, verify every released file with:

```bash
sha256sum -c SHA256SUMS
```

## License and citation

The dataset is released under CC BY-NC 4.0; see `LICENSE.md`. Please cite the
OAL-FOD paper and https://github.com/Perfectfox/OAL-FOD when using PaveFOD.

