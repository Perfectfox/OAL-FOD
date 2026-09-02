# Reproducibility guide

## Data paths

After extracting the release, use the extracted directory as `--fod-root` and
the hard ROI mask as `--ground-roi-mask`:

```bash
export PAVEFOD_ROOT=/path/to/PaveFOD-v1.0.0
export PAVEFOD_ROI="$PAVEFOD_ROOT/roi/ground_roi_hard.png"
```

The fixed distances are `05`, `10`, `15`, `20`, `25`, and `30`. Distance `30`
is the mixed-range `5-30m` group. The loader detects the released `original`
layout automatically.

## External dependencies

OAL-FOD relies on separately cloned INP-Former and Dinomaly implementations.
OER additionally needs user-provided, legally obtained external object crops.
The paper uses VisDrone bounding boxes only to construct those crops; VisDrone
content is not redistributed here.

```bash
git clone https://github.com/zhangzjn/INP-Former.git
git clone https://github.com/guojiajeremy/Dinomaly.git
```

Use `--help` on each script for the full option set. The main stages are:

1. `scripts/train_reconstruct.py`
2. `scripts/rebuild_familiarity_memory.py`
3. `scripts/build_adaptive_center_teacher.py`
4. `scripts/train_reconstruct_object_erasing.py`
5. `scripts/eval_sliding_reconstruct.py`

The optional inline-validation flags in the two training scripts refer to
project-internal cache-analysis helpers that are not part of this compact
release. Leave inline validation disabled and use the standalone evaluation
script for the public workflow.

The paper protocol uses no PaveFOD anomaly image or annotation during model
training. Validation annotations are reserved for operating-point selection,
and the selected threshold is then applied unchanged to the test split.

## Release scope

This repository provides the method implementation and dataset needed to build
the experimental pipeline. Large pretrained backbone weights and trained
OAL-FOD checkpoints are not included in v1.0.0. Random seeds, architecture
settings, training schedules, and evaluation definitions are reported in the
paper and exposed by the scripts.
