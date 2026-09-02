#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch
import torch.nn as nn

_MISSING = object()
_RUNTIME_MODEL_ATTRIBUTES = (
    "_guided_prototype_image",
    "_guided_update_prior_stats",
    "_guided_source_group_ids",
    "_guided_pending_prior_state",
    "_guided_last_prior_state",
    "_guided_prototype_diag",
    "_guided_aggregation_diag",
    "_guided_spatial_diag",
    "_guided_extra_diag",
    "_context_normal_source_group_ids",
    "_context_normal_diag",
    "_normal_loss_diag",
    "_adaptive_mask_state",
    "_candidate_mask_state",
    "_masked_gather_loss",
    "distribution",
    "distance",
    "cluster_index",
)


def capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def seed_validation(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def preserve_training_runtime(model: nn.Module, validation_seed: int) -> Iterator[None]:
    """Isolate validation from the live training state and random streams."""

    module_modes = [(module, bool(module.training)) for module in model.modules()]
    rng_state = capture_rng_state()
    runtime_attributes = {
        name: getattr(model, name, _MISSING) for name in _RUNTIME_MODEL_ATTRIBUTES
    }
    try:
        seed_validation(validation_seed)
        model.eval()
        if getattr(model, "_guided_prototype_enabled", False):
            model._guided_prototype_image = None
            model._guided_update_prior_stats = False
            model._guided_source_group_ids = None
        if getattr(model, "_context_normal_enabled", False):
            model._context_normal_source_group_ids = None
        yield
    finally:
        for name, value in runtime_attributes.items():
            if value is _MISSING:
                if hasattr(model, name):
                    delattr(model, name)
            else:
                setattr(model, name, value)
        # Assign directly so a parent .train() call cannot overwrite a child's
        # intentionally different mode (for example, a frozen mask planner).
        for module, training in module_modes:
            module.training = training
        restore_rng_state(rng_state)


@dataclass(frozen=True)
class InlineValidationSchedule:
    origin_step: int
    epoch_iters: int
    every_epochs: int
    include_origin: bool = False
    start_epoch: int = 0

    def __post_init__(self) -> None:
        if self.origin_step < 0:
            raise ValueError("origin_step must be non-negative")
        if self.epoch_iters <= 0:
            raise ValueError("epoch_iters must be positive")
        if self.every_epochs <= 0:
            raise ValueError("every_epochs must be positive")
        if self.start_epoch < 0:
            raise ValueError("start_epoch must be non-negative")

    @property
    def interval_steps(self) -> int:
        return self.epoch_iters * self.every_epochs

    def should_validate(self, step: int) -> bool:
        relative_step = step - self.origin_step
        if relative_step < 0:
            return False
        if relative_step % self.epoch_iters != 0:
            return False
        continued_epoch = relative_step // self.epoch_iters
        if continued_epoch == 0:
            return self.start_epoch == 0 and self.include_origin
        first_positive_epoch = self.start_epoch if self.start_epoch > 0 else self.every_epochs
        return (
            continued_epoch >= first_positive_epoch
            and (continued_epoch - first_positive_epoch) % self.every_epochs == 0
        )

    def continued_epoch(self, step: int) -> int:
        relative_step = step - self.origin_step
        if relative_step < 0 or relative_step % self.epoch_iters != 0:
            raise ValueError(
                f"step={step} is not an epoch boundary relative to origin={self.origin_step}"
            )
        return relative_step // self.epoch_iters


class InlineFODValidator:
    """Run FOD validation inside a live training process without touching its loader."""

    def __init__(self, train_args: argparse.Namespace, schedule: InlineValidationSchedule):
        if train_args.dataset != "fod":
            raise ValueError("Inline FOD validation requires --dataset fod.")
        if train_args.architecture not in {"dinomaly", "inpformer"}:
            raise ValueError(
                "Inline FOD validation currently supports dinomaly and inpformer only."
            )
        if train_args.masked_recon and train_args.mask_strategy == "candidate":
            raise ValueError(
                "Inline FOD validation does not yet define candidate-mask inference."
            )
        if train_args.inline_fod_val_output_dir is None:
            raise ValueError("--inline-fod-val-output-dir is required when inline validation is enabled.")
        self.args = train_args
        self.schedule = schedule
        self.output_dir = Path(train_args.inline_fod_val_output_dir)
        self.selection_dir = self.output_dir / "selection"
        self.selection_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "inline_validation_config.txt").write_text(
            "\n".join(
                [
                    f"origin_step={schedule.origin_step}",
                    f"epoch_iters={schedule.epoch_iters}",
                    f"every_epochs={schedule.every_epochs}",
                    f"start_epoch={schedule.start_epoch}",
                    f"include_origin={int(schedule.include_origin)}",
                    f"physical_crop_size={train_args.crop_size}",
                    "model_input_size="
                    + str(
                        train_args.patch_output_size
                        if train_args.patch_output_size > 0
                        else train_args.image_size
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def should_validate(self, step: int) -> bool:
        return self.schedule.should_validate(step)

    def _cache_args(self, cache_dir: Path) -> argparse.Namespace:
        from cache_fod_learnable_prior_inputs import build_parser as build_cache_parser

        model_input_size = (
            self.args.patch_output_size
            if self.args.patch_output_size > 0
            else self.args.image_size
        )
        argv = [
            "--fod-root",
            str(self.args.fod_root),
            "--layout",
            self.args.layout,
            "--distances",
            *self.args.inline_fod_val_distances,
            "--source-split",
            "val_model",
            "--output-dir",
            str(cache_dir),
            "--external-repo",
            str(self.args.external_repo),
            "--inpformer-repo",
            str(self.args.inpformer_repo),
            "--architecture",
            self.args.architecture,
            "--encoder",
            self.args.encoder,
            "--inp-num",
            str(self.args.inp_num),
            "--crop-size",
            str(self.args.crop_size),
            "--model-input-size",
            str(model_input_size),
            "--stride",
            str(self.args.inline_fod_val_stride or self.args.patch_stride),
            "--batch-size",
            str(self.args.inline_fod_val_batch_size),
            "--device",
            self.args.device,
            "--seed",
            str(self.args.inline_fod_val_seed),
            "--feature-dtype",
            "float16",
            "--resume",
            "--progress-every",
            str(self.args.inline_fod_val_progress_every),
            "--skip-memory-cache",
            "--skip-dino-feature-cache",
        ]
        if self.args.amp:
            argv.append("--amp")
        if self.args.ground_roi_mask is not None:
            argv.extend(["--valid-roi-mask", str(self.args.ground_roi_mask)])
        if self.args.masked_recon:
            argv.extend(
                [
                    "--masked-recon",
                    "--mask-strategy",
                    self.args.mask_strategy,
                    "--mask-band-block-sizes",
                    self.args.mask_band_block_sizes,
                    "--mask-fill",
                    self.args.mask_fill,
                    "--mask-prototype-source",
                    self.args.mask_prototype_source,
                    "--adaptive-mask-hidden-dim",
                    str(self.args.adaptive_mask_hidden_dim),
                    "--adaptive-mask-ratio",
                    str(self.args.adaptive_mask_ratio),
                    "--adaptive-mask-segments",
                    str(self.args.adaptive_mask_segments),
                    "--adaptive-mask-temperature",
                    str(self.args.adaptive_mask_temperature),
                ]
            )
        if self.args.inp_local_mask_recon:
            argv.extend(
                [
                    "--inp-local-mask-recon",
                    "--local-context-radius",
                    str(self.args.local_context_radius),
                    "--mask-planner-checkpoint",
                    str(self.args.mask_planner_checkpoint),
                ]
            )
        return build_cache_parser().parse_args(argv)

    def _analysis_args(self, cache_dir: Path, metrics_dir: Path) -> argparse.Namespace:
        from analyze_fod_cached_object_level import build_parser as build_analysis_parser

        argv = [
            "--cache-dir",
            str(cache_dir),
            "--output-dir",
            str(metrics_dir),
            "--variants",
            "sliding_raw",
            "--object-threshold-scope",
            "global",
            "--progress-every",
            str(self.args.inline_fod_val_progress_every),
            "--min-pred-area",
            str(self.args.inline_fod_val_min_pred_area),
            "--iou-threshold",
            str(self.args.inline_fod_val_iou_threshold),
        ]
        return build_analysis_parser().parse_args(argv)

    def validate(
        self,
        model: nn.Module,
        step: int,
        save_checkpoint: Callable[[Path, int], None],
    ) -> tuple[dict[str, object], bool]:
        from analyze_fod_cached_object_level import analyze_cached_fod
        from cache_fod_learnable_prior_inputs import cache_fod_inputs
        from update_fod_val_checkpoint_selection import update_selection

        if not self.should_validate(step):
            raise ValueError(f"Inline validation was requested at unscheduled step={step}.")
        continued_epoch = self.schedule.continued_epoch(step)
        label = f"val_ep{continued_epoch:03d}"
        cache_dir = self.output_dir / f"cache_{label}"
        metrics_dir = self.output_dir / f"metrics_{label}"
        current_checkpoint = self.selection_dir / "current_val.pth"
        print(
            f"[inline_val] start step={step} continued_epoch={continued_epoch}",
            flush=True,
        )
        # Save before reseeding for validation. The selected checkpoint must
        # carry the exact training RNG state so it remains a valid resume point.
        save_checkpoint(current_checkpoint, step)
        with preserve_training_runtime(model, self.args.inline_fod_val_seed):
            metrics_complete = (
                (metrics_dir / "streaming_metrics.json").exists()
                and (metrics_dir / "object_level_summary.json").exists()
                and (metrics_dir / "thresholds.json").exists()
            )
            if not metrics_complete:
                cache_fod_inputs(self._cache_args(cache_dir), model=model)
                analyze_cached_fod(self._analysis_args(cache_dir, metrics_dir))
            selection_args = argparse.Namespace(
                metrics_dir=metrics_dir,
                checkpoint=current_checkpoint,
                selection_dir=self.selection_dir,
                continued_epoch=continued_epoch,
                absolute_iter=step,
                resolution=(
                    self.args.patch_output_size
                    if self.args.patch_output_size > 0
                    else self.args.image_size
                ),
            )
            record, improved = update_selection(selection_args)
        if not self.args.inline_fod_val_keep_cache:
            shutil.rmtree(cache_dir, ignore_errors=True)
        print(
            f"[inline_val] done step={step} continued_epoch={continued_epoch} "
            f"P-AP={float(record['P-AP']):.8f} improved={int(improved)}",
            flush=True,
        )
        return record, improved
