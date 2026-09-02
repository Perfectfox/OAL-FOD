#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fod_recon_ad.data import make_crop_train_loader, make_mvtec_train_loader, make_train_loader, normalize_distance
from fod_recon_ad.blindspot_context import (
    add_blindspot_context_args,
    attach_blindspot_context_head,
    blindspot_context_score,
)
from fod_recon_ad.context_normal_prototype import (
    context_normal_memory_ready,
    fit_context_normal_memory,
    get_context_normal_diag,
    set_context_normal_source_groups,
)
from fod_recon_ad.context_familiarity_transport import (
    context_familiarity_transport_memory_ready,
    fit_context_familiarity_transport_memory,
)
from fod_recon_ad.normal_descriptor_gate import (
    fit_normal_descriptor_memory,
    normal_descriptor_memory_ready,
)
from fod_recon_ad.ext import (
    build_reconstruction_model,
    checkpoint_payload,
    load_checkpoint,
    load_stable_adamw,
    select_reconstruction_trainable_modules,
)
from fod_recon_ad.masking import (
    PerspectiveMaskConfig,
    adaptive_mask_planner_loss,
    attach_adaptive_mask_planner,
    attach_candidate_mask_prompt,
    attach_local_context_reconstructor,
    forward_masked_reconstruction,
    parse_block_sizes,
)
from fod_recon_ad.native_losses import (
    compute_masked_normal_loss,
    compute_normal_loss,
    compute_roi_masked_normal_loss,
)
from fod_recon_ad.inpformer_plus import configure_inp_coherence
from fod_recon_ad.learned_mask import load_planner_state
from fod_recon_ad.normal_calibration import SourceGroupResolver
from fod_recon_ad.prototype_guidance import (
    add_guided_prototype_args,
    configure_guided_prototypes,
    guided_prototype_extra_loss,
    guided_prototype_trainable_modules,
    get_guided_prototype_diag,
    inpformer_forward_with_prototype_context,
    set_guided_prototype_image,
    set_guided_prototype_source_groups,
    set_guided_prototype_valid_roi,
)
from fod_recon_ad.scoring import reconstruction_components
from fod_inline_validation import InlineFODValidator, InlineValidationSchedule


CENTER6_DIAGNOSTIC_FIELDS = (
    "guided_center6_loss",
    "guided_center6_accuracy",
    "guided_center6_teacher_entropy",
    "guided_center6_student_entropy",
    "guided_center6_student_slot_entropy",
    "guided_center6_effective_mode_count",
    "guided_center6_prototype_slot_count",
    "guided_center6_teacher_confidence",
    "guided_center6_teacher_min_usage",
    "guided_center6_student_min_usage",
    "guided_center6_teacher_dead_modes",
    "guided_center6_student_dead_modes",
    "guided_center6_student_dead_slots",
    "guided_center6_pair_cosine_mean",
    "guided_center6_pair_cosine_max",
    "guided_center6_pair_cosine_min",
    "guided_center6_pair_min_distance",
    "guided_center6_shared_parent_cosine_mean",
    "guided_center6_shared_parent_cosine_max",
    "guided_center6_shared_parent_pair_count",
    "guided_center6_collapsed_diversity_loss",
    "guided_center6_collapsed_diversity_weight",
    "guided_center6_alignment_mean",
    "guided_center6_alignment_min",
    "guided_center6_radius_min",
    "guided_center6_radius_mean",
    "guided_center6_radius_max",
    "guided_center6_member_count_min",
    "guided_center6_hierarchical_reliability",
    "guided_center6_group_loss",
    "guided_center6_conditional_loss",
    "guided_center6_group_reliability_mean",
    "guided_center6_group_reliability_min",
    *(f"guided_center6_group_reliability_{index}" for index in range(3)),
    *(f"guided_center6_mode_loss_{index}" for index in range(6)),
    *(f"guided_center6_mode_weight_{index}" for index in range(6)),
    *(f"guided_center6_teacher_usage_{index}" for index in range(6)),
    *(f"guided_center6_student_usage_{index}" for index in range(6)),
    *(f"guided_center6_student_slot_usage_{index}" for index in range(6)),
)

DESCRIPTOR_DIAGNOSTIC_FIELDS = (
    "guided_semantic_groups_active",
    "guided_descriptor_variant_id",
    "guided_descriptor_memory_count",
    "guided_descriptor_appearance_novelty",
    "guided_descriptor_context_surprise",
    "guided_descriptor_confidence",
    "guided_descriptor_local_objectness",
    "guided_descriptor_source_diversity",
    "guided_descriptor_write_risk",
    "guided_descriptor_read_risk_raw",
    "guided_descriptor_read_risk",
    "guided_descriptor_context_fallback_ratio",
    "guided_descriptor_appearance_fallback_ratio",
    "guided_descriptor_objectness_factor",
    "guided_descriptor_read_activation",
    "guided_descriptor_read_layer_start",
    "guided_descriptor_read_layer_count",
    "guided_descriptor_read_strength",
    "guided_descriptor_scale_r1",
    "guided_descriptor_scale_r2",
    "guided_descriptor_scale_r4",
    "guided_descriptor_reliability_r1",
    "guided_descriptor_reliability_r2",
    "guided_descriptor_reliability_r4",
)


def parse_float_list(text: str) -> tuple[float, ...]:
    values = [float(part) for part in text.replace(",", " ").split() if part.strip()]
    if any(value <= 0 for value in values):
        raise ValueError(f"All weights must be positive: {values}")
    return tuple(values)


def resolve_best_loss_checkpoint_path(output_dir: Path, filename: str) -> Path:
    path = Path(filename)
    if path.name != filename or path.suffix != ".pth":
        raise ValueError(
            "--best-loss-checkpoint-name must be a .pth filename without directories."
        )
    return output_dir / path


def replace_checkpoint_with_hardlink(source: Path, destination: Path) -> None:
    """Atomically point a same-filesystem checkpoint name at an existing payload."""

    temporary = destination.with_name(f".{destination.name}.link-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def analysis_checkpoint_due(epoch: int, total_epochs: int, every_epochs: int) -> bool:
    if every_epochs <= 0:
        return False
    return epoch == total_epochs or epoch % every_epochs == 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a FOD-specific reconstruction model.")
    parser.add_argument("--dataset", choices=["fod", "mvtec"], default="fod")
    parser.add_argument("--fod-root", type=Path, default=PROJECT_ROOT.parent / "datasets" / "FOD_dataset")
    parser.add_argument("--mvtec-root", type=Path, default=PROJECT_ROOT.parent / "mvtec_anomaly_detection")
    parser.add_argument("--mvtec-category", default="toothbrush")
    parser.add_argument("--layout", choices=["auto", "original", "mvtec"], default="auto")
    parser.add_argument("--external-repo", type=Path, default=PROJECT_ROOT.parent / "Dinomaly")
    parser.add_argument("--inpformer-repo", type=Path, default=PROJECT_ROOT.parent / "INP-Former")
    parser.add_argument("--architecture", choices=["dinomaly", "inpformer", "mamba"], default="dinomaly")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional model checkpoint for warm-start training.")
    parser.add_argument("--strict-load", action="store_true", help="Strictly load --checkpoint model weights.")
    parser.add_argument(
        "--checkpoint-allowed-missing-prefixes",
        default="",
        help=(
            "Comma-separated model-state prefixes that may be absent from a strict "
            "warm-start checkpoint, for example newly attached frozen guidance buffers."
        ),
    )
    parser.add_argument(
        "--trainable-scope",
        choices=["full", "prototype_aggregation", "reconstruction_only", "blindspot_context"],
        default="full",
        help=(
            "Parameter update boundary. prototype_aggregation freezes encoder, "
            "bottleneck, decoder, and auxiliary guidance modules. reconstruction_only "
            "freezes the encoder and all prototype/guidance modules."
        ),
    )
    parser.add_argument(
        "--blindspot-calibration-batches",
        type=int,
        default=0,
        help="Normal training batches used to persist Adaptive/context score quantiles.",
    )
    parser.add_argument("--train-distance", default="05")
    parser.add_argument("--encoder", default="dinov2reg_vit_base_14")
    parser.add_argument("--inp-num", type=int, default=6)
    parser.add_argument("--prototype-loss-weight", type=float, default=0.2)
    parser.add_argument("--mamba-layers", type=int, default=4)
    parser.add_argument("--mamba-scan", choices=["hilbert", "zorder", "snake", "raster"], default="hilbert")
    parser.add_argument("--mamba-d-state", type=int, default=16)
    parser.add_argument("--mamba-d-conv", type=int, default=4)
    parser.add_argument("--mamba-expand", type=int, default=2)
    parser.add_argument("--mamba-unidirectional", action="store_true")
    parser.add_argument("--mamba-drop", type=float, default=0.0)
    parser.add_argument("--mamba-multi-output", action="store_true")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--crop-size", type=int, default=448)
    parser.add_argument(
        "--mvtec-physical-crop-size",
        type=int,
        default=0,
        help="For MVTec-style training, randomly crop this physical side before resizing to --image-size; 0 keeps legacy resize/crop behavior.",
    )
    parser.add_argument("--patch-train", action="store_true")
    parser.add_argument("--patch-full-height", type=int, default=672)
    parser.add_argument("--patch-stride", type=int, default=224)
    parser.add_argument("--ground-roi-mask", type=Path, default=None)
    parser.add_argument("--ground-roi-min-coverage", type=float, default=0.0)
    parser.add_argument(
        "--roi-mask-loss",
        action="store_true",
        help="Restrict every spatial Normal reconstruction loss to the crop Clean ROI.",
    )
    parser.add_argument(
        "--roi-mask-erode-pixels",
        type=int,
        default=0,
        help="Erode the training Clean ROI in physical pre-resize pixels.",
    )
    parser.add_argument("--patch-output-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=7)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--train-drop-last",
        action="store_true",
        help="Drop an undersized final training batch (off by default).",
    )
    parser.add_argument("--total-iters", type=int, default=5000)
    parser.add_argument("--epoch-iters", type=int, default=0, help="Steps per epoch for best checkpoint selection.")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "stable_adamw"],
        default="adamw",
    )
    parser.add_argument(
        "--optimizer-eps",
        type=float,
        default=1e-8,
        help="Numerical epsilon for AdamW/StableAdamW.",
    )
    parser.add_argument("--optimizer-amsgrad", action="store_true")
    parser.add_argument(
        "--stable-adamw-clip-threshold",
        type=float,
        default=1.0,
        help="RMS update clipping threshold used only by StableAdamW.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument(
        "--lr-schedule",
        choices=["constant", "cosine", "warmup_cosine"],
        default="constant",
        help=(
            "Learning-rate schedule. cosine is flat through "
            "--lr-decay-start-epoch; warmup_cosine linearly warms from zero "
            "through --lr-warmup-epochs and then decays."
        ),
    )
    parser.add_argument(
        "--lr-warmup-epochs",
        type=int,
        default=0,
        help="Linear warmup length for warmup_cosine.",
    )
    parser.add_argument(
        "--lr-decay-start-epoch",
        type=int,
        default=0,
        help="First completed epoch after which cosine decay begins.",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=0.0,
        help="Final learning rate at --total-iters for cosine scheduling.",
    )
    parser.add_argument(
        "--resume-learning-rate",
        type=float,
        default=None,
        help="Override optimizer learning rates after loading a resume checkpoint.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--normal-loss",
        choices=["native", "hard_weighted", "legacy_plain", "inp_soft_mining"],
        default="native",
    )
    parser.add_argument(
        "--normal-loss-weight",
        type=float,
        default=1.0,
        help="Scale the complete normal objective; 0.5 matches the OE normal branch.",
    )
    parser.add_argument("--hard-quantile", type=float, default=0.9)
    parser.add_argument("--easy-weight", type=float, default=0.1)
    parser.add_argument("--dinomaly-native-p-final", type=float, default=0.9)
    parser.add_argument("--dinomaly-native-warmup-iters", type=int, default=1000)
    parser.add_argument("--dinomaly-native-factor", type=float, default=0.1)
    parser.add_argument("--inpformer-native-y", type=float, default=3.0)
    parser.add_argument(
        "--inp-coherence-loss",
        choices=["hard", "soft"],
        default="hard",
        help="INP prototype objective: original nearest-prototype loss or INP-Former++ soft coherence.",
    )
    parser.add_argument("--inp-soft-mining-gamma", type=float, default=3.0)
    parser.add_argument("--masked-recon", action="store_true")
    parser.add_argument("--mask-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--mask-strategy",
        choices=["perspective", "uniform", "uniform_single", "adaptive", "candidate", "prototype_grid"],
        default="perspective",
    )
    parser.add_argument("--mask-band-block-sizes", default="1,1,2,2,4,4")
    parser.add_argument("--mask-fill", choices=["visible_mean", "zero"], default="visible_mean")
    parser.add_argument("--mask-prototype-source", choices=["masked", "full"], default="masked")
    parser.add_argument("--mask-band-loss-weights", default="")
    parser.add_argument("--full-anchor-loss-weight", type=float, default=0.0)
    parser.add_argument("--inp-local-mask-recon", action="store_true", help="For INP-Former, reconstruct masked tokens from local visible context and visible tokens from prototypes.")
    parser.add_argument("--local-context-radius", type=int, default=2)
    parser.add_argument("--local-visible-loss-weight", type=float, default=1.0)
    parser.add_argument("--mask-planner-checkpoint", type=Path, default=None, help="Learned detector/prior mask planner checkpoint.")
    parser.add_argument("--train-mask-planner", action="store_true", help="Also update the mask planner during reconstruction training.")
    parser.add_argument("--adaptive-mask-hidden-dim", type=int, default=192)
    parser.add_argument("--adaptive-mask-ratio", type=float, default=0.25)
    parser.add_argument("--adaptive-mask-segments", type=int, default=4)
    parser.add_argument("--adaptive-mask-temperature", type=float, default=1.0)
    parser.add_argument("--adaptive-planner-loss-weight", type=float, default=0.1)
    parser.add_argument("--adaptive-ratio-loss-weight", type=float, default=5.0)
    parser.add_argument("--adaptive-prior-loss-weight", type=float, default=1.0)
    parser.add_argument("--adaptive-tv-loss-weight", type=float, default=0.05)
    parser.add_argument("--adaptive-binary-loss-weight", type=float, default=0.01)
    parser.add_argument("--adaptive-difficulty-loss-weight", type=float, default=0.2)
    parser.add_argument("--candidate-mask-ratio", type=float, default=0.08)
    parser.add_argument("--candidate-mask-segments", type=int, default=4)
    parser.add_argument("--candidate-min-score", type=float, default=0.0)
    parser.add_argument("--candidate-dilate", type=int, default=1)
    parser.add_argument("--candidate-prior-weight", type=float, default=0.60)
    parser.add_argument("--candidate-mask-prompt", action="store_true")
    parser.add_argument("--prototype-grid-ratio", type=float, default=0.20)
    parser.add_argument("--prototype-grid-threshold", type=float, default=0.0)
    parser.add_argument("--prototype-grid-block-size", type=int, default=2)
    parser.add_argument(
        "--mask-prototype-handling",
        choices=["standard", "masked_frozen", "exclude_detach"],
        default="standard",
    )
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--latest-interval", type=int, default=50, help="Overwrite latest.pth every N steps for resume; 0 disables step-based latest saves.")
    parser.add_argument(
        "--latest-epoch-interval",
        type=int,
        default=1,
        help=(
            "Save a non-best latest.pth every N completed epochs; 0 disables "
            "non-best epoch snapshots. Improved best-loss epochs remain resumable."
        ),
    )
    parser.add_argument(
        "--best-loss-checkpoint-name",
        default="best.pth",
        help="Filename for the checkpoint selected by lowest epoch training loss.",
    )
    parser.add_argument(
        "--final-only-checkpoints",
        action="store_true",
        help=(
            "Do not write best-loss checkpoints. Periodic latest checkpoints may "
            "still be written for interruption recovery, and the completed run "
            "writes model.pth without optimizer/training state."
        ),
    )
    analysis_checkpoints = parser.add_argument_group(
        "analysis checkpoints",
        "Keep model snapshots for post-hoc normal/pseudo-anomaly metric studies.",
    )
    analysis_checkpoints.add_argument(
        "--analysis-checkpoint-every-epochs",
        type=int,
        default=0,
        help=(
            "Save an analysis checkpoint every N completed epochs and at the final "
            "epoch; 0 disables this feature."
        ),
    )
    analysis_checkpoints.add_argument(
        "--analysis-checkpoint-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to output_dir/analysis_checkpoints.",
    )
    analysis_checkpoints.add_argument(
        "--analysis-checkpoint-state",
        choices=["full", "without_encoder"],
        default="without_encoder",
        help=(
            "Store the full model or omit the frozen encoder. The latter remains "
            "strict-loadable through fod_recon_ad.ext.load_checkpoint."
        ),
    )
    parser.add_argument("--resume", type=Path, default=None, help="Resume full training state from a checkpoint produced by this script.")
    parser.add_argument("--auto-resume", action="store_true", help="Resume from output_dir/latest.pth when it exists.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--dist-backend", default="nccl")
    parser.add_argument("--seed", type=int, default=1)
    inline_val = parser.add_argument_group(
        "inline FOD validation",
        "Pause the live training loop at relative epoch boundaries without rebuilding its DataLoader.",
    )
    inline_val.add_argument(
        "--inline-fod-val-every-epochs",
        type=int,
        default=0,
        help="Run in-process FOD val every N relative epochs; 0 disables it.",
    )
    inline_val.add_argument(
        "--inline-fod-val-origin-step",
        type=int,
        default=None,
        help="Step treated as continued epoch 0; defaults to the initial resume step and is saved for later resume.",
    )
    inline_val.add_argument(
        "--inline-fod-val-start-epoch",
        type=int,
        default=0,
        help=(
            "First positive relative epoch to validate. For example, 60 with "
            "--inline-fod-val-every-epochs 10 validates at 60,70,..."
        ),
    )
    inline_val.add_argument("--inline-fod-val-output-dir", type=Path, default=None)
    inline_val.add_argument("--inline-fod-val-include-origin", action="store_true")
    inline_val.add_argument(
        "--inline-fod-val-distances",
        nargs="+",
        default=["05", "10", "15", "20", "25", "30"],
    )
    inline_val.add_argument("--inline-fod-val-stride", type=int, default=0)
    inline_val.add_argument("--inline-fod-val-batch-size", type=int, default=8)
    inline_val.add_argument("--inline-fod-val-progress-every", type=int, default=16)
    inline_val.add_argument("--inline-fod-val-min-pred-area", type=int, default=16)
    inline_val.add_argument("--inline-fod-val-iou-threshold", type=float, default=0.10)
    inline_val.add_argument("--inline-fod-val-seed", type=int, default=20260715)
    inline_val.add_argument("--inline-fod-val-keep-cache", action="store_true")
    add_guided_prototype_args(parser)
    add_blindspot_context_args(parser)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_config(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def init_distributed(args: argparse.Namespace) -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA.")
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
        dist.init_process_group(backend=args.dist_backend)
    return rank, local_rank, world_size


def is_main_process(rank: int) -> bool:
    return rank == 0


def sync_gradients(parameters, world_size: int) -> None:
    if world_size <= 1:
        return
    for param in parameters:
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world_size)


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def resolve_resume_path(args: argparse.Namespace) -> Path | None:
    if args.resume is not None:
        return args.resume
    if args.auto_resume:
        latest = args.output_dir / "latest.pth"
        if latest.exists():
            return latest
    return None


def scheduled_learning_rate(
    *,
    step: int,
    total_iters: int,
    epoch_iters: int,
    base_lr: float,
    schedule: str,
    decay_start_epoch: int,
    min_lr: float,
    warmup_epochs: int = 0,
) -> float:
    """Return the LR for an absolute optimizer step.

    Computing this directly from ``step`` keeps interrupted/resumed runs on the
    same trajectory without serializing a separate scheduler state.
    """

    if schedule == "constant":
        return float(base_lr)
    if schedule == "warmup_cosine":
        warmup_step = int(warmup_epochs) * int(epoch_iters)
        if warmup_step <= 0 or warmup_step >= total_iters:
            raise ValueError(
                "warmup_cosine requires warmup steps strictly between zero and total_iters."
            )
        if step <= warmup_step:
            progress = min(max(step / warmup_step, 0.0), 1.0)
            return float(base_lr * progress)
        progress = min(
            max((step - warmup_step) / (total_iters - warmup_step), 0.0),
            1.0,
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_lr + (base_lr - min_lr) * cosine)
    if schedule != "cosine":
        raise ValueError(f"Unsupported LR schedule: {schedule}")
    decay_start_step = int(decay_start_epoch) * int(epoch_iters)
    if step <= decay_start_step:
        return float(base_lr)
    progress = min(
        max((step - decay_start_step) / (total_iters - decay_start_step), 0.0),
        1.0,
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr + (base_lr - min_lr) * cosine)


def main() -> None:
    args = parse_args()
    rank, _, world_size = init_distributed(args)
    setup_seed(args.seed)
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be positive.")
    if args.optimizer_eps <= 0:
        raise ValueError("--optimizer-eps must be positive.")
    if args.stable_adamw_clip_threshold <= 0:
        raise ValueError("--stable-adamw-clip-threshold must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if args.normal_loss_weight < 0:
        raise ValueError("--normal-loss-weight cannot be negative.")
    if args.checkpoint_allowed_missing_prefixes and not args.strict_load:
        raise ValueError(
            "--checkpoint-allowed-missing-prefixes requires --strict-load."
        )
    if args.trainable_scope != "full" and args.architecture != "inpformer":
        raise ValueError(
            f"--trainable-scope {args.trainable_scope} is only defined for INP-Former."
        )
    if args.blindspot_context:
        if args.architecture != "inpformer":
            raise ValueError("--blindspot-context requires INP-Former.")
        if args.trainable_scope != "blindspot_context":
            raise ValueError(
                "--blindspot-context requires --trainable-scope blindspot_context."
            )
        if args.masked_recon:
            raise ValueError(
                "--blindspot-context is incompatible with masked/INP+ reconstruction."
            )
        if not args.guided_prototype:
            raise ValueError(
                "--blindspot-context requires the Adaptive guided-prototype path."
            )
        if args.checkpoint is None and not args.resume:
            raise ValueError(
                "--blindspot-context is a frozen Adaptive warm-start; provide --checkpoint."
            )
        if args.blindspot_calibration_batches < 0:
            raise ValueError("--blindspot-calibration-batches cannot be negative.")
    elif args.trainable_scope == "blindspot_context":
        raise ValueError(
            "--trainable-scope blindspot_context requires --blindspot-context."
        )
    if args.lr_warmup_epochs < 0:
        raise ValueError("--lr-warmup-epochs cannot be negative.")
    if args.lr_schedule in {"cosine", "warmup_cosine"}:
        if args.epoch_iters <= 0:
            raise ValueError("Cosine LR scheduling requires a positive --epoch-iters.")
        if not 0.0 <= args.min_learning_rate < args.learning_rate:
            raise ValueError("--min-learning-rate must be in [0, --learning-rate) for cosine scheduling.")
        if args.resume_learning_rate is not None:
            raise ValueError(
                "--resume-learning-rate is incompatible with cosine scheduling; "
                "the absolute-step schedule restores the intended LR automatically."
            )
    if args.lr_schedule == "cosine":
        if args.lr_decay_start_epoch < 0:
            raise ValueError("--lr-decay-start-epoch cannot be negative.")
        if args.lr_decay_start_epoch * args.epoch_iters >= args.total_iters:
            raise ValueError("Cosine LR decay must start before --total-iters.")
    elif args.lr_schedule == "warmup_cosine":
        warmup_iters = args.lr_warmup_epochs * args.epoch_iters
        if warmup_iters <= 0 or warmup_iters >= args.total_iters:
            raise ValueError(
                "warmup_cosine requires --lr-warmup-epochs to span fewer than "
                "--total-iters and at least one optimizer step."
            )
    if args.latest_epoch_interval < 0:
        raise ValueError("--latest-epoch-interval must be non-negative.")
    if args.roi_mask_erode_pixels < 0:
        raise ValueError("--roi-mask-erode-pixels must be non-negative.")
    if args.roi_mask_loss and (not args.patch_train or args.ground_roi_mask is None):
        raise ValueError(
            "--roi-mask-loss requires --patch-train and --ground-roi-mask."
        )
    if args.roi_mask_loss and args.architecture == "inpformer":
        if (
            args.guided_prototype_distill_weight > 0.0
            and not args.guided_prototype_roi_aware_loss
        ):
            raise ValueError(
                "Guided distillation is not part of the ROI-masked native-coherence "
                "path; enable --guided-prototype-roi-aware-loss or set its weight to 0."
            )
        if (
            args.guided_prototype_semantic_coverage_weight > 0.0
            and not args.guided_prototype_roi_aware_loss
        ):
            raise ValueError(
                "Semantic prototype coverage is not part of the ROI-masked "
                "native-coherence path; enable --guided-prototype-roi-aware-loss "
                "or set its weight to 0."
            )
    if args.analysis_checkpoint_every_epochs < 0:
        raise ValueError("--analysis-checkpoint-every-epochs must be non-negative.")
    if args.analysis_checkpoint_every_epochs > 0:
        if args.epoch_iters <= 0:
            raise ValueError("Analysis checkpoints require a positive --epoch-iters.")
        if args.total_iters % args.epoch_iters != 0:
            raise ValueError(
                "Analysis checkpoints require --total-iters to end on an epoch boundary."
            )
    best_loss_checkpoint_path = resolve_best_loss_checkpoint_path(
        args.output_dir, args.best_loss_checkpoint_name
    )
    if args.inp_local_mask_recon:
        if args.architecture != "inpformer":
            raise ValueError("--inp-local-mask-recon is only defined for --architecture inpformer.")
        if not args.masked_recon:
            raise ValueError("--inp-local-mask-recon requires --masked-recon.")
        if args.mask_strategy != "adaptive":
            raise ValueError("--inp-local-mask-recon expects --mask-strategy adaptive so masks come from the learned planner.")
        if args.mask_planner_checkpoint is None:
            raise ValueError("--inp-local-mask-recon requires --mask-planner-checkpoint.")
    if args.inp_coherence_loss != "hard" and args.architecture != "inpformer":
        raise ValueError("--inp-coherence-loss is only available for --architecture inpformer.")
    if args.inp_coherence_loss == "soft" and getattr(args, "guided_prototype", False):
        raise ValueError("Soft INP coherence and Guided Prototype define different prototype objectives; enable only one.")
    if args.context_normal_prototype:
        if args.architecture != "inpformer":
            raise ValueError("--context-normal-prototype requires --architecture inpformer.")
        if args.masked_recon:
            raise ValueError("The first context-normal prototype version supports full-token training only.")
        if args.inp_coherence_loss != "hard":
            raise ValueError("Context-normal prototype keeps the original INP hard gather objective.")
    if args.normal_loss == "inp_soft_mining":
        if args.architecture != "inpformer":
            raise ValueError("--normal-loss inp_soft_mining is only available for --architecture inpformer.")
        if args.masked_recon:
            raise ValueError("INP-Former++ soft mining currently requires full-token normal training.")
    if args.mask_strategy == "adaptive" and args.mask_band_loss_weights:
        raise ValueError("--mask-band-loss-weights is a perspective prior and is disabled for adaptive masks.")
    if args.mask_strategy in {"uniform", "uniform_single"}:
        uniform_blocks = parse_block_sizes(args.mask_band_block_sizes)
        if len(uniform_blocks) != 1:
            raise ValueError(
                f"--mask-strategy {args.mask_strategy} requires exactly one "
                "--mask-band-block-sizes value."
            )
        if args.mask_band_loss_weights:
            raise ValueError(
                "--mask-band-loss-weights is a perspective prior and is disabled "
                f"for {args.mask_strategy}."
            )
    if args.mask_strategy in {"candidate", "prototype_grid"} and args.architecture != "inpformer":
        raise ValueError(
            f"--mask-strategy {args.mask_strategy} is currently implemented for INP-Former."
        )
    if args.mask_prototype_handling != "standard":
        if args.mask_strategy != "prototype_grid":
            raise ValueError("--mask-prototype-handling requires --mask-strategy prototype_grid.")
        if args.trainable_scope != "reconstruction_only":
            raise ValueError(
                "Frozen mask prototype handling requires --trainable-scope reconstruction_only."
            )
    if args.inline_fod_val_every_epochs < 0:
        raise ValueError("--inline-fod-val-every-epochs cannot be negative.")
    if args.inline_fod_val_every_epochs > 0:
        if args.dataset != "fod":
            raise ValueError("Inline FOD validation requires --dataset fod.")
        if args.epoch_iters <= 0:
            raise ValueError("Inline FOD validation requires a positive --epoch-iters.")
        if args.inline_fod_val_output_dir is None:
            raise ValueError(
                "--inline-fod-val-output-dir is required when inline validation is enabled."
            )
        if args.inline_fod_val_batch_size <= 0:
            raise ValueError("--inline-fod-val-batch-size must be positive.")
        if args.inline_fod_val_start_epoch < 0:
            raise ValueError("--inline-fod-val-start-epoch cannot be negative.")
        if args.inline_fod_val_stride < 0:
            raise ValueError("--inline-fod-val-stride cannot be negative.")
        if not 0.0 <= args.inline_fod_val_iou_threshold <= 1.0:
            raise ValueError("--inline-fod-val-iou-threshold must be in [0,1].")
    if not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"
    if args.dataset == "fod":
        args.train_distance = normalize_distance(args.train_distance)
    if is_main_process(rank):
        write_config(args)
    if world_size > 1:
        dist.barrier()

    if args.dataset == "mvtec":
        if args.patch_train:
            raise ValueError("--patch-train is not implemented for --dataset mvtec.")
        loader = make_mvtec_train_loader(
            root=args.mvtec_root,
            category=args.mvtec_category,
            image_size=args.image_size,
            crop_size=args.crop_size,
            physical_crop_size=args.mvtec_physical_crop_size,
            batch_size=args.batch_size,
            workers=args.workers,
            shuffle=True,
        )
        if is_main_process(rank):
            print(f"[data] mvtec category={args.mvtec_category} train_images={len(loader.dataset)}", flush=True)
    elif args.patch_train:
        loader = make_crop_train_loader(
            root=args.fod_root,
            distance=args.train_distance,
            full_height=args.patch_full_height,
            crop_size=args.crop_size,
            output_size=args.patch_output_size if args.patch_output_size > 0 else None,
            stride=args.patch_stride,
            batch_size=args.batch_size,
            workers=args.workers,
            layout=args.layout,
            shuffle=True,
            drop_last=args.train_drop_last,
            roi_mask_path=args.ground_roi_mask,
            min_roi_coverage=args.ground_roi_min_coverage,
            return_roi_mask=args.roi_mask_loss,
            roi_erode_pixels=args.roi_mask_erode_pixels,
        )
        if is_main_process(rank):
            effective_batch = args.batch_size * world_size * args.grad_accum_steps
            print(
                f"[data] patch_train crops={len(loader.dataset)} micro_batch={args.batch_size * world_size} "
                f"grad_accum={args.grad_accum_steps} effective_batch={effective_batch}",
                flush=True,
            )
    else:
        loader = make_train_loader(
            root=args.fod_root,
            distance=args.train_distance,
            image_size=args.image_size,
            crop_size=args.crop_size,
            batch_size=args.batch_size,
            workers=args.workers,
            layout=args.layout,
            shuffle=True,
        )
        if is_main_process(rank):
            effective_batch = args.batch_size * world_size * args.grad_accum_steps
            print(
                f"[data] full_train images={len(loader.dataset)} micro_batch={args.batch_size * world_size} "
                f"grad_accum={args.grad_accum_steps} effective_batch={effective_batch}",
                flush=True,
            )
    model, trainable = build_reconstruction_model(
        architecture=args.architecture,
        dinomaly_repo=args.external_repo,
        inpformer_repo=args.inpformer_repo,
        encoder=args.encoder,
        device=device,
        inp_num=args.inp_num,
        mamba_layers=args.mamba_layers,
        mamba_scan=args.mamba_scan,
        mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand,
        mamba_bidirectional=not args.mamba_unidirectional,
        mamba_drop=args.mamba_drop,
        mamba_multi_output=args.mamba_multi_output,
    )
    if args.architecture == "inpformer":
        configure_inp_coherence(model, args.inp_coherence_loss)
    configure_guided_prototypes(model, args, args.architecture)
    if args.blindspot_context:
        attach_blindspot_context_head(
            model,
            hidden_dim=args.blindspot_context_hidden_dim,
            radius=args.blindspot_context_radius,
            prototype_temperature=args.blindspot_context_prototype_temperature,
        )
    if args.trainable_scope == "full":
        for module in guided_prototype_trainable_modules(model):
            trainable.append(module)
    trainable = select_reconstruction_trainable_modules(
        model,
        trainable,
        scope=args.trainable_scope,
    )
    if args.masked_recon and args.mask_strategy == "adaptive":
        planner = attach_adaptive_mask_planner(model, hidden_dim=args.adaptive_mask_hidden_dim)
        if args.mask_planner_checkpoint is not None:
            planner.load_state_dict(load_planner_state(args.mask_planner_checkpoint, device), strict=True)
        if args.inp_local_mask_recon and not args.train_mask_planner:
            planner.eval()
            for param in planner.parameters():
                param.requires_grad_(False)
        else:
            trainable.append(planner)
    if args.inp_local_mask_recon:
        local_reconstructor = attach_local_context_reconstructor(model, radius=args.local_context_radius)
        trainable.append(local_reconstructor)
    if args.masked_recon and args.mask_strategy == "candidate" and args.candidate_mask_prompt:
        candidate_prompt = attach_candidate_mask_prompt(model)
        trainable.append(candidate_prompt)
    optimizer_kwargs = {
        "lr": args.learning_rate,
        "weight_decay": args.weight_decay,
        "betas": (0.9, 0.999),
        "eps": args.optimizer_eps,
        "amsgrad": args.optimizer_amsgrad,
    }
    if args.optimizer == "stable_adamw":
        StableAdamW = load_stable_adamw(args.inpformer_repo)
        optimizer = StableAdamW(
            trainable.parameters(),
            clip_threshold=args.stable_adamw_clip_threshold,
            **optimizer_kwargs,
        )
    else:
        optimizer = torch.optim.AdamW(trainable.parameters(), **optimizer_kwargs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if is_main_process(rank):
        print(
            f"[optimizer] name={args.optimizer} eps={args.optimizer_eps:.3e} "
            f"amsgrad={int(args.optimizer_amsgrad)} weight_decay={args.weight_decay:.3e} "
            f"stable_clip_threshold={args.stable_adamw_clip_threshold:g}",
            flush=True,
        )
        if args.lr_schedule == "cosine":
            print(
                f"[lr] schedule=cosine base={args.learning_rate:.6g} "
                f"decay_start_epoch={args.lr_decay_start_epoch} "
                f"decay_start_iter={args.lr_decay_start_epoch * args.epoch_iters} "
                f"min={args.min_learning_rate:.6g} total_iters={args.total_iters}",
                flush=True,
            )
        elif args.lr_schedule == "warmup_cosine":
            print(
                f"[lr] schedule=warmup_cosine base={args.learning_rate:.6g} "
                f"warmup_epochs={args.lr_warmup_epochs} "
                f"warmup_iters={args.lr_warmup_epochs * args.epoch_iters} "
                f"min={args.min_learning_rate:.6g} total_iters={args.total_iters}",
                flush=True,
            )
        else:
            print(f"[lr] schedule=constant value={args.learning_rate:.6g}", flush=True)

    log_path = args.output_dir / "train_log.csv"
    resume_path = resolve_resume_path(args)
    if args.context_normal_prototype and resume_path is None and args.checkpoint is None:
        raise ValueError(
            "Context-normal prototype is a Native warm-start adaptation; provide --checkpoint "
            "or resume an existing context-normal run."
        )
    start_step = 0
    best_epoch_loss = float("inf")
    epoch_losses = []
    resumed_inline_val_origin_step: int | None = None
    resumed_inline_val_start_epoch: int | None = None
    resumed_inline_val_every_epochs: int | None = None
    if resume_path is None and args.checkpoint is not None:
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"Warm-start checkpoint not found: {args.checkpoint}")
        allowed_missing_prefixes = tuple(
            value.strip()
            for value in args.checkpoint_allowed_missing_prefixes.split(",")
            if value.strip()
        )
        if allowed_missing_prefixes:
            load_checkpoint(
                model,
                args.checkpoint,
                device,
                strict=True,
                allowed_missing_prefixes=allowed_missing_prefixes,
            )
            if is_main_process(rank):
                print(
                    f"[warm_start] checkpoint={args.checkpoint} "
                    f"allowed_missing_prefixes={allowed_missing_prefixes}",
                    flush=True,
                )
        else:
            state = torch.load(args.checkpoint, map_location=device)
            model_state = state["model"] if isinstance(state, dict) and "model" in state else state
            missing, unexpected = model.load_state_dict(model_state, strict=args.strict_load)
            if is_main_process(rank):
                print(
                    f"[warm_start] checkpoint={args.checkpoint} "
                    f"missing={len(missing)} unexpected={len(unexpected)}",
                    flush=True,
                )
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        state = torch.load(resume_path, map_location=device)
        model_state = state["model"] if isinstance(state, dict) and "model" in state else state
        model.load_state_dict(model_state, strict=True)
        if not isinstance(state, dict) or "optimizer" not in state:
            raise RuntimeError(f"Checkpoint {resume_path} does not contain optimizer state; use it as a model checkpoint, not --resume.")
        metadata = state.get("metadata", {})
        saved_optimizer = metadata.get("optimizer")
        if saved_optimizer is not None and saved_optimizer != args.optimizer:
            raise ValueError(
                "Optimizer differs from the resume checkpoint: "
                f"checkpoint={saved_optimizer}, args={args.optimizer}."
            )
        saved_lr_schedule = metadata.get("lr_schedule")
        if saved_lr_schedule is not None and saved_lr_schedule != args.lr_schedule:
            raise ValueError(
                "LR schedule differs from the resume checkpoint: "
                f"checkpoint={saved_lr_schedule}, args={args.lr_schedule}."
            )
        saved_warmup_epochs = metadata.get("lr_warmup_epochs")
        if (
            saved_warmup_epochs is not None
            and int(saved_warmup_epochs) != args.lr_warmup_epochs
        ):
            raise ValueError(
                "LR warmup differs from the resume checkpoint: "
                f"checkpoint={saved_warmup_epochs}, args={args.lr_warmup_epochs}."
            )
        optimizer.load_state_dict(state["optimizer"])
        if args.resume_learning_rate is not None:
            if args.resume_learning_rate <= 0:
                raise ValueError("--resume-learning-rate must be positive.")
            for param_group in optimizer.param_groups:
                param_group["lr"] = float(args.resume_learning_rate)
            if is_main_process(rank):
                print(
                    f"[resume] override_learning_rate={args.resume_learning_rate:.6g}",
                    flush=True,
                )
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        training_state = state.get("training_state", {})
        start_step = int(metadata.get("iter", training_state.get("step", 0)))
        best_epoch_loss = float(training_state.get("best_epoch_loss", metadata.get("best_loss", float("inf"))))
        epoch_losses = [float(value) for value in training_state.get("epoch_losses", [])]
        if training_state.get("inline_fod_val_origin_step") is not None:
            resumed_inline_val_origin_step = int(training_state["inline_fod_val_origin_step"])
        if training_state.get("inline_fod_val_start_epoch") is not None:
            resumed_inline_val_start_epoch = int(training_state["inline_fod_val_start_epoch"])
        if training_state.get("inline_fod_val_every_epochs") is not None:
            resumed_inline_val_every_epochs = int(training_state["inline_fod_val_every_epochs"])
        restore_rng_state(training_state.get("rng_state"))
        if is_main_process(rank):
            print(f"[resume] checkpoint={resume_path} start_step={start_step} best_loss={best_epoch_loss}", flush=True)
        del model_state, metadata, training_state, state
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.epoch_iters > 0 and start_step % args.epoch_iters == 0 and epoch_losses:
        # Older boundary checkpoints were written before this list was cleared.
        # Those values belong to the completed epoch and must not leak into the
        # first epoch after resume.
        if is_main_process(rank):
            print(
                f"[resume] discard_completed_epoch_losses={len(epoch_losses)} at step={start_step}",
                flush=True,
            )
        epoch_losses = []

    inline_val_origin_step: int | None = None
    if args.inline_fod_val_every_epochs > 0:
        if (
            resumed_inline_val_start_epoch is not None
            and resumed_inline_val_start_epoch != args.inline_fod_val_start_epoch
        ):
            raise ValueError(
                "Inline validation start epoch differs from the resume checkpoint: "
                f"checkpoint={resumed_inline_val_start_epoch}, args={args.inline_fod_val_start_epoch}."
            )
        if (
            resumed_inline_val_every_epochs is not None
            and resumed_inline_val_every_epochs != args.inline_fod_val_every_epochs
        ):
            raise ValueError(
                "Inline validation interval differs from the resume checkpoint: "
                f"checkpoint={resumed_inline_val_every_epochs}, "
                f"args={args.inline_fod_val_every_epochs}."
            )
        if args.inline_fod_val_origin_step is not None:
            inline_val_origin_step = args.inline_fod_val_origin_step
        elif resumed_inline_val_origin_step is not None:
            inline_val_origin_step = resumed_inline_val_origin_step
        else:
            inline_val_origin_step = start_step
        if inline_val_origin_step < 0 or inline_val_origin_step > start_step:
            raise ValueError(
                "Inline validation origin must be non-negative and no later than the current start step: "
                f"origin={inline_val_origin_step}, start_step={start_step}."
            )
        if inline_val_origin_step % args.epoch_iters != 0:
            raise ValueError(
                "Inline validation origin must be an --epoch-iters boundary: "
                f"origin={inline_val_origin_step}, epoch_iters={args.epoch_iters}."
            )

    descriptor_source_manifest = getattr(
        args, "guided_prototype_descriptor_source_manifest", None
    )
    source_group_resolver = SourceGroupResolver(
        [descriptor_source_manifest] if descriptor_source_manifest is not None else []
    )
    if (
        args.context_normal_prototype
        and args.context_normal_prototype_mix > 0.0
        and not context_normal_memory_ready(model)
    ):
        memory_diag = fit_context_normal_memory(
            model,
            loader,
            device=device,
            source_group_resolver=source_group_resolver,
        )
        if world_size > 1:
            for buffer in model.context_normal_memory.buffers():
                dist.broadcast(buffer, src=0)
        if is_main_process(rank):
            print(
                "[context_normal_memory] "
                + " ".join(f"{key}={value:g}" for key, value in memory_diag.items()),
                flush=True,
            )

    guided_config = getattr(model, "_guided_prototype_config", None)
    context_transport_enabled = bool(
        args.guided_prototype and getattr(guided_config, "context_transport", False)
    )
    if context_transport_enabled and not context_familiarity_transport_memory_ready(model):
        config = model._guided_prototype_config
        memory_diag = {}
        if is_main_process(rank):
            memory_diag = fit_context_familiarity_transport_memory(
                model,
                loader,
                device=device,
                source_group_resolver=source_group_resolver,
                candidates_per_image=config.context_candidates_per_image,
                candidates_per_group=config.context_candidates_per_group,
                memory_build_batches=config.context_memory_build_batches,
            )
        if world_size > 1:
            for buffer in model.guided_context_transport_memory.buffers():
                dist.broadcast(buffer, src=0)
        if is_main_process(rank):
            print(
                "[guided_context_transport_memory] "
                + " ".join(f"{key}={value:g}" for key, value in memory_diag.items()),
                flush=True,
            )

    descriptor_enabled = bool(
        args.guided_prototype
        and getattr(guided_config, "descriptor_variant", "off") != "off"
    )
    if descriptor_enabled and not normal_descriptor_memory_ready(model):
        config = model._guided_prototype_config
        memory_diag = {}
        if is_main_process(rank):
            memory_diag = fit_normal_descriptor_memory(
                model,
                loader,
                device=device,
                source_group_resolver=source_group_resolver,
                candidates_per_image=config.descriptor_candidates_per_image,
                candidates_per_group=config.descriptor_candidates_per_group,
                memory_build_batches=config.descriptor_memory_build_batches,
            )
        if world_size > 1:
            for buffer in model.guided_normal_descriptor_memory.buffers():
                dist.broadcast(buffer, src=0)
        if is_main_process(rank):
            print(
                "[guided_normal_descriptor_memory] "
                + " ".join(f"{key}={value:g}" for key, value in memory_diag.items()),
                flush=True,
            )

    if is_main_process(rank) and (not resume_path or not log_path.exists()):
        with log_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "iter",
                    "loss",
                    "lr",
                    "seconds",
                    "guided_prior_bg",
                    "guided_prior_texture",
                    "guided_prior_object",
                    "guided_assign_bg",
                    "guided_assign_texture",
                    "guided_assign_object",
                    "guided_proto_sep",
                    "guided_intra_balance",
                    "guided_intra_min_soft_usage",
                    "guided_intra_repulsion",
                    "guided_intra_similarity",
                    "guided_memory_mode_loss",
                    "guided_memory_mode_accuracy",
                    "guided_memory_mode_min_usage",
                    *CENTER6_DIAGNOSTIC_FIELDS,
                    *DESCRIPTOR_DIAGNOSTIC_FIELDS,
                    "guided_valid_ratio",
                    "guided_valid_bg",
                    "guided_valid_texture",
                    "guided_valid_object",
                    "guided_confidence",
                    "guided_gate_mean",
                    "guided_gate_min",
                    "guided_gate_suppressed",
                    "guided_familiarity_count",
                    "guided_novelty_mean",
                    "guided_novelty_distance",
                    "guided_pollution_risk",
                    "guided_novelty_threshold",
                    "guided_native_anchor_alpha",
                    "guided_distill_weight",
                    "guided_semantic_coverage_weight",
                    "guided_semantic_coverage_effective_weight",
                    "guided_semantic_coverage_min_role_mass",
                    "guided_semantic_coverage_loss",
                    "guided_semantic_coverage_active_roles",
                    "guided_semantic_coverage_active_image_fraction",
                    "guided_semantic_coverage_matched_similarity_mean",
                    "guided_semantic_coverage_matched_similarity_min",
                    "guided_semantic_coverage_matched_slot_fraction",
                    "guided_semantic_coverage_matched_slot_entropy",
                    "guided_semantic_coverage_role_mass_0",
                    "guided_semantic_coverage_role_mass_1",
                    "guided_semantic_coverage_role_mass_2",
                    "guided_semantic_coverage_role_active_0",
                    "guided_semantic_coverage_role_active_1",
                    "guided_semantic_coverage_role_active_2",
                    "guided_aggregation_alpha",
                    "guided_mode_routing_active",
                    "guided_mode_routing_strength",
                    "guided_mode_routing_floor",
                    "guided_mode_routing_factor_mean",
                    "guided_mode_routing_factor_min",
                    "guided_mode_routing_factor_max",
                    "guided_mode_routing_factor_std",
                    "guided_proto_loss",
                    "guided_native_gather",
                    "guided_group_gather",
                    "roi_masked_gather_loss",
                    "roi_aware_guided_gather",
                    "guided_calibration_groups",
                    "guided_calibrated_count",
                    "guided_prior_anchor",
                    "guided_normal_guard",
                    "guided_guard_rec",
                    "guided_guard_obj",
                    "guided_transport_gate",
                    "guided_transport_confidence",
                    "guided_transport_surprise",
                    "guided_transport_objectness",
                    "guided_transport_source_diversity",
                    "guided_transport_shift",
                    "guided_transport_scale_entropy",
                    "guided_transport_fallback_ratio",
                    "guided_transport_memory_count",
                    "inp_coherence_loss",
                    "inp_soft_mining_cosine",
                    "inp_soft_mining_mse",
                    "context_normal_mix",
                    "context_normal_memory_count",
                    "context_normal_similarity",
                    "context_normal_shift",
                    "context_normal_unexcluded_ratio",
                ],
            )
            writer.writeheader()

    if world_size > 1:
        setup_seed(args.seed + rank)
    iterator = iter(loader)
    use_source_groups = bool(
        args.context_normal_prototype
        or context_transport_enabled
        or descriptor_enabled
        or (
            args.guided_prototype
            and args.guided_prototype_familiarity_gate
            and args.guided_prototype_familiarity_calibration == "cross_group"
        )
    )
    start_time = time.time()
    recent_losses = []
    model.train()
    if args.blindspot_context:
        # The Adaptive mainline is an immutable teacher for this short
        # normal-only fit. Only the newly attached head remains in train mode.
        model.eval()
        model.blindspot_context_head.train()
        model._guided_update_prior_stats = False
    if args.mask_prototype_handling in {"masked_frozen", "exclude_detach"}:
        # Keep the proposal teacher, familiarity bank, and normalizers identical
        # across the causal mask/prototype ablation, including full-anchor passes.
        model._guided_update_prior_stats = False
    if args.inp_local_mask_recon and not args.train_mask_planner and hasattr(model, "adaptive_mask_planner"):
        model.adaptive_mask_planner.eval()
    mask_config = PerspectiveMaskConfig(
        strategy=args.mask_strategy,
        band_block_sizes=parse_block_sizes(args.mask_band_block_sizes),
        fill=args.mask_fill,
        prototype_source=args.mask_prototype_source,
        adaptive_mask_ratio=args.adaptive_mask_ratio,
        adaptive_segments=args.adaptive_mask_segments,
        adaptive_temperature=args.adaptive_mask_temperature,
        inp_mask_mode="local_context" if args.inp_local_mask_recon else "standard",
        local_context_radius=args.local_context_radius,
        candidate_mask_ratio=args.candidate_mask_ratio,
        candidate_segments=args.candidate_mask_segments,
        candidate_min_score=args.candidate_min_score,
        candidate_dilate=args.candidate_dilate,
        candidate_prior_weight=args.candidate_prior_weight,
        candidate_prompt=args.candidate_mask_prompt,
        prototype_grid_ratio=args.prototype_grid_ratio,
        prototype_grid_threshold=args.prototype_grid_threshold,
        prototype_grid_block_size=args.prototype_grid_block_size,
        prototype_handling=args.mask_prototype_handling,
    )
    mask_band_loss_weights = parse_float_list(args.mask_band_loss_weights) if args.mask_band_loss_weights else None

    def save_checkpoint(
        path: Path,
        step: int,
        include_training_state: bool = False,
        omit_model_state_prefixes: tuple[str, ...] = (),
        **metadata,
    ) -> None:
        payload = checkpoint_payload(
            model,
            omit_model_state_prefixes=omit_model_state_prefixes,
            iter=step,
            encoder=args.encoder,
            architecture=args.architecture,
            normal_loss=args.normal_loss,
            inp_coherence_loss=args.inp_coherence_loss,
            inp_soft_mining_gamma=args.inp_soft_mining_gamma,
            inp_num=args.inp_num,
            mamba_layers=args.mamba_layers,
            mamba_scan=args.mamba_scan,
            mamba_d_state=args.mamba_d_state,
            mamba_d_conv=args.mamba_d_conv,
            mamba_expand=args.mamba_expand,
            mamba_bidirectional=not args.mamba_unidirectional,
            mamba_multi_output=args.mamba_multi_output,
            optimizer=args.optimizer,
            optimizer_eps=args.optimizer_eps,
            optimizer_amsgrad=args.optimizer_amsgrad,
            stable_adamw_clip_threshold=args.stable_adamw_clip_threshold,
            learning_rate=args.learning_rate,
            lr_schedule=args.lr_schedule,
            lr_warmup_epochs=args.lr_warmup_epochs,
            lr_decay_start_epoch=args.lr_decay_start_epoch,
            min_learning_rate=args.min_learning_rate,
            **metadata,
        )
        if include_training_state:
            payload["optimizer"] = optimizer.state_dict()
            payload["scaler"] = scaler.state_dict()
            payload["training_state"] = {
                "step": int(step),
                "best_epoch_loss": float(best_epoch_loss),
                "epoch_losses": [float(value) for value in epoch_losses],
                "inline_fod_val_origin_step": inline_val_origin_step,
                "inline_fod_val_start_epoch": (
                    args.inline_fod_val_start_epoch
                    if args.inline_fod_val_every_epochs > 0
                    else None
                ),
                "inline_fod_val_every_epochs": (
                    args.inline_fod_val_every_epochs
                    if args.inline_fod_val_every_epochs > 0
                    else None
                ),
                "rng_state": capture_rng_state(),
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

    analysis_checkpoint_dir = (
        args.analysis_checkpoint_dir
        if args.analysis_checkpoint_dir is not None
        else args.output_dir / "analysis_checkpoints"
    )
    analysis_total_epochs = (
        args.total_iters // args.epoch_iters
        if args.analysis_checkpoint_every_epochs > 0
        else 0
    )

    def save_analysis_checkpoint(
        *,
        epoch: int,
        step: int,
        epoch_loss: float,
        learning_rate: float,
    ) -> None:
        if not analysis_checkpoint_due(
            epoch,
            analysis_total_epochs,
            args.analysis_checkpoint_every_epochs,
        ):
            return
        omitted_prefixes: tuple[str, ...] = ()
        if args.analysis_checkpoint_state == "without_encoder":
            omitted_prefixes = ("encoder.",)
            if not any(
                key.startswith(omitted_prefixes) for key in model.state_dict()
            ):
                raise RuntimeError(
                    "--analysis-checkpoint-state without_encoder requires model "
                    "state keys under the encoder. prefix."
                )

        analysis_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = analysis_checkpoint_dir / f"epoch_{epoch:03d}.pth"
        temporary_path = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
        temporary_path.unlink(missing_ok=True)
        try:
            save_checkpoint(
                temporary_path,
                step,
                include_training_state=False,
                omit_model_state_prefixes=omitted_prefixes,
                checkpoint_kind="analysis",
                epoch=epoch,
                epoch_loss=epoch_loss,
                current_learning_rate=learning_rate,
                best_loss=best_epoch_loss,
                analysis_checkpoint_state=args.analysis_checkpoint_state,
                training_config=str(args.output_dir / "config.json"),
            )
            os.replace(temporary_path, checkpoint_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        manifest_path = analysis_checkpoint_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = {
                "version": 1,
                "training_config": str(args.output_dir / "config.json"),
                "checkpoint_state": args.analysis_checkpoint_state,
                "checkpoints": [],
            }
        record = {
            "checkpoint": checkpoint_path.name,
            "epoch": int(epoch),
            "iter": int(step),
            "epoch_loss": float(epoch_loss),
            "learning_rate": float(learning_rate),
            "best_loss": float(best_epoch_loss),
            "size_bytes": checkpoint_path.stat().st_size,
        }
        records = [
            item
            for item in manifest.get("checkpoints", [])
            if int(item["epoch"]) != epoch
        ]
        records.append(record)
        manifest["checkpoints"] = sorted(records, key=lambda item: int(item["epoch"]))
        temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)
        print(
            f"[analysis_checkpoint] epoch={epoch} iter={step} "
            f"state={args.analysis_checkpoint_state} saved {checkpoint_path}",
            flush=True,
        )

    inline_val_schedule: InlineValidationSchedule | None = None
    inline_validator: InlineFODValidator | None = None
    if args.inline_fod_val_every_epochs > 0:
        assert inline_val_origin_step is not None
        inline_val_schedule = InlineValidationSchedule(
            origin_step=inline_val_origin_step,
            epoch_iters=args.epoch_iters,
            every_epochs=args.inline_fod_val_every_epochs,
            include_origin=args.inline_fod_val_include_origin,
            start_epoch=args.inline_fod_val_start_epoch,
        )
        if is_main_process(rank):
            inline_validator = InlineFODValidator(args, inline_val_schedule)

    def run_inline_validation(step: int) -> None:
        if inline_val_schedule is None or not inline_val_schedule.should_validate(step):
            return
        # Gradients are cleared at the beginning of the next optimizer step
        # anyway. Releasing them here gives validation more memory headroom
        # without changing the optimization trajectory.
        optimizer.zero_grad(set_to_none=True)
        if world_size > 1:
            dist.barrier()
        validation_error: BaseException | None = None
        if is_main_process(rank):
            assert inline_validator is not None

            def save_validation_checkpoint(path: Path, checkpoint_step: int) -> None:
                save_checkpoint(
                    path,
                    checkpoint_step,
                    include_training_state=True,
                    checkpoint_kind="inline_val",
                    continued_epoch=inline_val_schedule.continued_epoch(checkpoint_step),
                    best_loss=best_epoch_loss,
                )

            try:
                inline_validator.validate(model, step, save_validation_checkpoint)
            except BaseException as exc:  # Broadcast failure before unwinding distributed workers.
                validation_error = exc
        if world_size > 1:
            status = torch.tensor(
                [1 if validation_error is not None else 0],
                dtype=torch.int32,
                device=device,
            )
            dist.broadcast(status, src=0)
            if int(status.item()) != 0:
                if validation_error is not None:
                    raise validation_error
                raise RuntimeError("Inline FOD validation failed on rank 0.")
        elif validation_error is not None:
            raise validation_error

    def next_images() -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        nonlocal iterator
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        if args.roi_mask_loss:
            images, _, paths, roi_mask = batch
            roi_mask = roi_mask.to(device, non_blocking=True)
        else:
            images, _, paths = batch
            roi_mask = None
        source_group_ids = source_group_resolver.ids(paths, device=device) if use_source_groups else None
        return images.to(device, non_blocking=True), source_group_ids, roi_mask

    def normal_loss_from_output(
        output,
        loss_step: int,
        roi_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if roi_mask is None:
            loss, diagnostics = compute_normal_loss(
                output,
                architecture=args.architecture,
                step=loss_step,
                normal_loss=args.normal_loss,
                prototype_loss_weight=args.prototype_loss_weight,
                hard_quantile=args.hard_quantile,
                easy_weight=args.easy_weight,
                dinomaly_p_final=args.dinomaly_native_p_final,
                dinomaly_warmup_iters=args.dinomaly_native_warmup_iters,
                dinomaly_factor=args.dinomaly_native_factor,
                inpformer_y=args.inpformer_native_y,
                inpformer_soft_mining_gamma=args.inp_soft_mining_gamma,
            )
        else:
            loss, diagnostics = compute_roi_masked_normal_loss(
                output,
                roi_mask,
                architecture=args.architecture,
                step=loss_step,
                normal_loss=args.normal_loss,
                prototype_loss_weight=args.prototype_loss_weight,
                gather_distance=getattr(model, "distance", None),
                roi_aware_gather_loss=(
                    output[2]
                    if getattr(args, "guided_prototype_roi_aware_loss", False)
                    else None
                ),
                hard_quantile=args.hard_quantile,
                easy_weight=args.easy_weight,
                dinomaly_p_final=args.dinomaly_native_p_final,
                dinomaly_warmup_iters=args.dinomaly_native_warmup_iters,
                dinomaly_factor=args.dinomaly_native_factor,
                inpformer_y=args.inpformer_native_y,
            )
        model._normal_loss_diag = diagnostics
        return loss

    def add_guided_extra_loss(
        loss: torch.Tensor,
        en: Sequence[torch.Tensor],
        de: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        extra_loss, extra_diag = guided_prototype_extra_loss(model, en, de)
        if extra_diag:
            model._guided_extra_diag = extra_diag
        return loss + extra_loss

    def compute_loss(
        images: torch.Tensor,
        micro_step: int,
        source_group_ids: torch.Tensor | None,
        roi_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        set_guided_prototype_source_groups(model, source_group_ids)
        set_guided_prototype_valid_roi(model, roi_mask)
        set_context_normal_source_groups(model, source_group_ids)
        loss_step = micro_step + 1
        model._guided_training_step = loss_step
        if args.blindspot_context:
            set_guided_prototype_image(model, images)
            with torch.no_grad():
                _, _, context = inpformer_forward_with_prototype_context(
                    model, images
                )
            score = blindspot_context_score(
                model.blindspot_context_head,
                context["target_tokens"],
                context["agg_prototype"],
                side=context["side"],
            )
            model._blindspot_context_diag = {
                "mean": float(score.detach().mean().item()),
                "q99": float(
                    torch.quantile(score.detach().float().flatten(), 0.99).item()
                ),
            }
            return score.mean()
        if args.masked_recon:
            segment_count = (
                4
                if args.mask_strategy == "prototype_grid"
                else args.candidate_mask_segments
                if args.mask_strategy == "candidate"
                else args.adaptive_mask_segments
            )
            pattern = micro_step % max(1, segment_count)
            set_guided_prototype_image(model, images)
            en, de, mask = forward_masked_reconstruction(model, images, pattern=pattern, config=mask_config)
            if (
                args.mask_strategy == "prototype_grid"
                and is_main_process(rank)
                and (micro_step == 0 or (micro_step + 1) % args.log_interval == 0)
            ):
                grid_state = getattr(model, "_prototype_grid_mask_state", {})
                score = grid_state.get("score")
                block_score = grid_state.get("block_score")
                support = grid_state.get("support")
                if (
                    isinstance(score, torch.Tensor)
                    and isinstance(block_score, torch.Tensor)
                    and isinstance(support, torch.Tensor)
                ):
                    quantiles = torch.quantile(
                        score.detach().float().flatten(),
                        score.new_tensor([0.5, 0.9, 0.95]).float(),
                    )
                    block_quantiles = torch.quantile(
                        block_score.detach().float().flatten(),
                        block_score.new_tensor([0.5, 0.9, 0.95]).float(),
                    )
                    print(
                        "[prototype_grid] "
                        f"step={micro_step + 1} pattern={pattern} "
                        f"support={support.float().mean().item():.4f} "
                        f"mask={mask.float().mean().item():.4f} "
                        f"risk_q50={quantiles[0].item():.5f} "
                        f"risk_q90={quantiles[1].item():.5f} "
                        f"risk_q95={quantiles[2].item():.5f} "
                        f"block_q50={block_quantiles[0].item():.5f} "
                        f"block_q90={block_quantiles[1].item():.5f} "
                        f"block_q95={block_quantiles[2].item():.5f}",
                        flush=True,
                    )
            masked_gather_loss = getattr(model, "_masked_gather_loss", None)
            if args.inp_local_mask_recon:
                masked_loss, _ = compute_masked_normal_loss(
                    en,
                    de,
                    mask=mask,
                    architecture=args.architecture,
                    step=loss_step,
                    normal_loss=args.normal_loss,
                    prototype_loss_weight=args.prototype_loss_weight,
                    gather_loss=None,
                    hard_quantile=args.hard_quantile,
                    easy_weight=args.easy_weight,
                    band_weights=None,
                    dinomaly_p_final=args.dinomaly_native_p_final,
                    dinomaly_warmup_iters=args.dinomaly_native_warmup_iters,
                    dinomaly_factor=args.dinomaly_native_factor,
                    inpformer_y=args.inpformer_native_y,
                )
                visible_loss, _ = compute_masked_normal_loss(
                    en,
                    de,
                    mask=~mask.bool(),
                    architecture=args.architecture,
                    step=loss_step,
                    normal_loss=args.normal_loss,
                    prototype_loss_weight=args.prototype_loss_weight,
                    gather_loss=None,
                    hard_quantile=args.hard_quantile,
                    easy_weight=args.easy_weight,
                    band_weights=None,
                    dinomaly_p_final=args.dinomaly_native_p_final,
                    dinomaly_warmup_iters=args.dinomaly_native_warmup_iters,
                    dinomaly_factor=args.dinomaly_native_factor,
                    inpformer_y=args.inpformer_native_y,
                )
                loss = args.mask_loss_weight * masked_loss + args.local_visible_loss_weight * visible_loss
                if masked_gather_loss is not None:
                    loss = loss + args.prototype_loss_weight * masked_gather_loss
                loss = add_guided_extra_loss(loss, en, de)
                if args.full_anchor_loss_weight > 0:
                    set_guided_prototype_image(model, images)
                    output = model(images)
                    full_loss = normal_loss_from_output(output, loss_step)
                    loss = loss + args.full_anchor_loss_weight * full_loss
                return loss
            masked_loss, _ = compute_masked_normal_loss(
                en,
                de,
                mask=mask,
                architecture=args.architecture,
                step=loss_step,
                normal_loss=args.normal_loss,
                prototype_loss_weight=args.prototype_loss_weight,
                gather_loss=masked_gather_loss if args.architecture == "inpformer" else None,
                hard_quantile=args.hard_quantile,
                easy_weight=args.easy_weight,
                band_weights=mask_band_loss_weights,
                dinomaly_p_final=args.dinomaly_native_p_final,
                dinomaly_warmup_iters=args.dinomaly_native_warmup_iters,
                dinomaly_factor=args.dinomaly_native_factor,
                inpformer_y=args.inpformer_native_y,
            )
            loss = args.mask_loss_weight * masked_loss
            if args.mask_strategy == "adaptive" and args.adaptive_planner_loss_weight > 0:
                planner_loss = adaptive_mask_planner_loss(
                    model,
                    en,
                    de,
                    ratio_weight=args.adaptive_ratio_loss_weight,
                    prior_weight=args.adaptive_prior_loss_weight,
                    tv_weight=args.adaptive_tv_loss_weight,
                    binary_weight=args.adaptive_binary_loss_weight,
                    difficulty_weight=args.adaptive_difficulty_loss_weight,
                )
                loss = loss + args.adaptive_planner_loss_weight * planner_loss
            loss = add_guided_extra_loss(loss, en, de)
            if args.full_anchor_loss_weight > 0:
                set_guided_prototype_image(model, images)
                output = model(images)
                full_loss = normal_loss_from_output(output, loss_step)
                loss = loss + args.full_anchor_loss_weight * full_loss
            return loss
        set_guided_prototype_image(model, images)
        output = model(images)
        loss = normal_loss_from_output(output, loss_step, roi_mask)
        return add_guided_extra_loss(loss, output[0], output[1])

    if start_step >= args.total_iters:
        if is_main_process(rank):
            print(f"[resume] start_step={start_step} already reaches total_iters={args.total_iters}", flush=True)

    # This may evaluate continued epoch 0 or finish a validation interrupted
    # after a scheduled checkpoint. It does not consume the training iterator.
    run_inline_validation(start_step)

    for step in range(start_step + 1, args.total_iters + 1):
        lr = scheduled_learning_rate(
            step=step,
            total_iters=args.total_iters,
            epoch_iters=args.epoch_iters,
            base_lr=args.learning_rate,
            schedule=args.lr_schedule,
            decay_start_epoch=args.lr_decay_start_epoch,
            min_lr=args.min_learning_rate,
            warmup_epochs=args.lr_warmup_epochs,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        step_losses = []
        for accum_idx in range(args.grad_accum_steps):
            micro_step = (step - 1) * args.grad_accum_steps + accum_idx
            images, source_group_ids, roi_mask = next_images()
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = float(args.normal_loss_weight) * compute_loss(
                    images, micro_step, source_group_ids, roi_mask
                )
                scaled_loss = loss / args.grad_accum_steps
            scaler.scale(scaled_loss).backward()
            step_losses.append(float(loss.item()))
        scaler.unscale_(optimizer)
        sync_gradients(trainable.parameters(), world_size)
        nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
        scaler.step(optimizer)
        scaler.update()

        step_loss = float(np.mean(step_losses))
        recent_losses.append(step_loss)
        epoch_losses.append(step_loss)
        if is_main_process(rank) and (step % args.log_interval == 0 or step == 1):
            seconds = time.time() - start_time
            lr = optimizer.param_groups[0]["lr"]
            diag = get_guided_prototype_diag(model)
            context_diag = get_context_normal_diag(model)
            extra_diag = getattr(model, "_guided_extra_diag", {}) or {}
            row = {
                "iter": step,
                "loss": float(np.mean(recent_losses)),
                "lr": lr,
                "seconds": round(seconds, 2),
                "guided_prior_bg": diag.get("guided_prior_bg", ""),
                "guided_prior_texture": diag.get("guided_prior_texture", ""),
                "guided_prior_object": diag.get("guided_prior_object", ""),
                "guided_assign_bg": diag.get("guided_assign_bg", ""),
                "guided_assign_texture": diag.get("guided_assign_texture", ""),
                "guided_assign_object": diag.get("guided_assign_object", ""),
                "guided_proto_sep": diag.get("guided_proto_sep", ""),
                "guided_intra_balance": diag.get("guided_intra_balance", ""),
                "guided_intra_min_soft_usage": diag.get("guided_intra_min_soft_usage", ""),
                "guided_intra_repulsion": diag.get("guided_intra_repulsion", ""),
                "guided_intra_similarity": diag.get("guided_intra_similarity", ""),
                "guided_memory_mode_loss": diag.get("guided_memory_mode_loss", ""),
                "guided_memory_mode_accuracy": diag.get("guided_memory_mode_accuracy", ""),
                "guided_memory_mode_min_usage": diag.get("guided_memory_mode_min_usage", ""),
                **{name: diag.get(name, "") for name in CENTER6_DIAGNOSTIC_FIELDS},
                **{name: diag.get(name, "") for name in DESCRIPTOR_DIAGNOSTIC_FIELDS},
                "guided_valid_ratio": diag.get("guided_valid_ratio", ""),
                "guided_valid_bg": diag.get("guided_valid_bg", ""),
                "guided_valid_texture": diag.get("guided_valid_texture", ""),
                "guided_valid_object": diag.get("guided_valid_object", ""),
                "guided_confidence": diag.get("guided_confidence", ""),
                "guided_gate_mean": diag.get("guided_gate_mean", ""),
                "guided_gate_min": diag.get("guided_gate_min", ""),
                "guided_gate_suppressed": diag.get("guided_gate_suppressed", ""),
                "guided_familiarity_count": diag.get("guided_familiarity_count", ""),
                "guided_novelty_mean": diag.get("guided_novelty_mean", ""),
                "guided_novelty_distance": diag.get("guided_novelty_distance", ""),
                "guided_pollution_risk": diag.get("guided_pollution_risk", ""),
                "guided_novelty_threshold": diag.get("guided_novelty_threshold", ""),
                "guided_native_anchor_alpha": diag.get("guided_native_anchor_alpha", ""),
                "guided_distill_weight": diag.get("guided_distill_weight", ""),
                "guided_semantic_coverage_weight": diag.get(
                    "guided_semantic_coverage_weight", ""
                ),
                "guided_semantic_coverage_effective_weight": diag.get(
                    "guided_semantic_coverage_effective_weight", ""
                ),
                "guided_semantic_coverage_min_role_mass": diag.get(
                    "guided_semantic_coverage_min_role_mass", ""
                ),
                "guided_semantic_coverage_loss": diag.get(
                    "guided_semantic_coverage_loss", ""
                ),
                "guided_semantic_coverage_active_roles": diag.get(
                    "guided_semantic_coverage_active_roles", ""
                ),
                "guided_semantic_coverage_active_image_fraction": diag.get(
                    "guided_semantic_coverage_active_image_fraction", ""
                ),
                "guided_semantic_coverage_matched_similarity_mean": diag.get(
                    "guided_semantic_coverage_matched_similarity_mean", ""
                ),
                "guided_semantic_coverage_matched_similarity_min": diag.get(
                    "guided_semantic_coverage_matched_similarity_min", ""
                ),
                "guided_semantic_coverage_matched_slot_fraction": diag.get(
                    "guided_semantic_coverage_matched_slot_fraction", ""
                ),
                "guided_semantic_coverage_matched_slot_entropy": diag.get(
                    "guided_semantic_coverage_matched_slot_entropy", ""
                ),
                **{
                    f"guided_semantic_coverage_role_mass_{index}": diag.get(
                        f"guided_semantic_coverage_role_mass_{index}", ""
                    )
                    for index in range(3)
                },
                **{
                    f"guided_semantic_coverage_role_active_{index}": diag.get(
                        f"guided_semantic_coverage_role_active_{index}", ""
                    )
                    for index in range(3)
                },
                "guided_aggregation_alpha": diag.get("guided_aggregation_alpha", ""),
                "guided_mode_routing_active": diag.get(
                    "guided_mode_routing_active", ""
                ),
                "guided_mode_routing_strength": diag.get(
                    "guided_mode_routing_strength", ""
                ),
                "guided_mode_routing_floor": diag.get(
                    "guided_mode_routing_floor", ""
                ),
                "guided_mode_routing_factor_mean": diag.get(
                    "guided_mode_routing_factor_mean", ""
                ),
                "guided_mode_routing_factor_min": diag.get(
                    "guided_mode_routing_factor_min", ""
                ),
                "guided_mode_routing_factor_max": diag.get(
                    "guided_mode_routing_factor_max", ""
                ),
                "guided_mode_routing_factor_std": diag.get(
                    "guided_mode_routing_factor_std", ""
                ),
                "guided_proto_loss": diag.get("guided_proto_loss", ""),
                "guided_native_gather": diag.get("guided_native_gather", ""),
                "guided_group_gather": diag.get("guided_group_gather", ""),
                "roi_masked_gather_loss": getattr(
                    model, "_normal_loss_diag", {}
                ).get("roi_masked_gather_loss", ""),
                "roi_aware_guided_gather": getattr(
                    model, "_normal_loss_diag", {}
                ).get("roi_aware_guided_gather", ""),
                "guided_calibration_groups": diag.get("guided_calibration_groups", ""),
                "guided_calibrated_count": diag.get("guided_calibrated_count", ""),
                "guided_prior_anchor": extra_diag.get("guided_prior_anchor", ""),
                "guided_normal_guard": extra_diag.get("guided_normal_guard", ""),
                "guided_guard_rec": extra_diag.get("guided_guard_rec", ""),
                "guided_guard_obj": extra_diag.get("guided_guard_obj", ""),
                "guided_transport_gate": diag.get("guided_transport_gate", ""),
                "guided_transport_confidence": diag.get("guided_transport_confidence", ""),
                "guided_transport_surprise": diag.get("guided_transport_surprise", ""),
                "guided_transport_objectness": diag.get("guided_transport_objectness", ""),
                "guided_transport_source_diversity": diag.get(
                    "guided_transport_source_diversity", ""
                ),
                "guided_transport_shift": diag.get("guided_transport_shift", ""),
                "guided_transport_scale_entropy": diag.get("guided_transport_scale_entropy", ""),
                "guided_transport_fallback_ratio": diag.get("guided_transport_fallback_ratio", ""),
                "guided_transport_memory_count": diag.get("guided_transport_memory_count", ""),
                "inp_coherence_loss": args.inp_coherence_loss if args.architecture == "inpformer" else "",
                "inp_soft_mining_cosine": getattr(model, "_normal_loss_diag", {}).get("inp_soft_mining_cosine", ""),
                "inp_soft_mining_mse": getattr(model, "_normal_loss_diag", {}).get("inp_soft_mining_mse", ""),
                "context_normal_mix": context_diag.get("context_normal_mix", ""),
                "context_normal_memory_count": context_diag.get("context_normal_memory_count", ""),
                "context_normal_similarity": context_diag.get("context_normal_similarity", ""),
                "context_normal_shift": context_diag.get("context_normal_shift", ""),
                "context_normal_unexcluded_ratio": context_diag.get("context_normal_unexcluded_ratio", ""),
            }
            with log_path.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writerow(row)
            print(f"[train] iter={step}/{args.total_iters} loss={row['loss']:.6f} lr={lr:.3e}", flush=True)
            recent_losses = []

        if args.epoch_iters > 0 and step % args.epoch_iters == 0:
            epoch = step // args.epoch_iters
            epoch_loss = float(np.mean(epoch_losses))
            # Boundary checkpoints must contain no losses from the epoch that
            # just completed. Otherwise a later resume averages two epochs.
            epoch_losses = []
            if is_main_process(rank):
                improved = False
                if epoch_loss < best_epoch_loss:
                    best_epoch_loss = epoch_loss
                    best_path = best_loss_checkpoint_path
                    best_metrics = {
                        "checkpoint": None if args.final_only_checkpoints else str(best_path),
                        "iter": step,
                        "epoch": epoch,
                        "epoch_loss": epoch_loss,
                        "best_loss": best_epoch_loss,
                    }
                    (args.output_dir / "best_metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
                    if args.final_only_checkpoints:
                        print(
                            f"[best-loss] epoch={epoch} iter={step} "
                            f"loss={epoch_loss:.6f} checkpoint_skipped",
                            flush=True,
                        )
                    else:
                        save_checkpoint(
                            best_path,
                            step,
                            include_training_state=True,
                            checkpoint_kind="best",
                            epoch=epoch,
                            epoch_loss=epoch_loss,
                            best_loss=best_epoch_loss,
                        )
                        print(f"[best] epoch={epoch} iter={step} loss={epoch_loss:.6f} saved {best_path}", flush=True)
                    improved = True
                else:
                    print(
                        f"[epoch] epoch={epoch} iter={step} loss={epoch_loss:.6f} best={best_epoch_loss:.6f}",
                        flush=True,
                    )
                latest_path = args.output_dir / "latest.pth"
                if improved and not args.final_only_checkpoints:
                    replace_checkpoint_with_hardlink(best_path, latest_path)
                    print(f"[latest] linked {latest_path} -> {best_path.name}", flush=True)
                elif (
                    args.latest_epoch_interval > 0
                    and epoch % args.latest_epoch_interval == 0
                ):
                    save_checkpoint(
                        latest_path,
                        step,
                        include_training_state=True,
                        checkpoint_kind="latest",
                        epoch=epoch,
                        best_loss=best_epoch_loss,
                    )
                    print(f"[latest] {latest_path}", flush=True)
                save_analysis_checkpoint(
                    epoch=epoch,
                    step=step,
                    epoch_loss=epoch_loss,
                    learning_rate=lr,
                )

        if is_main_process(rank) and args.save_interval > 0 and step % args.save_interval == 0:
            ckpt_path = args.output_dir / f"iter_{step:06d}.pth"
            save_checkpoint(ckpt_path, step, include_training_state=True, checkpoint_kind="interval", best_loss=best_epoch_loss)
            print(f"[save] {ckpt_path}", flush=True)
        if is_main_process(rank) and args.latest_interval > 0 and step % args.latest_interval == 0:
            latest_path = args.output_dir / "latest.pth"
            save_checkpoint(latest_path, step, include_training_state=True, checkpoint_kind="latest", best_loss=best_epoch_loss)
            print(f"[latest] {latest_path}", flush=True)

        run_inline_validation(step)

    if world_size > 1:
        dist.barrier()
    if is_main_process(rank):
        if args.blindspot_context and args.blindspot_calibration_batches > 0:
            model.eval()
            adaptive_values = []
            context_values = []
            with torch.no_grad():
                for _ in range(args.blindspot_calibration_batches):
                    images, source_group_ids, _ = next_images()
                    set_guided_prototype_source_groups(model, source_group_ids)
                    set_guided_prototype_image(model, images)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        en, de, context = inpformer_forward_with_prototype_context(
                            model, images
                        )
                        adaptive_score = reconstruction_components(en, de)["base"]
                        context_score = blindspot_context_score(
                            model.blindspot_context_head,
                            context["target_tokens"],
                            context["agg_prototype"],
                            side=context["side"],
                        )
                    adaptive_values.append(
                        adaptive_score.detach().float().cpu().flatten()
                    )
                    context_values.append(
                        context_score.detach().float().cpu().flatten()
                    )

            def score_summary(values: list[torch.Tensor]) -> dict[str, float]:
                merged = torch.cat(values)
                quantiles = torch.quantile(
                    merged, torch.tensor([0.95, 0.99, 0.995])
                )
                return {
                    "count": int(merged.numel()),
                    "mean": float(merged.mean().item()),
                    "q95": float(quantiles[0].item()),
                    "q99": float(quantiles[1].item()),
                    "q995": float(quantiles[2].item()),
                }

            calibration = {
                "protocol": "normal_only_blindspot_v1",
                "batches": int(args.blindspot_calibration_batches),
                "encoder_calls": int(args.blindspot_calibration_batches),
                "adaptive": score_summary(adaptive_values),
                "context": score_summary(context_values),
            }
            calibration_path = (
                args.output_dir / "blindspot_normal_calibration.json"
            )
            with calibration_path.open("w", encoding="utf-8") as handle:
                json.dump(calibration, handle, indent=2)
            print(f"[blindspot_calibration] {calibration_path}", flush=True)
        final_path = args.output_dir / "model.pth"
        save_checkpoint(
            final_path,
            args.total_iters,
            include_training_state=not args.final_only_checkpoints,
            checkpoint_kind="final",
            best_loss=best_epoch_loss,
        )
        replace_checkpoint_with_hardlink(final_path, args.output_dir / "latest.pth")
        print(f"[done] saved {final_path}", flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
