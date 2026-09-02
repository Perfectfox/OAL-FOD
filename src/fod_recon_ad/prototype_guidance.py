from __future__ import annotations

import itertools
import math
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .familiarity_memory_rebuild import boundary_soft_semantic_weights


@dataclass(frozen=True)
class GuidedPrototypeConfig:
    groups: tuple[int, int, int] = (2, 2, 2)
    prior_floor: float = 0.03
    texture_power: float = 1.0
    object_power: float = 1.0
    feature_texture_weight: float = 0.5
    image_texture_weight: float = 0.5
    feature_object_weight: float = 0.75
    image_object_weight: float = 0.25
    object_texture_suppress: float = 0.25
    vlm_prior_checkpoint: str = ""
    vlm_prior_physical_weight: float = 0.0
    vlm_prior_fusion_mode: str = "global"
    vlm_prior_correction_max_weight: float = 0.5
    vlm_prior_correction_physical_margin: float = 0.15
    vlm_prior_correction_vlm_margin: float = 0.15
    separation_weight: float = 0.02
    separation_margin: float = 0.30
    trainable_prior: bool = False
    prior_hidden_dim: int = 0
    prior_anchor_weight: float = 0.0
    normal_guard_weight: float = 0.0
    normal_guard_beta: float = 0.5
    normal_guard_top_frac: float = 0.15
    scale_consistent: bool = False
    prior_reference_side: int = 32
    image_reference_size: int = 448
    group_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    multiscale_direct: bool = False
    spatial_scale_mode: str = "legacy"
    spatial_reference_side: int = 32
    object_kernels: tuple[int, ...] = (3, 5, 7)
    object_kernel_weights: tuple[float, ...] = (0.35, 0.25, 0.40)
    texture_offsets: tuple[int, ...] = (1, 2)
    texture_offset_weights: tuple[float, ...] = (0.60, 0.40)
    robust_quantile_low: float = 0.50
    robust_quantile_high: float = 0.99
    robust_quantile_momentum: float = 0.95
    group_contrastive: bool = False
    group_temperature: float = 0.10
    group_confidence_margin: float = 0.15
    intra_group_balance_weight: float = 0.0
    intra_group_temperature: float = 0.10
    intra_group_repulsion_weight: float = 0.0
    intra_group_repulsion_margin: float = 0.20
    memory_mode_weight: float = 0.0
    memory_mode_temperature: float = 0.10
    memory_mode_teacher_temperature: float = 0.10
    memory_mode_margin_weighting: bool = False
    memory_mode_soft_targets: bool = False
    memory_mode_soft_semantic: bool = False
    memory_mode_semantic_margin: float = 0.15
    center6_balanced: bool = False
    center6_reduction: str = "equal_mode"
    center6_temperature: float = 0.10
    center6_teacher_temperature: float = 0.10
    center6_prior_mapping: str = "soft"
    center6_loss_mapping: str = "soft"
    center6_radius_quantile: float = 0.95
    center6_radius_floor: float = 1e-3
    center6_hierarchical_teacher: bool = False
    center6_novelty_veto: bool = False
    center6_novelty_threshold: float = 1.0
    center6_novelty_temperature: float = 0.10
    center6_novelty_min_weight: float = 0.05
    center6_hierarchical_reliability: bool = False
    center6_collapsed_slot_diversity_weight: float = 0.0
    center6_collapsed_slot_diversity_margin: float = 0.90
    aggregation_gate: bool = False
    aggregation_min_weight: float = 0.05
    aggregation_power: float = 1.0
    aggregation_alpha: float | None = None
    mode_specific_routing: bool = False
    mode_routing_floor: float = 0.05
    mode_routing_strength: float = 1.0
    aggregation_familiarity_gate: bool = False
    aggregation_risk_composition: str = "product"
    familiarity_write_enabled: bool = True
    target_gate_calibration: bool = False
    target_gate_risk_source: str = "familiarity"
    target_reject_gate: bool = False
    target_mode_gate: bool = False
    target_mode_gate_min_support: int = 8
    target_mode_novelty_temperature: float = 0.10
    decoder_read_gate: bool = False
    decoder_read_strength: float = 0.5
    decoder_read_scope: str = "object"
    decoder_read_mode: str = "suppress"
    decoder_read_tail_threshold: float = 0.0
    decoder_read_tail_upper: float = 1.0
    decoder_read_tail_power: float = 1.0
    decoder_read_attention_aware: bool = False
    decoder_read_responsibility_aware: bool = False
    decoder_read_update_cap: float = 0.0
    decoder_read_layer_start: int = 0
    decoder_read_layer_strengths: tuple[float, ...] = ()
    decoder_read_adaptive_strength_power: float = 0.0
    decoder_read_risk_source: str = "aggregation"
    decoder_read_prototypewise: bool = False
    decoder_read_center6_support_alpha: float = 0.5
    decoder_read_center6_support_temperature: float = 0.10
    decoder_read_prototype_responsibility_floor: float = 0.5
    decoder_read_center6_mode_tail_thresholds: tuple[float, ...] = ()
    decoder_read_center6_mode_tail_uppers: tuple[float, ...] = ()
    familiarity_memory_size: int = 1024
    familiarity_tokens_per_image: int = 8
    familiarity_min_count: int = 256
    familiarity_novelty_floor: float = 0.08
    familiarity_temperature: float = 0.03
    familiarity_novelty_mapping: str = "sigmoid"
    familiarity_calibration_mode: str = "legacy"
    familiarity_calibration_quantile: float = 0.95
    native_anchor_alpha: float = 1.0
    guided_distill_weight: float = 0.0
    semantic_coverage_weight: float = 0.0
    semantic_coverage_min_role_mass: float = 0.05
    semantic_coverage_variant: str = "set_match"
    semantic_coverage_roles: tuple[int, ...] = (0, 1, 2)
    semantic_coverage_margin: float = 0.80
    semantic_coverage_min_confidence: float = 0.0
    semantic_coverage_max_risk: float = 1.0
    semantic_coverage_warmup_steps: int = 0
    semantic_coverage_ramp_steps: int = 0
    context_variant: str = "off"
    context_transport: bool = False
    context_adaptive_scale: bool = False
    free_prototypes: bool = False
    context_memory_size: int = 2048
    context_candidates_per_image: int = 32
    context_candidates_per_group: int = 128
    context_topk: int = 5
    context_temperature: float = 0.07
    context_query_chunk_size: int = 1024
    context_memory_build_batches: int = 0
    context_key_dim: int = 64
    descriptor_variant: str = "off"
    descriptor_radii: tuple[int, ...] = (1, 2, 4)
    descriptor_memory_size: int = 176
    descriptor_candidates_per_image: int = 16
    descriptor_candidates_per_group: int = 128
    descriptor_topk: int = 3
    descriptor_temperature: float = 0.07
    descriptor_query_chunk_size: int = 1024
    descriptor_key_dim: int = 64
    descriptor_memory_build_batches: int = 0
    descriptor_source_manifest: str = ""
    descriptor_read_strength: float = 0.5
    descriptor_read_tail_threshold: float = 0.95
    descriptor_read_tail_upper: float = 1.0
    descriptor_read_tail_power: float = 1.0
    descriptor_read_layer_start: int = 0
    descriptor_read_update_cap: float = 0.25
    descriptor_objectness_floor: float = 0.5


def add_guided_prototype_args(parser) -> None:
    parser.add_argument(
        "--rgpl-historical-compatible-raw",
        action="store_true",
        help=(
            "Use the audited single-path RGPL Raw calculation that preserves the "
            "2026-07-29 memory-176 risk gate and epsilon ROI aggregation."
        ),
    )
    parser.add_argument(
        "--prototype-hard-roi-attention",
        action="store_true",
        help=(
            "Restrict INP-Former prototype aggregation softmax to tokens fully "
            "inside the crop ROI. This is independent of the RGPL risk gate."
        ),
    )
    parser.add_argument(
        "--prototype-transition-roi-attention",
        action="store_true",
        help=(
            "Use token-level ROI coverage as a continuous prototype aggregation "
            "prior. Fully valid tokens receive weight one, fully invalid tokens "
            "receive the configured floor, and boundary tokens interpolate linearly."
        ),
    )
    parser.add_argument(
        "--prototype-transition-roi-floor",
        type=float,
        default=1e-6,
        help="Minimum aggregation prior outside the ROI in transition mode.",
    )
    parser.add_argument(
        "--guided-prototype",
        action="store_true",
        help="For INP-Former, guide prototype gather loss with bg/texture/objectness priors.",
    )
    parser.add_argument(
        "--guided-prototype-roi-aware-loss",
        action="store_true",
        help=(
            "Optimize the configured guided/native prototype objective on the "
            "same strict ROI tokens used by hard-ROI aggregation."
        ),
    )
    parser.add_argument("--guided-prototype-groups", default="2,2,2", help="Prototype counts for bg,texture,object groups.")
    parser.add_argument("--guided-prototype-prior-floor", type=float, default=0.03)
    parser.add_argument("--guided-prototype-texture-power", type=float, default=1.0)
    parser.add_argument("--guided-prototype-object-power", type=float, default=1.0)
    parser.add_argument("--guided-prototype-feature-texture-weight", type=float, default=0.5)
    parser.add_argument("--guided-prototype-image-texture-weight", type=float, default=0.5)
    parser.add_argument("--guided-prototype-feature-object-weight", type=float, default=0.75)
    parser.add_argument("--guided-prototype-image-object-weight", type=float, default=0.25)
    parser.add_argument("--guided-prototype-object-texture-suppress", type=float, default=0.25)
    parser.add_argument(
        "--guided-prototype-vlm-prior-checkpoint",
        type=Path,
        default=None,
        help="Frozen three-way semantic-prior head distilled from VLM Normal labels.",
    )
    parser.add_argument(
        "--guided-prototype-vlm-prior-physical-weight",
        type=float,
        default=0.0,
        help="Weak physical-prior log-pooling weight; 0 uses the VLM prior alone.",
    )
    parser.add_argument(
        "--guided-prototype-vlm-prior-fusion-mode",
        choices=("global", "low_confidence_correction"),
        default="global",
        help=(
            "global uses the historical fixed-weight pooling; low_confidence_correction "
            "keeps the physical prior and lets a confident VLM prediction adjust only "
            "tokens with an ambiguous physical-prior top1/top2 margin."
        ),
    )
    parser.add_argument(
        "--guided-prototype-vlm-prior-correction-max-weight", type=float, default=0.5
    )
    parser.add_argument(
        "--guided-prototype-vlm-prior-correction-physical-margin", type=float, default=0.15
    )
    parser.add_argument(
        "--guided-prototype-vlm-prior-correction-vlm-margin", type=float, default=0.15
    )
    parser.add_argument("--guided-prototype-separation-weight", type=float, default=0.02)
    parser.add_argument("--guided-prototype-separation-margin", type=float, default=0.30)
    parser.add_argument("--guided-prototype-trainable-prior", action="store_true")
    parser.add_argument("--guided-prototype-prior-hidden-dim", type=int, default=0)
    parser.add_argument("--guided-prototype-prior-anchor-weight", type=float, default=0.0)
    parser.add_argument("--guided-prototype-normal-guard-weight", type=float, default=0.0)
    parser.add_argument("--guided-prototype-normal-guard-beta", type=float, default=0.5)
    parser.add_argument("--guided-prototype-normal-guard-top-frac", type=float, default=0.15)
    parser.add_argument(
        "--guided-prototype-scale-consistent",
        action="store_true",
        help="Compute fixed guidance on a canonical physical scale before mapping it to the current token grid.",
    )
    parser.add_argument(
        "--guided-prototype-prior-reference-side",
        type=int,
        default=32,
        help="Canonical token-grid side used to compute scale-consistent fixed priors; 0 uses the current grid.",
    )
    parser.add_argument(
        "--guided-prototype-image-reference-size",
        type=int,
        default=448,
        help="Canonical image size used by the edge/texture prior; 0 uses the current input size.",
    )
    parser.add_argument(
        "--guided-prototype-group-weights",
        default="1,1,1",
        help="Multiplicative calibration weights for bg, texture and object fixed-prior channels.",
    )
    parser.add_argument(
        "--guided-prototype-multiscale-direct",
        action="store_true",
        help="Build a robust multi-scale fixed prior directly on the current token grid.",
    )
    parser.add_argument(
        "--guided-prototype-spatial-scale-mode",
        choices=("legacy", "integer"),
        default="legacy",
        help=(
            "Spatial-parameter policy for multiscale priors. 'legacy' uses the configured token "
            "windows verbatim; 'integer' scales kernel radii and offsets from a reference token grid."
        ),
    )
    parser.add_argument(
        "--guided-prototype-spatial-reference-side",
        type=int,
        default=32,
        help="Reference token-grid side for integer spatial scaling (32 corresponds to a 448 ViT/14 input).",
    )
    parser.add_argument("--guided-prototype-object-kernels", default="3,5,7")
    parser.add_argument("--guided-prototype-object-kernel-weights", default="0.35,0.25,0.40")
    parser.add_argument("--guided-prototype-texture-offsets", default="1,2")
    parser.add_argument("--guided-prototype-texture-offset-weights", default="0.60,0.40")
    parser.add_argument("--guided-prototype-robust-quantile-low", type=float, default=0.50)
    parser.add_argument("--guided-prototype-robust-quantile-high", type=float, default=0.99)
    parser.add_argument("--guided-prototype-robust-quantile-momentum", type=float, default=0.95)
    parser.add_argument(
        "--guided-prototype-group-contrastive",
        action="store_true",
        help="Replace soft all-group gather with confidence-filtered group contrastive learning.",
    )
    parser.add_argument("--guided-prototype-group-temperature", type=float, default=0.10)
    parser.add_argument("--guided-prototype-group-confidence-margin", type=float, default=0.15)
    parser.add_argument(
        "--guided-prototype-intra-group-balance-weight",
        type=float,
        default=0.0,
        help="KL weight that balances soft token assignments among prototypes inside each semantic group.",
    )
    parser.add_argument("--guided-prototype-intra-group-temperature", type=float, default=0.10)
    parser.add_argument(
        "--guided-prototype-intra-group-repulsion-weight",
        type=float,
        default=0.0,
        help="Weight for repelling image-conditioned prototypes inside the same semantic group.",
    )
    parser.add_argument("--guided-prototype-intra-group-repulsion-margin", type=float, default=0.20)
    parser.add_argument(
        "--guided-prototype-memory-mode-teacher",
        type=Path,
        default=None,
        help=(
            "P2: frozen P1 memory .npz containing bank and semantic_group_ids. "
            "It supervises distinct normal modes inside each prototype group."
        ),
    )
    parser.add_argument(
        "--guided-prototype-memory-mode-weight",
        type=float,
        default=0.0,
        help="P2 loss weight for frozen-memory intra-group mode supervision.",
    )
    parser.add_argument("--guided-prototype-memory-mode-temperature", type=float, default=0.10)
    parser.add_argument(
        "--guided-prototype-memory-mode-teacher-temperature",
        type=float,
        default=0.10,
        help="Temperature for optional stable-teacher soft mode targets.",
    )
    parser.add_argument(
        "--guided-prototype-memory-mode-margin-weighting",
        action="store_true",
        help="Downweight ambiguous mode labels using Normal-only teacher margin calibration.",
    )
    parser.add_argument(
        "--guided-prototype-memory-mode-soft-targets",
        action="store_true",
        help="Use frozen-teacher soft mode probabilities instead of hard argmax labels.",
    )
    parser.add_argument(
        "--guided-prototype-memory-mode-soft-semantic",
        action="store_true",
        help="Share ambiguous top-2 semantic-prior tokens in P2 mode supervision.",
    )
    parser.add_argument(
        "--guided-prototype-memory-mode-semantic-margin", type=float, default=0.15
    )
    parser.add_argument(
        "--guided-prototype-center6-balanced",
        action="store_true",
        help=(
            "Replace the fixed-prior group and within-group mode objectives with one "
            "mode-balanced six-center KL objective over every token."
        ),
    )
    parser.add_argument(
        "--guided-prototype-center6-teacher",
        type=Path,
        default=None,
        help=(
            "Frozen P1 memory sidecar used to construct six teacher centers and their "
            "within-mode radius calibration."
        ),
    )
    parser.add_argument(
        "--guided-prototype-center6-reduction",
        choices=("equal_mode", "token_mean", "sqrt_balanced"),
        default="equal_mode",
        help=(
            "Reduce per-token Center6 KL by equal observed modes, natural token "
            "frequency, or square-root mode occupancy."
        ),
    )
    parser.add_argument("--guided-prototype-center6-temperature", type=float, default=0.10)
    parser.add_argument(
        "--guided-prototype-center6-teacher-temperature", type=float, default=0.10
    )
    parser.add_argument(
        "--guided-prototype-center6-prior-mapping",
        choices=("soft", "hard"),
        default="soft",
        help=(
            "Use radius-calibrated soft Center6 group probabilities or a "
            "nearest-mode hard group prior."
        ),
    )
    parser.add_argument(
        "--guided-prototype-center6-loss-mapping",
        choices=("soft", "hard"),
        default="soft",
        help=(
            "Use soft teacher KL or nearest-mode hard cross-entropy for the "
            "Center6 prototype objective."
        ),
    )
    parser.add_argument(
        "--guided-prototype-center6-radius-quantile", type=float, default=0.95
    )
    parser.add_argument(
        "--guided-prototype-center6-radius-floor", type=float, default=1e-3
    )
    parser.add_argument(
        "--guided-prototype-center6-hierarchical-teacher",
        action="store_true",
        help=(
            "Build the Center6 teacher as group probability times conditional "
            "within-group mode probability, with mode-count-corrected group energies."
        ),
    )
    parser.add_argument(
        "--guided-prototype-center6-novelty-veto",
        action="store_true",
        help=(
            "Softly downweight prototype guidance for tokens outside every "
            "radius-normalized Normal mode."
        ),
    )
    parser.add_argument(
        "--guided-prototype-center6-novelty-threshold", type=float, default=1.0
    )
    parser.add_argument(
        "--guided-prototype-center6-novelty-temperature", type=float, default=0.10
    )
    parser.add_argument(
        "--guided-prototype-center6-novelty-min-weight", type=float, default=0.05
    )
    parser.add_argument(
        "--guided-prototype-center6-hierarchical-reliability",
        action="store_true",
        help=(
            "Decompose Center6 KL into group and conditional child-mode terms, "
            "then weight only the child term by Normal-only group reliability."
        ),
    )
    parser.add_argument(
        "--guided-prototype-center6-collapsed-slot-diversity-weight",
        type=float,
        default=0.0,
        help=(
            "Penalize redundant image-conditioned slots that share one adaptive "
            "parent mode; zero preserves the v1 objective."
        ),
    )
    parser.add_argument(
        "--guided-prototype-center6-collapsed-slot-diversity-margin",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--guided-prototype-aggregation-gate",
        action="store_true",
        help="Downweight high-objectness tokens only when they update image-conditioned prototypes.",
    )
    parser.add_argument("--guided-prototype-aggregation-min-weight", type=float, default=0.05)
    parser.add_argument("--guided-prototype-aggregation-power", type=float, default=1.0)
    parser.add_argument(
        "--guided-prototype-aggregation-alpha",
        type=float,
        default=None,
        help=(
            "Optional token-gate blend in [0,1]. By default it follows native-anchor-alpha; "
            "set it explicitly to isolate aggregation gating from the prototype gather loss."
        ),
    )
    parser.add_argument(
        "--guided-prototype-mode-routing",
        action="store_true",
        help=(
            "Route each physical prototype slot toward tokens supported by its "
            "assigned frozen Center6 Normal mode. This changes aggregation only; "
            "it does not change the prototype gather/coherence objective."
        ),
    )
    parser.add_argument(
        "--guided-prototype-mode-routing-floor",
        type=float,
        default=0.05,
        help=(
            "Minimum soft compatibility for a token outside a prototype slot's "
            "assigned Center6 mode."
        ),
    )
    parser.add_argument(
        "--guided-prototype-mode-routing-strength",
        type=float,
        default=1.0,
        help="Blend in [0,1] between scalar aggregation gating and full mode routing.",
    )
    parser.add_argument(
        "--guided-prototype-familiarity-gate",
        action="store_true",
        help="Gate prototype updates by objectness times novelty to a frozen normal object-token memory.",
    )
    parser.add_argument(
        "--guided-prototype-aggregation-risk-composition",
        choices=("product", "objectness_only", "novelty_only"),
        default="product",
        help=(
            "Controlled RGPL ablation for the scalar aggregation risk: the "
            "reported model uses objectness times novelty; the other choices "
            "retain only one factor."
        ),
    )
    parser.add_argument(
        "--guided-prototype-familiarity-read-only",
        action="store_true",
        help="Compute familiarity risk but disable its prototype-aggregation write gate.",
    )
    parser.add_argument(
        "--guided-prototype-target-gate-calibration",
        action="store_true",
        help=(
            "Apply a two-parameter monotonic calibration to familiarity write risk. "
            "The calibrator is initialized as the identity and is intended for "
            "TargetOnNormal token-mask supervision."
        ),
    )
    parser.add_argument(
        "--guided-prototype-target-gate-risk-source",
        choices=("familiarity", "mode_normalized"),
        default="familiarity",
        help=(
            "Risk presented to the two-parameter TargetGate. mode_normalized uses "
            "nearest Center6 cosine distance divided by that mode's Normal radius."
        ),
    )
    parser.add_argument(
        "--guided-prototype-target-reject-gate",
        action="store_true",
        help=(
            "Replace the scalar TargetGate with an interpretable monotonic gate over "
            "familiarity risk, mode-normalized novelty, objectness, and mode uncertainty."
        ),
    )
    parser.add_argument(
        "--guided-prototype-target-mode-gate",
        action="store_true",
        help=(
            "Use a monotonic per-effective-mode reject boundary over nearest "
            "Center6 distance/radius. Sparse modes fall back to a learned global "
            "boundary; this gate controls prototype aggregation writes only."
        ),
    )
    parser.add_argument(
        "--guided-prototype-target-mode-gate-min-support",
        type=int,
        default=8,
        help="Minimum cumulative pasted-target tokens before enabling a mode-specific boundary.",
    )
    parser.add_argument(
        "--guided-prototype-target-mode-novelty-temperature",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--guided-prototype-decoder-read-gate",
        action="store_true",
        help="Use familiarity risk to suppress prototype reads in the decoder.",
    )
    parser.add_argument("--guided-prototype-decoder-read-strength", type=float, default=0.5)
    parser.add_argument(
        "--guided-prototype-decoder-read-scope",
        choices=("object", "all"),
        default="object",
    )
    parser.add_argument(
        "--guided-prototype-decoder-read-mode",
        choices=("suppress", "tail_suppress", "selective_route"),
        default="suppress",
        help=(
            "Read-gate behavior. 'suppress' preserves the legacy raw-risk attenuation; "
            "'tail_suppress' applies normal-tail-calibrated attenuation without moving "
            "attention mass; 'selective_route' moves removed object-prototype attention "
            "mass to the background/texture prototypes."
        ),
    )
    parser.add_argument(
        "--guided-prototype-decoder-read-tail-threshold",
        type=float,
        default=0.0,
        help=(
            "Risk below this normal-calibrated tail threshold is left unchanged in "
            "selective-route mode (for example, the normal-token q99 risk)."
        ),
    )
    parser.add_argument(
        "--guided-prototype-decoder-read-tail-upper",
        type=float,
        default=1.0,
        help="Risk mapped to full routing strength (must exceed the tail threshold).",
    )
    parser.add_argument(
        "--guided-prototype-decoder-read-tail-power",
        type=float,
        default=1.0,
        help="Positive power applied after normal-tail risk calibration.",
    )
    parser.add_argument(
        "--guided-prototype-decoder-read-attention-aware",
        action="store_true",
        help=(
            "Scale tail activation by the pre-gate object-prototype attention share. "
            "The share is normalized by the uniform group-size prior, so no new normal "
            "or anomaly calibration is introduced."
        ),
    )
    parser.add_argument(
        "--guided-prototype-decoder-read-responsibility-aware",
        action="store_true",
        help=(
            "For a scalar object read risk, apportion suppression between the object "
            "prototype slots using each head's pre-gate object-attention responsibility."
        ),
    )
    parser.add_argument(
        "--guided-prototype-decoder-read-update-cap",
        type=float,
        default=0.0,
        help=(
            "If positive, cap the norm removed from each decoder attention update to "
            "this fraction of the ungated update norm; 0 disables the cap."
        ),
    )
    parser.add_argument(
        "--guided-prototype-decoder-read-layer-start",
        type=int,
        default=0,
        help=(
            "Zero-based first decoder block that uses the familiarity read gate. "
            "The default 0 gates every block; 4 selects the late four blocks in "
            "the standard eight-block INP-Former decoder."
        ),
    )
    parser.add_argument(
        "--guided-prototype-decoder-read-layer-strengths",
        default="",
        help=(
            "Optional comma-separated strength for every decoder block. An empty value "
            "uses the single decoder-read-strength for all enabled blocks."
        ),
    )
    parser.add_argument(
        "--guided-prototype-decoder-read-adaptive-strength-power",
        type=float,
        default=0.0,
        help=(
            "If positive, interpolate each layer's base strength toward one as "
            "base + (1-base) * calibrated_risk**power. Zero disables adaptation."
        ),
    )
    parser.add_argument(
        "--guided-prototype-decoder-read-risk-source",
        choices=(
            "aggregation",
            "physical",
            "center6_hybrid",
            "center6_radius",
            "center6_global",
            "center6_mode_novelty",
            "target_gate",
        ),
        default="aggregation",
        help=(
            "Objectness source used only by the decoder read gate. 'aggregation' "
            "preserves the historical behavior, 'physical' uses the fixed spatial "
            "prior, and 'center6_hybrid' combines the physical prior with absolute "
            "Center6 radius support. 'center6_radius' uses only familiarity novelty "
            "and the six absolute Center6 radius violations. 'center6_global' "
            "uses novelty and only the nearest Center6 normalized distance. "
            "'center6_mode_novelty' calibrates novelty inside the nearest Center6 mode."
            " 'target_gate' reuses the learned monotonic TargetGate write risk."
        ),
    )
    parser.add_argument(
        "--guided-prototype-decoder-read-prototypewise",
        action="store_true",
        help=(
            "For Center6 reads, construct one risk per prototype. Hybrid risk uses the "
            "two object modes; Center6-radius risk uses all six modes."
        ),
    )
    parser.add_argument(
        "--guided-prototype-decoder-read-center6-support-alpha",
        type=float,
        default=0.5,
        help="Floor on the legacy physical-novelty risk before Center6 radius support.",
    )
    parser.add_argument(
        "--guided-prototype-decoder-read-center6-support-temperature",
        type=float,
        default=0.10,
        help="Temperature for mapping Center6 distance/radius around one to support violation.",
    )
    parser.add_argument(
        "--guided-prototype-decoder-read-prototype-responsibility-floor",
        type=float,
        default=0.5,
        help="Responsibility floor for prototype-wise Center6 object-mode reads.",
    )
    parser.add_argument(
        "--guided-prototype-decoder-read-center6-mode-tail-thresholds",
        default="",
        help="Comma-separated q-tail novelty thresholds for the six nearest Center6 modes.",
    )
    parser.add_argument(
        "--guided-prototype-decoder-read-center6-mode-tail-uppers",
        default="",
        help="Comma-separated upper-tail novelty thresholds for the six Center6 modes.",
    )
    parser.add_argument("--guided-prototype-familiarity-memory-size", type=int, default=1024)
    parser.add_argument("--guided-prototype-familiarity-tokens-per-image", type=int, default=8)
    parser.add_argument("--guided-prototype-familiarity-min-count", type=int, default=256)
    parser.add_argument("--guided-prototype-familiarity-novelty-floor", type=float, default=0.08)
    parser.add_argument("--guided-prototype-familiarity-temperature", type=float, default=0.03)
    parser.add_argument(
        "--guided-prototype-familiarity-novelty-mapping",
        choices=("sigmoid", "hard"),
        default="sigmoid",
        help=(
            "Map calibrated nearest-memory distance to novelty. sigmoid preserves "
            "the original smooth mapping; hard uses only the calibrated threshold."
        ),
    )
    parser.add_argument(
        "--guided-prototype-familiarity-calibration",
        choices=("legacy", "cross_group"),
        default="legacy",
        help=(
            "Novelty-threshold calibration. 'cross_group' excludes every token from the same "
            "source image/group instead of only excluding the query token itself."
        ),
    )
    parser.add_argument("--guided-prototype-familiarity-calibration-quantile", type=float, default=0.95)
    parser.add_argument(
        "--guided-prototype-native-anchor-alpha",
        type=float,
        default=1.0,
        help=(
            "Non-ROI guided-gather blend in [0,1]. Under ROI-mask training, the "
            "optimized prototype loss is always native coherence. This value still "
            "controls aggregation gating unless aggregation-alpha is explicit."
        ),
    )
    parser.add_argument(
        "--guided-prototype-distill-weight",
        type=float,
        default=0.0,
        help=(
            "Add guided Center6 KL as an auxiliary loss without reducing native "
            "coherence: L_proto = L_native + weight * L_guided. This requires "
            "native-anchor-alpha=0 and an explicit aggregation-alpha."
        ),
    )
    parser.add_argument(
        "--guided-prototype-semantic-coverage-weight",
        type=float,
        default=0.0,
        help=(
            "Add permutation-invariant, image-conditioned semantic coverage to "
            "native coherence. Supported normal roles are matched injectively to "
            "dynamic prototypes without fixed slot identities."
        ),
    )
    parser.add_argument(
        "--guided-prototype-semantic-coverage-min-role-mass",
        type=float,
        default=0.05,
        help=(
            "Minimum risk-weighted token mass required for a normal semantic role "
            "to participate in image-conditioned prototype matching."
        ),
    )
    parser.add_argument(
        "--guided-prototype-semantic-coverage-variant",
        choices=("set_match", "selective_hinge"),
        default="set_match",
        help="Use the original injective set match or selective margin coverage.",
    )
    parser.add_argument(
        "--guided-prototype-semantic-coverage-roles",
        default="0,1,2",
        help="Comma-separated normal-role indices used by selective coverage.",
    )
    parser.add_argument(
        "--guided-prototype-semantic-coverage-margin", type=float, default=0.80
    )
    parser.add_argument(
        "--guided-prototype-semantic-coverage-min-confidence", type=float, default=0.0
    )
    parser.add_argument(
        "--guided-prototype-semantic-coverage-max-risk", type=float, default=1.0
    )
    parser.add_argument(
        "--guided-prototype-semantic-coverage-warmup-steps", type=int, default=0
    )
    parser.add_argument(
        "--guided-prototype-semantic-coverage-ramp-steps", type=int, default=0
    )
    parser.add_argument(
        "--guided-prototype-context-variant",
        choices=("off", "a1", "a2", "a2_safe", "a3"),
        default="off",
        help=(
            "Atomic context-familiarity preset. a1 uses fixed-scale transport and grouped gather; "
            "a2 adds normal-calibrated adaptive scales with the legacy semantic grouping; "
            "a2_safe confines calibrated ranks to the transport gate and restores free/native "
            "prototype gather; a3 is the original free-prototype ablation."
        ),
    )
    parser.add_argument(
        "--guided-prototype-context-transport",
        action="store_true",
        help=(
            "A1: replace attention downweighting with context-conditioned normal-center "
            "transport on the prototype aggregation path."
        ),
    )
    parser.add_argument(
        "--guided-prototype-context-adaptive-scale",
        action="store_true",
        help=(
            "A2: fuse context scales using leave-source-out normal reliability and query-context "
            "confidence instead of fixed scale weights."
        ),
    )
    parser.add_argument(
        "--guided-prototype-free-prototypes",
        action="store_true",
        help=(
            "A3: keep context transport but restore the original nearest-prototype gather objective; "
            "the six prototypes no longer have fixed bg/texture/object identities."
        ),
    )
    parser.add_argument("--guided-prototype-context-memory-size", type=int, default=2048)
    parser.add_argument("--guided-prototype-context-candidates-per-image", type=int, default=32)
    parser.add_argument("--guided-prototype-context-candidates-per-group", type=int, default=128)
    parser.add_argument("--guided-prototype-context-topk", type=int, default=5)
    parser.add_argument("--guided-prototype-context-temperature", type=float, default=0.07)
    parser.add_argument("--guided-prototype-context-query-chunk-size", type=int, default=1024)
    parser.add_argument("--guided-prototype-context-key-dim", type=int, default=64)
    parser.add_argument(
        "--guided-prototype-context-memory-build-batches",
        type=int,
        default=0,
        help="Normal batches used to build context memory; 0 scans one loader epoch.",
    )
    parser.add_argument(
        "--guided-prototype-descriptor-variant",
        choices=("off", "b", "c", "d"),
        default="off",
        help=(
            "Atomic semantic-free descriptor ablation: b=scalar write only, "
            "c=b+context-conditioned read route, d=c+physical-objectness activation."
        ),
    )
    parser.add_argument("--guided-prototype-descriptor-radii", default="1,2,4")
    parser.add_argument("--guided-prototype-descriptor-memory-size", type=int, default=176)
    parser.add_argument(
        "--guided-prototype-descriptor-candidates-per-image", type=int, default=16
    )
    parser.add_argument(
        "--guided-prototype-descriptor-candidates-per-group", type=int, default=128
    )
    parser.add_argument("--guided-prototype-descriptor-topk", type=int, default=3)
    parser.add_argument("--guided-prototype-descriptor-temperature", type=float, default=0.07)
    parser.add_argument(
        "--guided-prototype-descriptor-query-chunk-size", type=int, default=1024
    )
    parser.add_argument("--guided-prototype-descriptor-key-dim", type=int, default=64)
    parser.add_argument(
        "--guided-prototype-descriptor-memory-build-batches", type=int, default=0
    )
    parser.add_argument(
        "--guided-prototype-descriptor-source-manifest",
        type=Path,
        default=None,
        help="Normal patch manifest with source_file used for true source-level LOO calibration.",
    )
    parser.add_argument("--guided-prototype-descriptor-read-strength", type=float, default=0.5)
    parser.add_argument(
        "--guided-prototype-descriptor-read-tail-threshold", type=float, default=0.95
    )
    parser.add_argument(
        "--guided-prototype-descriptor-read-tail-upper", type=float, default=1.0
    )
    parser.add_argument("--guided-prototype-descriptor-read-tail-power", type=float, default=1.0)
    parser.add_argument("--guided-prototype-descriptor-read-layer-start", type=int, default=0)
    parser.add_argument(
        "--guided-prototype-descriptor-read-update-cap", type=float, default=0.25
    )
    parser.add_argument(
        "--guided-prototype-descriptor-objectness-floor",
        type=float,
        default=0.5,
        help="D-only multiplier floor: risk *= floor + (1-floor)*physical_objectness.",
    )
    # Keep this call centralized because all train/eval entry points already install
    # prototype-related arguments through this function.
    from .context_normal_prototype import add_context_normal_prototype_args

    add_context_normal_prototype_args(parser)


def _parse_groups(text: str) -> tuple[int, int, int]:
    parts = [int(item) for item in text.replace(",", " ").split() if item.strip()]
    if len(parts) != 3:
        raise ValueError(f"--guided-prototype-groups expects three integers, got {text!r}.")
    if any(value <= 0 for value in parts):
        raise ValueError(f"Guided prototype group sizes must be positive, got {parts}.")
    return tuple(parts)  # type: ignore[return-value]


def _parse_group_weights(text: str) -> tuple[float, float, float]:
    parts = [float(item) for item in text.replace(",", " ").split() if item.strip()]
    if len(parts) != 3:
        raise ValueError(f"--guided-prototype-group-weights expects three numbers, got {text!r}.")
    if any(value <= 0.0 for value in parts):
        raise ValueError(f"Guided prototype group weights must be positive, got {parts}.")
    return tuple(parts)  # type: ignore[return-value]


def _parse_positive_ints(text: str, name: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in text.replace(",", " ").split() if item.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} expects positive integers, got {text!r}.")
    return values


def _parse_positive_weights(text: str, count: int, name: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in text.replace(",", " ").split() if item.strip())
    if len(values) != count or any(value < 0.0 for value in values) or sum(values) <= 0.0:
        raise ValueError(f"{name} expects {count} non-negative weights with positive sum, got {text!r}.")
    total = sum(values)
    return tuple(value / total for value in values)


def guided_config_from_args(args) -> GuidedPrototypeConfig:
    object_kernels = _parse_positive_ints(args.guided_prototype_object_kernels, "--guided-prototype-object-kernels")
    if any(kernel % 2 == 0 for kernel in object_kernels):
        raise ValueError("--guided-prototype-object-kernels expects odd kernel sizes.")
    texture_offsets = _parse_positive_ints(args.guided_prototype_texture_offsets, "--guided-prototype-texture-offsets")
    quantile_low = float(args.guided_prototype_robust_quantile_low)
    quantile_high = float(args.guided_prototype_robust_quantile_high)
    if not 0.0 <= quantile_low < quantile_high <= 1.0:
        raise ValueError("Guided prototype robust quantiles must satisfy 0 <= low < high <= 1.")
    vlm_prior_checkpoint_arg = getattr(args, "guided_prototype_vlm_prior_checkpoint", None)
    vlm_prior_checkpoint = (
        "" if vlm_prior_checkpoint_arg is None else str(vlm_prior_checkpoint_arg)
    )
    vlm_prior_physical_weight = float(
        getattr(args, "guided_prototype_vlm_prior_physical_weight", 0.0)
    )
    if not 0.0 <= vlm_prior_physical_weight <= 1.0:
        raise ValueError("--guided-prototype-vlm-prior-physical-weight must be in [0,1].")
    vlm_prior_fusion_mode = str(
        getattr(args, "guided_prototype_vlm_prior_fusion_mode", "global")
    )
    if vlm_prior_fusion_mode not in {"global", "low_confidence_correction"}:
        raise ValueError(f"Unknown VLM prior fusion mode: {vlm_prior_fusion_mode!r}.")
    vlm_prior_correction_max_weight = float(
        getattr(args, "guided_prototype_vlm_prior_correction_max_weight", 0.5)
    )
    vlm_prior_correction_physical_margin = float(
        getattr(args, "guided_prototype_vlm_prior_correction_physical_margin", 0.15)
    )
    vlm_prior_correction_vlm_margin = float(
        getattr(args, "guided_prototype_vlm_prior_correction_vlm_margin", 0.15)
    )
    if not 0.0 <= vlm_prior_correction_max_weight <= 1.0:
        raise ValueError(
            "--guided-prototype-vlm-prior-correction-max-weight must be in [0,1]."
        )
    if (
        vlm_prior_correction_physical_margin <= 0.0
        or vlm_prior_correction_vlm_margin <= 0.0
    ):
        raise ValueError("VLM prior correction margins must be positive.")
    group_temperature = float(args.guided_prototype_group_temperature)
    group_confidence_margin = float(args.guided_prototype_group_confidence_margin)
    intra_group_balance_weight = float(
        getattr(args, "guided_prototype_intra_group_balance_weight", 0.0)
    )
    intra_group_temperature = float(
        getattr(args, "guided_prototype_intra_group_temperature", 0.10)
    )
    intra_group_repulsion_weight = float(
        getattr(args, "guided_prototype_intra_group_repulsion_weight", 0.0)
    )
    intra_group_repulsion_margin = float(
        getattr(args, "guided_prototype_intra_group_repulsion_margin", 0.20)
    )
    memory_mode_weight = float(getattr(args, "guided_prototype_memory_mode_weight", 0.0))
    memory_mode_temperature = float(
        getattr(args, "guided_prototype_memory_mode_temperature", 0.10)
    )
    memory_mode_teacher_temperature = float(
        getattr(args, "guided_prototype_memory_mode_teacher_temperature", 0.10)
    )
    memory_mode_margin_weighting = bool(
        getattr(args, "guided_prototype_memory_mode_margin_weighting", False)
    )
    memory_mode_soft_targets = bool(
        getattr(args, "guided_prototype_memory_mode_soft_targets", False)
    )
    memory_mode_soft_semantic = bool(
        getattr(args, "guided_prototype_memory_mode_soft_semantic", False)
    )
    memory_mode_semantic_margin = float(
        getattr(args, "guided_prototype_memory_mode_semantic_margin", 0.15)
    )
    center6_balanced = bool(
        getattr(args, "guided_prototype_center6_balanced", False)
    )
    center6_reduction = str(
        getattr(args, "guided_prototype_center6_reduction", "equal_mode")
    )
    center6_temperature = float(
        getattr(args, "guided_prototype_center6_temperature", 0.10)
    )
    center6_teacher_temperature = float(
        getattr(args, "guided_prototype_center6_teacher_temperature", 0.10)
    )
    center6_prior_mapping = str(
        getattr(args, "guided_prototype_center6_prior_mapping", "soft")
    )
    center6_loss_mapping = str(
        getattr(args, "guided_prototype_center6_loss_mapping", "soft")
    )
    center6_radius_quantile = float(
        getattr(args, "guided_prototype_center6_radius_quantile", 0.95)
    )
    center6_radius_floor = float(
        getattr(args, "guided_prototype_center6_radius_floor", 1e-3)
    )
    center6_hierarchical_teacher = bool(
        getattr(args, "guided_prototype_center6_hierarchical_teacher", False)
    )
    center6_novelty_veto = bool(
        getattr(args, "guided_prototype_center6_novelty_veto", False)
    )
    center6_novelty_threshold = float(
        getattr(args, "guided_prototype_center6_novelty_threshold", 1.0)
    )
    center6_novelty_temperature = float(
        getattr(args, "guided_prototype_center6_novelty_temperature", 0.10)
    )
    center6_novelty_min_weight = float(
        getattr(args, "guided_prototype_center6_novelty_min_weight", 0.05)
    )
    center6_hierarchical_reliability = bool(
        getattr(args, "guided_prototype_center6_hierarchical_reliability", False)
    )
    center6_collapsed_slot_diversity_weight = float(
        getattr(args, "guided_prototype_center6_collapsed_slot_diversity_weight", 0.0)
    )
    center6_collapsed_slot_diversity_margin = float(
        getattr(args, "guided_prototype_center6_collapsed_slot_diversity_margin", 0.90)
    )
    if group_temperature <= 0.0:
        raise ValueError("--guided-prototype-group-temperature must be positive.")
    if not 0.0 <= group_confidence_margin <= 1.0:
        raise ValueError("--guided-prototype-group-confidence-margin must be in [0, 1].")
    if (
        intra_group_balance_weight < 0.0
        or intra_group_repulsion_weight < 0.0
        or memory_mode_weight < 0.0
    ):
        raise ValueError("Guided prototype intra-group loss weights must be non-negative.")
    if intra_group_temperature <= 0.0:
        raise ValueError("--guided-prototype-intra-group-temperature must be positive.")
    if not -1.0 <= intra_group_repulsion_margin <= 1.0:
        raise ValueError("--guided-prototype-intra-group-repulsion-margin must be in [-1,1].")
    if memory_mode_temperature <= 0.0:
        raise ValueError("--guided-prototype-memory-mode-temperature must be positive.")
    if memory_mode_teacher_temperature <= 0.0:
        raise ValueError("--guided-prototype-memory-mode-teacher-temperature must be positive.")
    if not 0.0 <= memory_mode_semantic_margin <= 1.0:
        raise ValueError("--guided-prototype-memory-mode-semantic-margin must be in [0,1].")
    if center6_temperature <= 0.0 or center6_teacher_temperature <= 0.0:
        raise ValueError("Center6 student and teacher temperatures must be positive.")
    if center6_prior_mapping not in {"soft", "hard"}:
        raise ValueError(f"Unknown Center6 prior mapping: {center6_prior_mapping!r}.")
    if center6_loss_mapping not in {"soft", "hard"}:
        raise ValueError(f"Unknown Center6 loss mapping: {center6_loss_mapping!r}.")
    if center6_reduction not in {"equal_mode", "token_mean", "sqrt_balanced"}:
        raise ValueError(f"Unsupported Center6 reduction: {center6_reduction!r}")
    if not 0.0 < center6_radius_quantile <= 1.0:
        raise ValueError("--guided-prototype-center6-radius-quantile must be in (0,1].")
    if center6_radius_floor <= 0.0:
        raise ValueError("--guided-prototype-center6-radius-floor must be positive.")
    if center6_novelty_threshold <= 0.0 or center6_novelty_temperature <= 0.0:
        raise ValueError("Center6 novelty threshold and temperature must be positive.")
    if not 0.0 <= center6_novelty_min_weight <= 1.0:
        raise ValueError("Center6 novelty minimum weight must be in [0,1].")
    if center6_collapsed_slot_diversity_weight < 0.0:
        raise ValueError("Collapsed-slot diversity weight must be non-negative.")
    if not -1.0 <= center6_collapsed_slot_diversity_margin <= 1.0:
        raise ValueError("Collapsed-slot diversity margin must be in [-1,1].")
    aggregation_min_weight = float(args.guided_prototype_aggregation_min_weight)
    aggregation_power = float(args.guided_prototype_aggregation_power)
    aggregation_alpha_arg = getattr(args, "guided_prototype_aggregation_alpha", None)
    aggregation_alpha = (
        None if aggregation_alpha_arg is None else float(aggregation_alpha_arg)
    )
    mode_specific_routing = bool(
        getattr(args, "guided_prototype_mode_routing", False)
    )
    mode_routing_floor = float(
        getattr(args, "guided_prototype_mode_routing_floor", 0.05)
    )
    mode_routing_strength = float(
        getattr(args, "guided_prototype_mode_routing_strength", 1.0)
    )
    if not 0.0 < aggregation_min_weight <= 1.0:
        raise ValueError("--guided-prototype-aggregation-min-weight must be in (0, 1].")
    if aggregation_power <= 0.0:
        raise ValueError("--guided-prototype-aggregation-power must be positive.")
    if aggregation_alpha is not None and not 0.0 <= aggregation_alpha <= 1.0:
        raise ValueError("--guided-prototype-aggregation-alpha must be in [0,1].")
    if not 0.0 < mode_routing_floor <= 1.0:
        raise ValueError("--guided-prototype-mode-routing-floor must be in (0,1].")
    if not 0.0 <= mode_routing_strength <= 1.0:
        raise ValueError("--guided-prototype-mode-routing-strength must be in [0,1].")
    familiarity_memory_size = int(args.guided_prototype_familiarity_memory_size)
    familiarity_tokens_per_image = int(args.guided_prototype_familiarity_tokens_per_image)
    familiarity_min_count = int(args.guided_prototype_familiarity_min_count)
    familiarity_novelty_floor = float(args.guided_prototype_familiarity_novelty_floor)
    familiarity_temperature = float(args.guided_prototype_familiarity_temperature)
    familiarity_novelty_mapping = str(
        getattr(args, "guided_prototype_familiarity_novelty_mapping", "sigmoid")
    )
    familiarity_calibration_mode = str(
        getattr(args, "guided_prototype_familiarity_calibration", "legacy")
    )
    familiarity_calibration_quantile = float(
        getattr(args, "guided_prototype_familiarity_calibration_quantile", 0.95)
    )
    familiarity_write_enabled = not bool(
        getattr(args, "guided_prototype_familiarity_read_only", False)
    )
    aggregation_risk_composition = str(
        getattr(args, "guided_prototype_aggregation_risk_composition", "product")
    )
    target_gate_calibration = bool(
        getattr(args, "guided_prototype_target_gate_calibration", False)
    )
    target_gate_risk_source = str(
        getattr(args, "guided_prototype_target_gate_risk_source", "familiarity")
    )
    target_reject_gate = bool(
        getattr(args, "guided_prototype_target_reject_gate", False)
    )
    target_mode_gate = bool(
        getattr(args, "guided_prototype_target_mode_gate", False)
    )
    target_mode_gate_min_support = int(
        getattr(args, "guided_prototype_target_mode_gate_min_support", 8)
    )
    if target_mode_gate_min_support <= 0:
        raise ValueError("Target mode-gate minimum support must be positive.")
    target_mode_novelty_temperature = float(
        getattr(args, "guided_prototype_target_mode_novelty_temperature", 0.10)
    )
    if target_mode_novelty_temperature <= 0.0:
        raise ValueError("Target mode-novelty temperature must be positive.")
    decoder_read_gate = bool(getattr(args, "guided_prototype_decoder_read_gate", False))
    decoder_read_strength = float(
        getattr(args, "guided_prototype_decoder_read_strength", 0.5)
    )
    decoder_read_scope = str(
        getattr(args, "guided_prototype_decoder_read_scope", "object")
    )
    decoder_read_mode = str(
        getattr(args, "guided_prototype_decoder_read_mode", "suppress")
    )
    decoder_read_tail_threshold = float(
        getattr(args, "guided_prototype_decoder_read_tail_threshold", 0.0)
    )
    decoder_read_tail_upper = float(
        getattr(args, "guided_prototype_decoder_read_tail_upper", 1.0)
    )
    decoder_read_tail_power = float(
        getattr(args, "guided_prototype_decoder_read_tail_power", 1.0)
    )
    decoder_read_attention_aware = bool(
        getattr(args, "guided_prototype_decoder_read_attention_aware", False)
    )
    decoder_read_responsibility_aware = bool(
        getattr(args, "guided_prototype_decoder_read_responsibility_aware", False)
    )
    decoder_read_update_cap = float(
        getattr(args, "guided_prototype_decoder_read_update_cap", 0.0)
    )
    decoder_read_layer_start = int(
        getattr(args, "guided_prototype_decoder_read_layer_start", 0)
    )
    decoder_read_layer_strengths_raw = str(
        getattr(args, "guided_prototype_decoder_read_layer_strengths", "")
    ).strip()
    decoder_read_layer_strengths = (
        tuple(float(value.strip()) for value in decoder_read_layer_strengths_raw.split(","))
        if decoder_read_layer_strengths_raw
        else ()
    )
    decoder_read_adaptive_strength_power = float(
        getattr(args, "guided_prototype_decoder_read_adaptive_strength_power", 0.0)
    )
    decoder_read_risk_source = str(
        getattr(args, "guided_prototype_decoder_read_risk_source", "aggregation")
    )
    decoder_read_prototypewise = bool(
        getattr(args, "guided_prototype_decoder_read_prototypewise", False)
    )
    decoder_read_center6_support_alpha = float(
        getattr(args, "guided_prototype_decoder_read_center6_support_alpha", 0.5)
    )
    decoder_read_center6_support_temperature = float(
        getattr(args, "guided_prototype_decoder_read_center6_support_temperature", 0.10)
    )
    decoder_read_prototype_responsibility_floor = float(
        getattr(args, "guided_prototype_decoder_read_prototype_responsibility_floor", 0.5)
    )
    decoder_read_center6_mode_tail_thresholds_raw = str(
        getattr(args, "guided_prototype_decoder_read_center6_mode_tail_thresholds", "")
    ).strip()
    decoder_read_center6_mode_tail_thresholds = (
        tuple(
            float(value.strip())
            for value in decoder_read_center6_mode_tail_thresholds_raw.split(",")
        )
        if decoder_read_center6_mode_tail_thresholds_raw
        else ()
    )
    decoder_read_center6_mode_tail_uppers_raw = str(
        getattr(args, "guided_prototype_decoder_read_center6_mode_tail_uppers", "")
    ).strip()
    decoder_read_center6_mode_tail_uppers = (
        tuple(
            float(value.strip())
            for value in decoder_read_center6_mode_tail_uppers_raw.split(",")
        )
        if decoder_read_center6_mode_tail_uppers_raw
        else ()
    )
    native_anchor_alpha = float(getattr(args, "guided_prototype_native_anchor_alpha", 1.0))
    guided_distill_weight = float(
        getattr(args, "guided_prototype_distill_weight", 0.0)
    )
    semantic_coverage_weight = float(
        getattr(args, "guided_prototype_semantic_coverage_weight", 0.0)
    )
    semantic_coverage_min_role_mass = float(
        getattr(args, "guided_prototype_semantic_coverage_min_role_mass", 0.05)
    )
    semantic_coverage_variant = str(
        getattr(args, "guided_prototype_semantic_coverage_variant", "set_match")
    )
    semantic_coverage_roles = tuple(
        int(value.strip())
        for value in str(
            getattr(args, "guided_prototype_semantic_coverage_roles", "0,1,2")
        ).split(",")
        if value.strip()
    )
    semantic_coverage_margin = float(
        getattr(args, "guided_prototype_semantic_coverage_margin", 0.80)
    )
    semantic_coverage_min_confidence = float(
        getattr(args, "guided_prototype_semantic_coverage_min_confidence", 0.0)
    )
    semantic_coverage_max_risk = float(
        getattr(args, "guided_prototype_semantic_coverage_max_risk", 1.0)
    )
    semantic_coverage_warmup_steps = int(
        getattr(args, "guided_prototype_semantic_coverage_warmup_steps", 0)
    )
    semantic_coverage_ramp_steps = int(
        getattr(args, "guided_prototype_semantic_coverage_ramp_steps", 0)
    )
    context_variant = str(getattr(args, "guided_prototype_context_variant", "off"))
    direct_context_transport = bool(
        getattr(args, "guided_prototype_context_transport", False)
    )
    direct_context_adaptive_scale = bool(
        getattr(args, "guided_prototype_context_adaptive_scale", False)
    )
    direct_free_prototypes = bool(getattr(args, "guided_prototype_free_prototypes", False))
    if context_variant not in {"off", "a1", "a2", "a2_safe", "a3"}:
        raise ValueError(f"Unknown context-familiarity variant: {context_variant!r}.")
    if context_variant == "a1" and direct_context_adaptive_scale:
        raise ValueError("A1 uses fixed-scale transport; do not also enable context-adaptive-scale.")
    if context_variant in {"a1", "a2"} and direct_free_prototypes:
        raise ValueError(f"{context_variant.upper()} uses grouped prototypes; free prototypes define A3.")
    if context_variant == "off":
        context_transport = direct_context_transport
        context_adaptive_scale = direct_context_adaptive_scale
        free_prototypes = direct_free_prototypes
        resolved_context_variant = "custom" if context_transport else "off"
    else:
        context_transport = True
        context_adaptive_scale = context_variant in {"a2", "a2_safe", "a3"}
        # A2-safe deliberately keeps the calibrated objectness percentile inside
        # the transport gate. Free prototypes prevent that percentile from being
        # reused as a semantic group label in the gather objective.
        free_prototypes = context_variant in {"a2_safe", "a3"}
        resolved_context_variant = context_variant
    context_memory_size = int(getattr(args, "guided_prototype_context_memory_size", 2048))
    context_candidates_per_image = int(
        getattr(args, "guided_prototype_context_candidates_per_image", 32)
    )
    context_candidates_per_group = int(
        getattr(args, "guided_prototype_context_candidates_per_group", 128)
    )
    context_topk = int(getattr(args, "guided_prototype_context_topk", 5))
    context_temperature = float(getattr(args, "guided_prototype_context_temperature", 0.07))
    context_query_chunk_size = int(
        getattr(args, "guided_prototype_context_query_chunk_size", 1024)
    )
    context_memory_build_batches = int(
        getattr(args, "guided_prototype_context_memory_build_batches", 0)
    )
    context_key_dim = int(getattr(args, "guided_prototype_context_key_dim", 64))
    descriptor_variant = str(
        getattr(args, "guided_prototype_descriptor_variant", "off")
    )
    descriptor_radii = _parse_positive_ints(
        str(getattr(args, "guided_prototype_descriptor_radii", "1,2,4")),
        "--guided-prototype-descriptor-radii",
    )
    descriptor_memory_size = int(
        getattr(args, "guided_prototype_descriptor_memory_size", 176)
    )
    descriptor_candidates_per_image = int(
        getattr(args, "guided_prototype_descriptor_candidates_per_image", 16)
    )
    descriptor_candidates_per_group = int(
        getattr(args, "guided_prototype_descriptor_candidates_per_group", 128)
    )
    descriptor_topk = int(getattr(args, "guided_prototype_descriptor_topk", 3))
    descriptor_temperature = float(
        getattr(args, "guided_prototype_descriptor_temperature", 0.07)
    )
    descriptor_query_chunk_size = int(
        getattr(args, "guided_prototype_descriptor_query_chunk_size", 1024)
    )
    descriptor_key_dim = int(
        getattr(args, "guided_prototype_descriptor_key_dim", 64)
    )
    descriptor_memory_build_batches = int(
        getattr(args, "guided_prototype_descriptor_memory_build_batches", 0)
    )
    descriptor_source_manifest_arg = getattr(
        args, "guided_prototype_descriptor_source_manifest", None
    )
    descriptor_source_manifest = (
        ""
        if descriptor_source_manifest_arg is None
        else str(descriptor_source_manifest_arg)
    )
    descriptor_read_strength = float(
        getattr(args, "guided_prototype_descriptor_read_strength", 0.5)
    )
    descriptor_read_tail_threshold = float(
        getattr(args, "guided_prototype_descriptor_read_tail_threshold", 0.95)
    )
    descriptor_read_tail_upper = float(
        getattr(args, "guided_prototype_descriptor_read_tail_upper", 1.0)
    )
    descriptor_read_tail_power = float(
        getattr(args, "guided_prototype_descriptor_read_tail_power", 1.0)
    )
    descriptor_read_layer_start = int(
        getattr(args, "guided_prototype_descriptor_read_layer_start", 0)
    )
    descriptor_read_update_cap = float(
        getattr(args, "guided_prototype_descriptor_read_update_cap", 0.25)
    )
    descriptor_objectness_floor = float(
        getattr(args, "guided_prototype_descriptor_objectness_floor", 0.5)
    )
    spatial_scale_mode = str(getattr(args, "guided_prototype_spatial_scale_mode", "legacy"))
    spatial_reference_side = int(getattr(args, "guided_prototype_spatial_reference_side", 32))
    if spatial_scale_mode not in {"legacy", "integer"}:
        raise ValueError(f"Unknown guided prototype spatial scale mode: {spatial_scale_mode!r}.")
    if spatial_reference_side <= 0:
        raise ValueError("--guided-prototype-spatial-reference-side must be positive.")
    if spatial_scale_mode == "integer" and bool(getattr(args, "guided_prototype_scale_consistent", False)):
        raise ValueError(
            "Integer spatial scaling operates on the current token grid and cannot be combined with "
            "--guided-prototype-scale-consistent reference-grid interpolation."
        )
    if familiarity_memory_size <= 1:
        raise ValueError("--guided-prototype-familiarity-memory-size must be greater than one.")
    if familiarity_tokens_per_image <= 0:
        raise ValueError("--guided-prototype-familiarity-tokens-per-image must be positive.")
    if not 1 < familiarity_min_count <= familiarity_memory_size:
        raise ValueError("Familiarity min count must be in (1, memory size].")
    if not 0.0 <= familiarity_novelty_floor < 2.0:
        raise ValueError("--guided-prototype-familiarity-novelty-floor must be in [0, 2).")
    if familiarity_temperature <= 0.0:
        raise ValueError("--guided-prototype-familiarity-temperature must be positive.")
    if familiarity_novelty_mapping not in {"sigmoid", "hard"}:
        raise ValueError(
            f"Unknown familiarity novelty mapping: {familiarity_novelty_mapping!r}."
        )
    if familiarity_calibration_mode not in {"legacy", "cross_group"}:
        raise ValueError(f"Unknown familiarity calibration mode: {familiarity_calibration_mode!r}.")
    if not 0.0 < familiarity_calibration_quantile < 1.0:
        raise ValueError("Familiarity calibration quantile must be in (0,1).")
    if not 0.0 <= decoder_read_strength <= 1.0:
        raise ValueError("--guided-prototype-decoder-read-strength must be in [0,1].")
    if decoder_read_scope not in {"object", "all"}:
        raise ValueError(f"Unknown decoder read-gate scope: {decoder_read_scope!r}.")
    if decoder_read_mode not in {"suppress", "tail_suppress", "selective_route"}:
        raise ValueError(f"Unknown decoder read-gate mode: {decoder_read_mode!r}.")
    if not 0.0 <= decoder_read_tail_threshold < decoder_read_tail_upper <= 1.0:
        raise ValueError(
            "Decoder read-gate tail calibration must satisfy "
            "0 <= threshold < upper <= 1."
        )
    if decoder_read_tail_power <= 0.0:
        raise ValueError("--guided-prototype-decoder-read-tail-power must be positive.")
    all_mode_radius_tail = bool(
        decoder_read_mode == "tail_suppress"
        and decoder_read_scope == "all"
        and decoder_read_prototypewise
        and decoder_read_risk_source == "center6_radius"
    )
    if (
        decoder_read_mode in {"tail_suppress", "selective_route"}
        and decoder_read_scope != "object"
        and not all_mode_radius_tail
    ):
        raise ValueError(
            "Tail-calibrated decoder reads require scope 'object', except prototype-wise "
            "Center6-radius tail suppression across all six modes."
        )
    if decoder_read_attention_aware and decoder_read_scope != "object":
        raise ValueError("Attention-aware decoder read gating requires read scope 'object'.")
    if decoder_read_responsibility_aware and decoder_read_scope != "object":
        raise ValueError(
            "Responsibility-aware decoder read gating requires read scope 'object'."
        )
    if not 0.0 <= decoder_read_update_cap <= 1.0:
        raise ValueError("--guided-prototype-decoder-read-update-cap must be in [0,1].")
    if decoder_read_layer_start < 0:
        raise ValueError("--guided-prototype-decoder-read-layer-start must be non-negative.")
    if any(not 0.0 <= value <= 1.0 for value in decoder_read_layer_strengths):
        raise ValueError("Every decoder read layer strength must be in [0,1].")
    if decoder_read_adaptive_strength_power < 0.0:
        raise ValueError("Decoder read adaptive-strength power must be non-negative.")
    if decoder_read_risk_source not in {
        "aggregation",
        "physical",
        "center6_hybrid",
        "center6_radius",
        "center6_global",
        "center6_mode_novelty",
        "target_gate",
    }:
        raise ValueError(f"Unknown decoder read risk source: {decoder_read_risk_source!r}.")
    if not 0.0 <= decoder_read_center6_support_alpha <= 1.0:
        raise ValueError("Center6 decoder-read support alpha must be in [0,1].")
    if decoder_read_center6_support_temperature <= 0.0:
        raise ValueError("Center6 decoder-read support temperature must be positive.")
    if not 0.0 <= decoder_read_prototype_responsibility_floor <= 1.0:
        raise ValueError("Decoder-read prototype responsibility floor must be in [0,1].")
    if decoder_read_prototypewise and decoder_read_risk_source not in {
        "center6_hybrid",
        "center6_radius",
    }:
        raise ValueError("Prototype-wise decoder reads require a Center6 risk source.")
    if decoder_read_responsibility_aware and decoder_read_prototypewise:
        raise ValueError(
            "Responsibility-aware scalar reads cannot be combined with prototype-wise risk."
        )
    if decoder_read_risk_source == "target_gate" and not target_gate_calibration:
        raise ValueError("TargetGate decoder reads require target-gate calibration.")
    if decoder_read_risk_source == "center6_radius" and not decoder_read_prototypewise:
        raise ValueError("Center6-radius decoder reads require prototype-wise risk.")
    if decoder_read_risk_source == "center6_global" and decoder_read_prototypewise:
        raise ValueError("Center6-global decoder reads use one scalar risk per token.")
    if decoder_read_risk_source == "center6_mode_novelty":
        prototype_count = sum(_parse_groups(args.guided_prototype_groups))
        if decoder_read_prototypewise:
            raise ValueError("Center6 mode-conditioned novelty uses one scalar risk per token.")
        if (
            len(decoder_read_center6_mode_tail_thresholds) != prototype_count
            or len(decoder_read_center6_mode_tail_uppers) != prototype_count
        ):
            raise ValueError(
                "Center6 mode-conditioned novelty requires one tail threshold and upper "
                "for every prototype mode."
            )
        for threshold, upper in zip(
            decoder_read_center6_mode_tail_thresholds,
            decoder_read_center6_mode_tail_uppers,
        ):
            if not 0.0 <= threshold < upper <= 1.0:
                raise ValueError(
                    "Every Center6 mode novelty calibration must satisfy "
                    "0 <= threshold < upper <= 1."
                )
    if not 0.0 <= native_anchor_alpha <= 1.0:
        raise ValueError("--guided-prototype-native-anchor-alpha must be in [0,1].")
    if guided_distill_weight < 0.0:
        raise ValueError("--guided-prototype-distill-weight must be non-negative.")
    if guided_distill_weight > 0.0 and native_anchor_alpha != 0.0:
        raise ValueError(
            "Auxiliary guided distillation keeps full native coherence and therefore "
            "requires --guided-prototype-native-anchor-alpha 0."
        )
    if semantic_coverage_weight < 0.0:
        raise ValueError(
            "--guided-prototype-semantic-coverage-weight must be non-negative."
        )
    if not 0.0 < semantic_coverage_min_role_mass <= 1.0:
        raise ValueError(
            "--guided-prototype-semantic-coverage-min-role-mass must be in (0,1]."
        )
    if semantic_coverage_variant not in {"set_match", "selective_hinge"}:
        raise ValueError("Unknown semantic coverage variant.")
    if not semantic_coverage_roles or len(set(semantic_coverage_roles)) != len(
        semantic_coverage_roles
    ):
        raise ValueError("Semantic coverage roles must be non-empty and unique.")
    if any(role < 0 or role >= 3 for role in semantic_coverage_roles):
        raise ValueError("Semantic coverage roles must be selected from 0,1,2.")
    if not 0.0 <= semantic_coverage_margin <= 1.0:
        raise ValueError("Semantic coverage margin must be in [0,1].")
    if not 0.0 <= semantic_coverage_min_confidence <= 1.0:
        raise ValueError("Semantic coverage minimum confidence must be in [0,1].")
    if not 0.0 <= semantic_coverage_max_risk <= 1.0:
        raise ValueError("Semantic coverage maximum risk must be in [0,1].")
    if semantic_coverage_warmup_steps < 0 or semantic_coverage_ramp_steps < 0:
        raise ValueError("Semantic coverage warmup/ramp steps must be non-negative.")
    trainable_prior = bool(getattr(args, "guided_prototype_trainable_prior", False))
    if context_adaptive_scale and not context_transport:
        raise ValueError("Context-adaptive scale requires --guided-prototype-context-transport.")
    if free_prototypes and not context_transport:
        raise ValueError("Free prototypes are defined for the context-transport ablation.")
    if free_prototypes and not context_adaptive_scale:
        raise ValueError("Free prototypes define A3 and require context-adaptive scale.")
    if context_transport and trainable_prior:
        raise ValueError(
            "Context-familiarity variants use a normal-calibrated fixed prior; "
            "do not combine them with --guided-prototype-trainable-prior."
        )
    if context_transport and bool(getattr(args, "guided_prototype_aggregation_gate", False)):
        raise ValueError("Context transport replaces the old aggregation gate; do not enable both.")
    if context_variant != "off" and bool(
        getattr(args, "guided_prototype_group_contrastive", False)
    ):
        raise ValueError(
            "Atomic A1/A2/A3 presets use their declared gather objective; "
            "group-contrastive is a separate custom ablation."
        )
    if context_memory_size <= 1:
        raise ValueError("--guided-prototype-context-memory-size must be greater than one.")
    if context_candidates_per_image <= 0 or context_candidates_per_group <= 0:
        raise ValueError("Context transport candidate counts must be positive.")
    if context_topk <= 0 or context_topk > context_memory_size:
        raise ValueError("Context transport top-k must be positive and no larger than memory size.")
    if context_temperature <= 0.0 or context_query_chunk_size <= 0 or context_key_dim <= 0:
        raise ValueError("Context transport temperature, query chunk size and key dim must be positive.")
    if context_memory_build_batches < 0:
        raise ValueError("Context transport memory-build batches cannot be negative.")
    if descriptor_variant not in {"off", "b", "c", "d"}:
        raise ValueError(f"Unknown descriptor gate variant: {descriptor_variant!r}.")
    if descriptor_memory_size <= 1:
        raise ValueError("Descriptor memory size must be greater than one.")
    if descriptor_candidates_per_image <= 0 or descriptor_candidates_per_group <= 0:
        raise ValueError("Descriptor candidate counts must be positive.")
    if descriptor_topk <= 0 or descriptor_topk > descriptor_memory_size:
        raise ValueError("Descriptor top-k must be positive and no larger than memory size.")
    if descriptor_temperature <= 0.0 or descriptor_query_chunk_size <= 0 or descriptor_key_dim <= 0:
        raise ValueError("Descriptor temperature, query chunk size and key dim must be positive.")
    if descriptor_memory_build_batches < 0:
        raise ValueError("Descriptor memory-build batches cannot be negative.")
    if not 0.0 <= descriptor_read_strength <= 1.0:
        raise ValueError("Descriptor read strength must be in [0,1].")
    if not 0.0 <= descriptor_read_tail_threshold < descriptor_read_tail_upper <= 1.0:
        raise ValueError("Descriptor read tail calibration must satisfy 0 <= threshold < upper <= 1.")
    if descriptor_read_tail_power <= 0.0:
        raise ValueError("Descriptor read tail power must be positive.")
    if descriptor_read_layer_start < 0:
        raise ValueError("Descriptor read layer start must be non-negative.")
    if not 0.0 <= descriptor_read_update_cap <= 1.0:
        raise ValueError("Descriptor read update cap must be in [0,1].")
    if not 0.0 <= descriptor_objectness_floor <= 1.0:
        raise ValueError("Descriptor objectness floor must be in [0,1].")
    descriptor_enabled = descriptor_variant != "off"
    if descriptor_enabled:
        forbidden = {
            "context transport": context_transport,
            "legacy aggregation gate": bool(
                getattr(args, "guided_prototype_aggregation_gate", False)
            ),
            "legacy familiarity gate": bool(
                getattr(args, "guided_prototype_familiarity_gate", False)
            ),
            "legacy decoder read gate": decoder_read_gate,
            "Center6": center6_balanced,
            "memory-mode teacher": memory_mode_weight > 0.0,
            "group contrastive": bool(
                getattr(args, "guided_prototype_group_contrastive", False)
            ),
            "trainable prior": trainable_prior,
        }
        conflicts = [name for name, active in forbidden.items() if active]
        if conflicts:
            raise ValueError(
                "Descriptor B/C/D is an atomic semantic-free ablation; disable "
                + ", ".join(conflicts)
                + "."
            )
        free_prototypes = True
    preset_enabled = context_variant != "off"
    return GuidedPrototypeConfig(
        groups=(2, 2, 2) if preset_enabled else _parse_groups(args.guided_prototype_groups),
        prior_floor=float(args.guided_prototype_prior_floor),
        texture_power=float(args.guided_prototype_texture_power),
        object_power=float(args.guided_prototype_object_power),
        feature_texture_weight=float(args.guided_prototype_feature_texture_weight),
        image_texture_weight=float(args.guided_prototype_image_texture_weight),
        feature_object_weight=float(args.guided_prototype_feature_object_weight),
        image_object_weight=float(args.guided_prototype_image_object_weight),
        object_texture_suppress=float(args.guided_prototype_object_texture_suppress),
        vlm_prior_checkpoint=vlm_prior_checkpoint,
        vlm_prior_physical_weight=vlm_prior_physical_weight,
        vlm_prior_fusion_mode=vlm_prior_fusion_mode,
        vlm_prior_correction_max_weight=vlm_prior_correction_max_weight,
        vlm_prior_correction_physical_margin=vlm_prior_correction_physical_margin,
        vlm_prior_correction_vlm_margin=vlm_prior_correction_vlm_margin,
        separation_weight=float(args.guided_prototype_separation_weight),
        separation_margin=float(args.guided_prototype_separation_margin),
        trainable_prior=trainable_prior,
        prior_hidden_dim=int(getattr(args, "guided_prototype_prior_hidden_dim", 0)),
        prior_anchor_weight=float(getattr(args, "guided_prototype_prior_anchor_weight", 0.0)),
        normal_guard_weight=float(getattr(args, "guided_prototype_normal_guard_weight", 0.0)),
        normal_guard_beta=float(getattr(args, "guided_prototype_normal_guard_beta", 0.5)),
        normal_guard_top_frac=float(getattr(args, "guided_prototype_normal_guard_top_frac", 0.15)),
        scale_consistent=(
            False if preset_enabled else bool(getattr(args, "guided_prototype_scale_consistent", False))
        ),
        prior_reference_side=int(getattr(args, "guided_prototype_prior_reference_side", 32)),
        image_reference_size=int(getattr(args, "guided_prototype_image_reference_size", 448)),
        group_weights=(
            (0.92, 1.10, 1.10)
            if preset_enabled
            else _parse_group_weights(getattr(args, "guided_prototype_group_weights", "1,1,1"))
        ),
        multiscale_direct=(
            True if preset_enabled else bool(getattr(args, "guided_prototype_multiscale_direct", False))
        ),
        spatial_scale_mode=spatial_scale_mode,
        spatial_reference_side=spatial_reference_side,
        object_kernels=(3, 5, 7) if preset_enabled else object_kernels,
        object_kernel_weights=(
            (0.35, 0.25, 0.40)
            if preset_enabled
            else _parse_positive_weights(
                args.guided_prototype_object_kernel_weights,
                len(object_kernels),
                "--guided-prototype-object-kernel-weights",
            )
        ),
        texture_offsets=texture_offsets,
        texture_offset_weights=_parse_positive_weights(
            args.guided_prototype_texture_offset_weights,
            len(texture_offsets),
            "--guided-prototype-texture-offset-weights",
        ),
        robust_quantile_low=quantile_low,
        robust_quantile_high=quantile_high,
        robust_quantile_momentum=float(args.guided_prototype_robust_quantile_momentum),
        group_contrastive=bool(getattr(args, "guided_prototype_group_contrastive", False)),
        group_temperature=group_temperature,
        group_confidence_margin=group_confidence_margin,
        intra_group_balance_weight=intra_group_balance_weight,
        intra_group_temperature=intra_group_temperature,
        intra_group_repulsion_weight=intra_group_repulsion_weight,
        intra_group_repulsion_margin=intra_group_repulsion_margin,
        memory_mode_weight=memory_mode_weight,
        memory_mode_temperature=memory_mode_temperature,
        memory_mode_teacher_temperature=memory_mode_teacher_temperature,
        memory_mode_margin_weighting=memory_mode_margin_weighting,
        memory_mode_soft_targets=memory_mode_soft_targets,
        memory_mode_soft_semantic=memory_mode_soft_semantic,
        memory_mode_semantic_margin=memory_mode_semantic_margin,
        center6_balanced=center6_balanced,
        center6_reduction=center6_reduction,
        center6_temperature=center6_temperature,
        center6_teacher_temperature=center6_teacher_temperature,
        center6_prior_mapping=center6_prior_mapping,
        center6_loss_mapping=center6_loss_mapping,
        center6_radius_quantile=center6_radius_quantile,
        center6_radius_floor=center6_radius_floor,
        center6_hierarchical_teacher=center6_hierarchical_teacher,
        center6_novelty_veto=center6_novelty_veto,
        center6_novelty_threshold=center6_novelty_threshold,
        center6_novelty_temperature=center6_novelty_temperature,
        center6_novelty_min_weight=center6_novelty_min_weight,
        center6_hierarchical_reliability=center6_hierarchical_reliability,
        center6_collapsed_slot_diversity_weight=(
            center6_collapsed_slot_diversity_weight
        ),
        center6_collapsed_slot_diversity_margin=(
            center6_collapsed_slot_diversity_margin
        ),
        aggregation_gate=bool(getattr(args, "guided_prototype_aggregation_gate", False)),
        aggregation_min_weight=aggregation_min_weight,
        aggregation_power=aggregation_power,
        aggregation_alpha=aggregation_alpha,
        mode_specific_routing=mode_specific_routing,
        mode_routing_floor=mode_routing_floor,
        mode_routing_strength=mode_routing_strength,
        aggregation_familiarity_gate=bool(
            getattr(args, "guided_prototype_familiarity_gate", False)
        ),
        aggregation_risk_composition=aggregation_risk_composition,
        familiarity_write_enabled=familiarity_write_enabled,
        target_gate_calibration=target_gate_calibration,
        target_gate_risk_source=target_gate_risk_source,
        target_reject_gate=target_reject_gate,
        target_mode_gate=target_mode_gate,
        target_mode_gate_min_support=target_mode_gate_min_support,
        target_mode_novelty_temperature=target_mode_novelty_temperature,
        decoder_read_gate=decoder_read_gate,
        decoder_read_strength=decoder_read_strength,
        decoder_read_scope=decoder_read_scope,
        decoder_read_mode=decoder_read_mode,
        decoder_read_tail_threshold=decoder_read_tail_threshold,
        decoder_read_tail_upper=decoder_read_tail_upper,
        decoder_read_tail_power=decoder_read_tail_power,
        decoder_read_attention_aware=decoder_read_attention_aware,
        decoder_read_responsibility_aware=decoder_read_responsibility_aware,
        decoder_read_update_cap=decoder_read_update_cap,
        decoder_read_layer_start=decoder_read_layer_start,
        decoder_read_layer_strengths=decoder_read_layer_strengths,
        decoder_read_adaptive_strength_power=decoder_read_adaptive_strength_power,
        decoder_read_risk_source=decoder_read_risk_source,
        decoder_read_prototypewise=decoder_read_prototypewise,
        decoder_read_center6_support_alpha=decoder_read_center6_support_alpha,
        decoder_read_center6_support_temperature=decoder_read_center6_support_temperature,
        decoder_read_prototype_responsibility_floor=(
            decoder_read_prototype_responsibility_floor
        ),
        decoder_read_center6_mode_tail_thresholds=(
            decoder_read_center6_mode_tail_thresholds
        ),
        decoder_read_center6_mode_tail_uppers=decoder_read_center6_mode_tail_uppers,
        familiarity_memory_size=familiarity_memory_size,
        familiarity_tokens_per_image=familiarity_tokens_per_image,
        familiarity_min_count=familiarity_min_count,
        familiarity_novelty_floor=familiarity_novelty_floor,
        familiarity_temperature=familiarity_temperature,
        familiarity_novelty_mapping=familiarity_novelty_mapping,
        familiarity_calibration_mode=familiarity_calibration_mode,
        familiarity_calibration_quantile=familiarity_calibration_quantile,
        native_anchor_alpha=native_anchor_alpha,
        guided_distill_weight=guided_distill_weight,
        semantic_coverage_weight=semantic_coverage_weight,
        semantic_coverage_min_role_mass=semantic_coverage_min_role_mass,
        semantic_coverage_variant=semantic_coverage_variant,
        semantic_coverage_roles=semantic_coverage_roles,
        semantic_coverage_margin=semantic_coverage_margin,
        semantic_coverage_min_confidence=semantic_coverage_min_confidence,
        semantic_coverage_max_risk=semantic_coverage_max_risk,
        semantic_coverage_warmup_steps=semantic_coverage_warmup_steps,
        semantic_coverage_ramp_steps=semantic_coverage_ramp_steps,
        context_variant=resolved_context_variant,
        context_transport=context_transport,
        context_adaptive_scale=context_adaptive_scale,
        free_prototypes=free_prototypes,
        context_memory_size=context_memory_size,
        context_candidates_per_image=context_candidates_per_image,
        context_candidates_per_group=context_candidates_per_group,
        context_topk=context_topk,
        context_temperature=context_temperature,
        context_query_chunk_size=context_query_chunk_size,
        context_memory_build_batches=context_memory_build_batches,
        context_key_dim=context_key_dim,
        descriptor_variant=descriptor_variant,
        descriptor_radii=descriptor_radii,
        descriptor_memory_size=descriptor_memory_size,
        descriptor_candidates_per_image=descriptor_candidates_per_image,
        descriptor_candidates_per_group=descriptor_candidates_per_group,
        descriptor_topk=descriptor_topk,
        descriptor_temperature=descriptor_temperature,
        descriptor_query_chunk_size=descriptor_query_chunk_size,
        descriptor_key_dim=descriptor_key_dim,
        descriptor_memory_build_batches=descriptor_memory_build_batches,
        descriptor_source_manifest=descriptor_source_manifest,
        descriptor_read_strength=descriptor_read_strength,
        descriptor_read_tail_threshold=descriptor_read_tail_threshold,
        descriptor_read_tail_upper=descriptor_read_tail_upper,
        descriptor_read_tail_power=descriptor_read_tail_power,
        descriptor_read_layer_start=descriptor_read_layer_start,
        descriptor_read_update_cap=descriptor_read_update_cap,
        descriptor_objectness_floor=descriptor_objectness_floor,
    )


class RobustPriorNormalizer(nn.Module):
    """EMA normal-reference quantiles for fixed-prior response channels."""

    CHANNELS = ("feature_texture", "image_texture", "feature_object", "image_edge")

    def __init__(self, low: float, high: float, momentum: float) -> None:
        super().__init__()
        if not 0.0 <= momentum < 1.0:
            raise ValueError("Robust prior momentum must satisfy 0 <= momentum < 1.")
        self.low_quantile = float(low)
        self.high_quantile = float(high)
        self.momentum = float(momentum)
        self.register_buffer("low", torch.zeros(len(self.CHANNELS), dtype=torch.float32))
        self.register_buffer("high", torch.ones(len(self.CHANNELS), dtype=torch.float32))
        self.register_buffer("initialized", torch.zeros(len(self.CHANNELS), dtype=torch.bool))

    def normalize(self, value: torch.Tensor, channel: str, update: bool) -> torch.Tensor:
        index = self.CHANNELS.index(channel)
        flat = value.detach().float().reshape(-1)
        batch_low = torch.quantile(flat, self.low_quantile)
        batch_high = torch.quantile(flat, self.high_quantile)
        if update:
            with torch.no_grad():
                if not bool(self.initialized[index]):
                    self.low[index].copy_(batch_low)
                    self.high[index].copy_(batch_high)
                    self.initialized[index].fill_(True)
                else:
                    self.low[index].mul_(self.momentum).add_(batch_low, alpha=1.0 - self.momentum)
                    self.high[index].mul_(self.momentum).add_(batch_high, alpha=1.0 - self.momentum)
        low = self.low[index] if bool(self.initialized[index]) else batch_low
        high = self.high[index] if bool(self.initialized[index]) else batch_high
        return ((value - low.to(value)) / (high - low).clamp_min(1e-6).to(value)).clamp(0.0, 1.0)


class PrototypePriorHead(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 0) -> None:
        super().__init__()
        hidden = int(hidden_dim) if hidden_dim > 0 else max(dim // 4, 64)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)


class FamiliarityRiskCalibrator(nn.Module):
    """Monotonic identity-initialized calibration of familiarity write risk."""

    def __init__(self) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        risk: torch.Tensor,
        *,
        detach_parameters: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eps = 1e-6
        clipped = risk.clamp(eps, 1.0 - eps)
        logit = torch.logit(clipped)
        log_scale = self.log_scale.detach() if detach_parameters else self.log_scale
        bias = self.bias.detach() if detach_parameters else self.bias
        calibrated_logit = log_scale.exp() * logit + bias
        return calibrated_logit.sigmoid(), calibrated_logit


class TargetRejectGate(nn.Module):
    """Monotonic residual gate for target rejection.

    Familiarity risk keeps coefficient one.  Three non-negative residual
    coefficients allow mode novelty, objectness, and mode uncertainty to
    reorder tokens while preserving an explicit, inspectable decision rule.
    """

    def __init__(self, initial_residual_weight: float = 0.05) -> None:
        super().__init__()
        if initial_residual_weight <= 0.0:
            raise ValueError("Target-reject residual weights must initialize positive.")
        inverse_softplus = math.log(math.expm1(float(initial_residual_weight)))
        self.raw_weights = nn.Parameter(torch.full((3,), inverse_softplus))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        familiarity_risk: torch.Tensor,
        mode_novelty: torch.Tensor,
        objectness: torch.Tensor,
        mode_uncertainty: torch.Tensor,
        *,
        detach_parameters: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shapes = {
            tuple(familiarity_risk.shape),
            tuple(mode_novelty.shape),
            tuple(objectness.shape),
            tuple(mode_uncertainty.shape),
        }
        if len(shapes) != 1:
            raise ValueError(f"Target-reject features must share one shape, got {shapes}.")
        eps = 1e-6
        base_logit = torch.logit(familiarity_risk.clamp(eps, 1.0 - eps))
        raw_weights = self.raw_weights.detach() if detach_parameters else self.raw_weights
        bias = self.bias.detach() if detach_parameters else self.bias
        weights = F.softplus(raw_weights)
        residual_features = torch.stack(
            (mode_novelty, objectness, mode_uncertainty), dim=-1
        )
        centered = 4.0 * (residual_features.clamp(0.0, 1.0) - 0.5)
        logits = base_logit + (centered * weights).sum(dim=-1) + bias
        return logits.sigmoid(), logits

    def weights(self) -> torch.Tensor:
        return F.softplus(self.raw_weights)


class PerModeTargetGate(nn.Module):
    """Monotonic reject boundary calibrated by pseudo anomalies per effective mode."""

    def __init__(
        self,
        mode_count: int,
        *,
        initial_temperature: float = 0.10,
        initial_boundary: float = 1.0,
        minimum_positive_support: int = 8,
    ) -> None:
        super().__init__()
        if mode_count <= 0:
            raise ValueError("Per-mode target gate requires at least one effective mode.")
        if initial_temperature <= 0.0:
            raise ValueError("Per-mode target-gate temperature must be positive.")
        if minimum_positive_support <= 0:
            raise ValueError("Per-mode target-gate support must be positive.")
        self.mode_count = int(mode_count)
        self.minimum_positive_support = int(minimum_positive_support)
        self.global_log_slope = nn.Parameter(
            torch.tensor(math.log(1.0 / float(initial_temperature)))
        )
        self.global_boundary = nn.Parameter(torch.tensor(float(initial_boundary)))
        self.mode_log_slope_delta = nn.Parameter(torch.zeros(self.mode_count))
        self.mode_boundary_delta = nn.Parameter(torch.zeros(self.mode_count))
        self.register_buffer(
            "positive_support", torch.zeros(self.mode_count, dtype=torch.long)
        )
        self.register_buffer(
            "negative_support", torch.zeros(self.mode_count, dtype=torch.long)
        )

    def forward(
        self,
        distance_ratio: torch.Tensor,
        mode_assignment: torch.Tensor,
        *,
        detach_parameters: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if distance_ratio.shape != mode_assignment.shape:
            raise ValueError("Per-mode target-gate ratios and assignments must match.")
        mode_assignment = mode_assignment.long()
        if bool((mode_assignment < 0).any()) or bool(
            (mode_assignment >= self.mode_count).any()
        ):
            raise ValueError("Per-mode target-gate assignment is out of range.")

        global_log_slope = self.global_log_slope
        global_boundary = self.global_boundary
        log_slope_delta = self.mode_log_slope_delta
        boundary_delta = self.mode_boundary_delta
        if detach_parameters:
            global_log_slope = global_log_slope.detach()
            global_boundary = global_boundary.detach()
            log_slope_delta = log_slope_delta.detach()
            boundary_delta = boundary_delta.detach()

        supported = (
            self.positive_support >= int(self.minimum_positive_support)
        ).to(mode_assignment.device)
        token_supported = supported[mode_assignment]
        token_log_slope = global_log_slope + torch.where(
            token_supported,
            log_slope_delta[mode_assignment],
            torch.zeros_like(distance_ratio),
        )
        token_boundary = global_boundary + torch.where(
            token_supported,
            boundary_delta[mode_assignment],
            torch.zeros_like(distance_ratio),
        )
        slope = token_log_slope.exp().clamp(max=100.0)
        logits = slope * (distance_ratio - token_boundary)
        return logits.sigmoid(), logits

    @torch.no_grad()
    def record_support(
        self,
        mode_assignment: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> None:
        flat_mode = mode_assignment.detach().long().flatten()
        for mask, destination in (
            (positive.detach().bool().flatten(), self.positive_support),
            (negative.detach().bool().flatten(), self.negative_support),
        ):
            counts = torch.bincount(
                flat_mode[mask], minlength=self.mode_count
            ).to(destination)
            destination.add_(counts)

    def slopes(self) -> torch.Tensor:
        return (self.global_log_slope + self.mode_log_slope_delta).exp()

    def boundaries(self) -> torch.Tensor:
        return self.global_boundary + self.mode_boundary_delta


class NormalObjectTokenMemory(nn.Module):
    """Frozen-at-capacity memory used to distinguish familiar normal structure from novel objects."""

    def __init__(
        self,
        dim: int,
        size: int,
        tokens_per_image: int,
        min_count: int,
        novelty_floor: float,
        temperature: float,
        novelty_mapping: str = "sigmoid",
        calibration_mode: str = "legacy",
        calibration_quantile: float = 0.95,
    ) -> None:
        super().__init__()
        self.size = int(size)
        self.tokens_per_image = int(tokens_per_image)
        self.min_count = int(min_count)
        self.novelty_floor = float(novelty_floor)
        self.temperature = float(temperature)
        self.novelty_mapping = str(novelty_mapping)
        if self.novelty_mapping not in {"sigmoid", "hard"}:
            raise ValueError(
                f"Unknown familiarity novelty mapping: {self.novelty_mapping!r}."
            )
        self.calibration_mode = str(calibration_mode)
        self.calibration_quantile = float(calibration_quantile)
        self.register_buffer("bank", torch.zeros(self.size, dim, dtype=torch.float32))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))
        self.register_buffer("bank_group_ids", torch.full((self.size,), -1, dtype=torch.long))
        self.register_buffer("calibrated_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("calibration_group_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("local_radii", torch.ones(self.size, dtype=torch.float32))
        self.register_buffer("local_radius_reference", torch.ones((), dtype=torch.float32))
        self.register_buffer("local_radius_enabled", torch.zeros((), dtype=torch.bool))
        self.register_buffer(
            "novelty_threshold",
            torch.tensor(self.novelty_floor, dtype=torch.float32),
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        optional_keys = {
            f"{prefix}bank_group_ids",
            f"{prefix}calibrated_count",
            f"{prefix}calibration_group_count",
            f"{prefix}local_radii",
            f"{prefix}local_radius_reference",
            f"{prefix}local_radius_enabled",
        }
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        missing_keys[:] = [key for key in missing_keys if key not in optional_keys]

    @property
    def ready(self) -> bool:
        return int(self.count.item()) >= self.min_count

    @torch.no_grad()
    def novelty(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        count = int(self.count.item())
        if count < self.min_count:
            zeros = tokens.new_zeros(tokens.shape[:2])
            return zeros, tokens.new_ones(tokens.shape[:2])

        query = F.normalize(tokens.detach().float(), dim=-1)
        memory = self.bank[:count]
        best_similarity = query.new_full(query.shape[:2], -1.0)
        best_index = torch.zeros(query.shape[:2], dtype=torch.long, device=query.device)
        for start in range(0, count, 256):
            similarity = query @ memory[start : start + 256].T
            chunk_similarity, chunk_index = similarity.max(dim=-1)
            better = chunk_similarity > best_similarity
            best_similarity = torch.where(better, chunk_similarity, best_similarity)
            best_index = torch.where(better, chunk_index + start, best_index)
        raw_distance = (1.0 - best_similarity).clamp(0.0, 2.0)
        if bool(self.local_radius_enabled.item()):
            nearest_radius = self.local_radii[:count][best_index].to(raw_distance)
            reference = self.local_radius_reference.to(raw_distance)
            distance = raw_distance * reference / nearest_radius.clamp_min(1e-8)
        else:
            distance = raw_distance
        threshold = self.novelty_threshold.to(distance)
        if self.novelty_mapping == "hard":
            novelty = (distance >= threshold).to(distance.dtype)
        else:
            novelty = torch.sigmoid((distance - threshold) / self.temperature)
        return novelty.to(tokens.dtype), distance.to(tokens.dtype)

    @torch.no_grad()
    def update(
        self,
        tokens: torch.Tensor,
        objectness: torch.Tensor,
        source_group_ids: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        count = int(self.count.item())
        if (
            self.calibration_mode == "cross_group"
            and count > 0
            and not bool((self.bank_group_ids[:count] >= 0).any())
        ):
            raise RuntimeError(
                "A legacy familiarity memory has no source-group calibration state. "
                "Rebuild the memory from normal training data before using cross_group mode."
            )
        if count >= self.size:
            return
        if self.calibration_mode == "cross_group" and source_group_ids is None:
            raise RuntimeError(
                "Cross-group familiarity calibration requires source group IDs for every normal image."
            )
        if source_group_ids is not None and source_group_ids.shape != (tokens.shape[0],):
            raise ValueError(
                f"Expected source_group_ids shape {(tokens.shape[0],)}, got {tuple(source_group_ids.shape)}."
            )
        per_image = min(self.tokens_per_image, tokens.shape[1])
        ranking = objectness.detach().float()
        if valid_mask is not None:
            if valid_mask.shape != ranking.shape:
                raise ValueError(
                    f"Familiarity valid-mask shape mismatch: "
                    f"{tuple(valid_mask.shape)} != {tuple(ranking.shape)}."
                )
            valid_mask = valid_mask.to(device=ranking.device, dtype=torch.bool)
            if bool((valid_mask.sum(dim=1) < per_image).any()):
                raise ValueError(
                    "Familiarity memory requires at least tokens_per_image valid ROI tokens."
                )
            ranking = ranking.masked_fill(~valid_mask, float("-inf"))
        indices = ranking.topk(per_image, dim=1).indices
        selected = tokens.detach().float().gather(
            1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        )
        selected = F.normalize(selected.reshape(-1, tokens.shape[-1]), dim=-1)
        if source_group_ids is None:
            selected_group_ids = torch.full(
                (selected.shape[0],), -1, dtype=torch.long, device=selected.device
            )
        else:
            selected_group_ids = (
                source_group_ids.detach().to(device=selected.device, dtype=torch.long)
                .view(-1, 1)
                .expand(-1, per_image)
                .reshape(-1)
            )
        take = min(self.size - count, selected.shape[0])
        if take <= 0:
            return
        self.bank[count : count + take].copy_(selected[:take])
        self.bank_group_ids[count : count + take].copy_(selected_group_ids[:take])
        new_count = count + take
        self.count.fill_(new_count)
        needs_first_calibration = (
            new_count >= self.min_count and int(self.calibrated_count.item()) < self.min_count
        )
        if needs_first_calibration or new_count == self.size:
            self._calibrate_threshold()

    @torch.no_grad()
    def _calibrate_threshold(self) -> None:
        count = int(self.count.item())
        if count < self.min_count:
            return
        memory = F.normalize(self.bank[:count], dim=-1)
        similarity = memory @ memory.T
        if self.calibration_mode == "cross_group":
            group_ids = self.bank_group_ids[:count]
            valid = group_ids[:, None] != group_ids[None, :]
            valid = valid & (group_ids[:, None] >= 0) & (group_ids[None, :] >= 0)
            valid_rows = valid.any(dim=1)
            group_count = int(torch.unique(group_ids[group_ids >= 0]).numel())
            self.calibration_group_count.fill_(group_count)
            if not bool(valid_rows.any()):
                if count == self.size:
                    raise RuntimeError(
                        "Cross-group familiarity memory reached capacity without two distinct source groups."
                    )
                return
            similarity = similarity.masked_fill(~valid, -1.0)
            distances = (1.0 - similarity[valid_rows].max(dim=1).values).clamp(0.0, 2.0)
        else:
            similarity.fill_diagonal_(-1.0)
            distances = (1.0 - similarity.max(dim=1).values).clamp(0.0, 2.0)
            self.calibration_group_count.fill_(0)
        calibrated = torch.quantile(distances, self.calibration_quantile)
        self.novelty_threshold.copy_(calibrated.clamp_min(self.novelty_floor))
        self.calibrated_count.fill_(count)


class FrozenMemoryModeTeacher(nn.Module):
    """Fixed normal-mode centers distilled from a P1 memory bank."""

    def __init__(
        self,
        centers: torch.Tensor,
        groups: tuple[int, int, int],
        *,
        group_reliability: torch.Tensor | None = None,
        margin_floor: torch.Tensor | None = None,
        margin_scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if centers.ndim != 2 or centers.shape[0] != sum(groups):
            raise ValueError(
                f"Expected {sum(groups)} mode centers, got {tuple(centers.shape)}."
            )
        self.groups = tuple(int(value) for value in groups)
        self.register_buffer("centers", F.normalize(centers.detach().float(), dim=-1))
        calibration = (group_reliability, margin_floor, margin_scale)
        self.has_confidence_calibration = all(value is not None for value in calibration)
        if any(value is not None for value in calibration) and not self.has_confidence_calibration:
            raise ValueError("Mode confidence calibration requires reliability, floor and scale together.")
        if self.has_confidence_calibration:
            expected = (len(groups),)
            tensors = tuple(value.detach().float() for value in calibration if value is not None)
            if any(tensor.shape != expected for tensor in tensors):
                raise ValueError(f"Mode confidence calibration must have shape {expected}.")
            reliability, floor, scale = tensors
            if (
                not bool(torch.isfinite(reliability).all())
                or not bool(torch.isfinite(floor).all())
                or not bool(torch.isfinite(scale).all())
                or bool((reliability < 0.0).any())
                or bool((reliability > 1.0).any())
                or bool((scale <= floor).any())
            ):
                raise ValueError("Mode confidence calibration contains invalid values.")
        else:
            reliability = torch.ones(len(groups), dtype=torch.float32)
            floor = torch.zeros(len(groups), dtype=torch.float32)
            scale = torch.ones(len(groups), dtype=torch.float32)
        # Keep calibration out of checkpoint state for compatibility with
        # existing P2 checkpoints.  It is deterministic sidecar metadata and is
        # reloaded whenever the teacher is configured.
        self.register_buffer("group_reliability", reliability, persistent=False)
        self.register_buffer("margin_floor", floor, persistent=False)
        self.register_buffer("margin_scale", scale, persistent=False)


class FrozenCenter6Teacher(nn.Module):
    """Frozen Normal modes with six prototype slots and calibrated radii.

    Legacy Center6 uses one mode per prototype slot.  Adaptive Center6 can map
    several slots to one parent mode while keeping the model's six physical
    prototypes and all familiarity-gate geometry unchanged.
    """

    def __init__(
        self,
        centers: torch.Tensor,
        radii: torch.Tensor,
        member_counts: torch.Tensor,
        groups: tuple[int, int, int],
        *,
        mode_groups: tuple[int, int, int] | None = None,
        slot_to_mode: torch.Tensor | None = None,
        group_reliability: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        slot_groups = tuple(int(value) for value in groups)
        mode_groups = (
            slot_groups if mode_groups is None else tuple(int(value) for value in mode_groups)
        )
        if len(slot_groups) != 3 or len(mode_groups) != 3:
            raise ValueError("Center teacher requires three semantic groups.")
        slot_count = sum(slot_groups)
        mode_count = sum(mode_groups)
        if centers.ndim != 2 or centers.shape[0] != mode_count:
            raise ValueError(f"Expected {mode_count} center modes, got {tuple(centers.shape)}.")
        if radii.shape != (mode_count,) or member_counts.shape != (mode_count,):
            raise ValueError("Center radii and member counts must have one value per mode.")
        if not bool(torch.isfinite(radii).all()) or bool((radii <= 0.0).any()):
            raise ValueError("Center6 radii must be finite and positive.")
        if bool((member_counts <= 0).any()):
            raise ValueError("Every Center6 mode must have at least one calibration member.")
        if slot_to_mode is None:
            if slot_count != mode_count:
                raise ValueError("Adaptive center modes require an explicit slot-to-mode mapping.")
            slot_to_mode = torch.arange(slot_count, dtype=torch.long)
        slot_to_mode = slot_to_mode.detach().long().flatten()
        if slot_to_mode.shape != (slot_count,):
            raise ValueError("Center slot-to-mode mapping must contain one entry per prototype.")
        if bool((slot_to_mode < 0).any()) or bool((slot_to_mode >= mode_count).any()):
            raise ValueError("Center slot-to-mode mapping contains an invalid mode index.")
        if torch.unique(slot_to_mode).numel() != mode_count:
            raise ValueError("Every adaptive center mode must own at least one prototype slot.")
        mode_group_ids = torch.repeat_interleave(
            torch.arange(len(mode_groups), dtype=torch.long),
            torch.as_tensor(mode_groups, dtype=torch.long),
        )
        slot_group_ids = torch.repeat_interleave(
            torch.arange(len(slot_groups), dtype=torch.long),
            torch.as_tensor(slot_groups, dtype=torch.long),
        )
        if not torch.equal(mode_group_ids[slot_to_mode], slot_group_ids):
            raise ValueError("Prototype slots may only map to modes in their semantic group.")
        reliability_provided = group_reliability is not None
        if group_reliability is None:
            group_reliability = torch.ones(len(mode_groups), dtype=torch.float32)
        group_reliability = group_reliability.detach().float().flatten()
        if group_reliability.shape != (len(mode_groups),):
            raise ValueError("Center group reliability must contain one value per semantic group.")
        if (
            not bool(torch.isfinite(group_reliability).all())
            or bool((group_reliability < 0.0).any())
            or bool((group_reliability > 1.0).any())
        ):
            raise ValueError("Center group reliability must be finite and in [0,1].")
        self.groups = slot_groups
        self.mode_groups = mode_groups
        self.group_reliability_provided = reliability_provided
        self.register_buffer("centers", F.normalize(centers.detach().float(), dim=-1))
        self.register_buffer("radii", radii.detach().float())
        self.register_buffer("member_counts", member_counts.detach().long())
        # Deterministic sidecar metadata.  Keeping it non-persistent preserves
        # strict compatibility with legacy Center6 checkpoints.
        self.register_buffer("slot_to_mode", slot_to_mode, persistent=False)
        self.register_buffer("mode_group_ids", mode_group_ids, persistent=False)
        self.register_buffer("group_reliability", group_reliability, persistent=False)


def center6_mode_normalized_features(
    tokens: torch.Tensor,
    teacher: FrozenCenter6Teacher,
    *,
    temperature: float,
    return_assignment: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Return nearest radius-normalized novelty, distance ratio, and uncertainty."""

    if tokens.ndim != 3 or tokens.shape[-1] != teacher.centers.shape[-1]:
        raise ValueError("Center6 mode features require [B,N,C] tokens matching teacher C.")
    if temperature <= 0.0:
        raise ValueError("Center6 mode feature temperature must be positive.")
    normalized = F.normalize(tokens.float(), dim=-1)
    distances = (1.0 - normalized @ teacher.centers.to(normalized).T).clamp(0.0, 2.0)
    ratios = distances / teacher.radii.to(distances).view(1, 1, -1).clamp_min(1e-6)
    nearest_ratio, nearest_mode = ratios.min(dim=-1)
    novelty = torch.sigmoid((nearest_ratio - 1.0) / float(temperature))
    responsibilities = torch.softmax(-ratios / float(temperature), dim=-1)
    entropy = -(responsibilities * responsibilities.clamp_min(1e-8).log()).sum(dim=-1)
    if responsibilities.shape[-1] > 1:
        entropy = entropy / math.log(float(responsibilities.shape[-1]))
    else:
        entropy = torch.zeros_like(entropy)
    outputs = (
        novelty.to(tokens.dtype),
        nearest_ratio.to(tokens.dtype),
        entropy.clamp(0.0, 1.0).to(tokens.dtype),
    )
    if return_assignment:
        return (*outputs, nearest_mode)
    return outputs


def _deterministic_cosine_modes(features: torch.Tensor, count: int) -> torch.Tensor:
    """Deterministic spherical k-means used only to construct the frozen teacher."""

    features = F.normalize(features.detach().float().cpu(), dim=-1)
    if features.ndim != 2 or features.shape[0] < count:
        raise ValueError(
            f"Need at least {count} memory tokens to construct modes, got {tuple(features.shape)}."
        )
    mean = F.normalize(features.mean(dim=0, keepdim=True), dim=-1)
    first = (features @ mean.T).squeeze(1).argmin()
    selected = [first]
    best_similarity = features @ features[first]
    chosen = torch.zeros(features.shape[0], dtype=torch.bool)
    chosen[first] = True
    for _ in range(1, int(count)):
        next_index = best_similarity.masked_fill(chosen, float("inf")).argmin()
        selected.append(next_index)
        chosen[next_index] = True
        best_similarity = torch.maximum(best_similarity, features @ features[next_index])
    centers = features[torch.stack(selected)].clone()
    for _ in range(25):
        assignment = (features @ centers.T).argmax(dim=1)
        updated = centers.clone()
        for index in range(int(count)):
            members = features[assignment == index]
            if members.numel():
                updated[index] = F.normalize(members.mean(dim=0), dim=0)
        if torch.allclose(updated, centers, atol=1e-7, rtol=0.0):
            break
        centers = updated
    return F.normalize(centers, dim=-1)


def _calibrate_center6_radii(
    bank: torch.Tensor,
    semantic_group_ids: torch.Tensor,
    centers: torch.Tensor,
    groups: tuple[int, int, int],
    *,
    quantile: float,
    radius_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate each center radius from its own within-semantic-mode members."""

    bank = F.normalize(bank.detach().float().cpu(), dim=-1)
    semantic_group_ids = semantic_group_ids.detach().long().cpu()
    centers = F.normalize(centers.detach().float().cpu(), dim=-1)
    if bank.ndim != 2 or semantic_group_ids.shape != (bank.shape[0],):
        raise ValueError("Center6 bank and semantic_group_ids have incompatible shapes.")
    if centers.shape != (sum(groups), bank.shape[1]):
        raise ValueError("Center6 centers and bank have incompatible shapes.")

    radii = []
    member_counts = []
    for group_index, group_slice in enumerate(_group_slices(groups)):
        group_bank = bank[semantic_group_ids == group_index]
        if group_bank.shape[0] < int(group_slice.stop - group_slice.start):
            raise ValueError(
                f"Center6 teacher group {group_index} has too few radius-calibration tokens."
            )
        group_distance = 1.0 - group_bank @ centers[group_slice].T
        assignment = group_distance.argmin(dim=1)
        for local_mode in range(int(group_slice.stop - group_slice.start)):
            mode_distance = group_distance[assignment == local_mode, local_mode]
            if mode_distance.numel() == 0:
                raise ValueError(
                    f"Center6 teacher group {group_index} mode {local_mode} has no calibration members."
                )
            radii.append(
                torch.quantile(mode_distance, float(quantile)).clamp_min(float(radius_floor))
            )
            member_counts.append(mode_distance.new_tensor(mode_distance.numel(), dtype=torch.long))
    return torch.stack(radii), torch.stack(member_counts)


def load_frozen_memory_mode_teacher(
    path: str | Path,
    groups: tuple[int, int, int],
) -> FrozenMemoryModeTeacher:
    """Load P1 bank features and form one fixed mode center per prototype."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"P2 memory-mode teacher does not exist: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if "mode_teacher_centers" in payload.files:
            required_stable = {
                "mode_teacher_centers",
                "mode_teacher_groups",
                "mode_teacher_group_reliability",
                "mode_teacher_margin_floor",
                "mode_teacher_margin_scale",
            }
            missing = required_stable.difference(payload.files)
            if missing:
                raise ValueError(
                    f"Stable P2 teacher {path} is missing arrays: {sorted(missing)}"
                )
            stored_groups = tuple(
                int(value)
                for value in np.asarray(payload["mode_teacher_groups"], dtype=np.int64).tolist()
            )
            if stored_groups != tuple(groups):
                raise ValueError(
                    f"Stable P2 teacher groups {stored_groups} do not match requested groups {groups}."
                )
            centers = torch.from_numpy(
                np.asarray(payload["mode_teacher_centers"], dtype=np.float32)
            )
            return FrozenMemoryModeTeacher(
                centers,
                groups,
                group_reliability=torch.from_numpy(
                    np.asarray(payload["mode_teacher_group_reliability"], dtype=np.float32)
                ),
                margin_floor=torch.from_numpy(
                    np.asarray(payload["mode_teacher_margin_floor"], dtype=np.float32)
                ),
                margin_scale=torch.from_numpy(
                    np.asarray(payload["mode_teacher_margin_scale"], dtype=np.float32)
                ),
            )
        required = {"bank", "semantic_group_ids"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"P2 teacher {path} is missing arrays: {sorted(missing)}")
        bank = torch.from_numpy(np.asarray(payload["bank"], dtype=np.float32))
        semantic_group_ids = torch.from_numpy(
            np.asarray(payload["semantic_group_ids"], dtype=np.int64)
        )
    if bank.ndim != 2 or semantic_group_ids.shape != (bank.shape[0],):
        raise ValueError("P2 teacher bank and semantic_group_ids have incompatible shapes.")
    centers = []
    for group_index, mode_count in enumerate(groups):
        group_features = bank[semantic_group_ids == group_index]
        if group_features.shape[0] < mode_count:
            raise ValueError(
                f"P2 teacher group {group_index} has {group_features.shape[0]} tokens, "
                f"fewer than its {mode_count} prototypes."
            )
        centers.append(_deterministic_cosine_modes(group_features, mode_count))
    return FrozenMemoryModeTeacher(torch.cat(centers, dim=0), groups)


def load_frozen_center6_teacher(
    path: str | Path,
    groups: tuple[int, int, int],
    *,
    radius_quantile: float = 0.95,
    radius_floor: float = 1e-3,
) -> FrozenCenter6Teacher:
    """Load the six P1 modes and calibrate a radius for every mode.

    The radius population is intentionally restricted to the center's original
    semantic group.  The resulting radius-normalized prior is nevertheless a
    full six-way distribution for every training token.
    """

    path = Path(path)
    precomputed_radii = None
    precomputed_member_counts = None
    with np.load(path, allow_pickle=False) as payload:
        required = {"bank", "semantic_group_ids"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(
                f"Center6 teacher {path} needs the P1 bank for radius calibration; "
                f"missing arrays: {sorted(missing)}"
            )
        bank = torch.from_numpy(np.asarray(payload["bank"], dtype=np.float32))
        semantic_group_ids = torch.from_numpy(
            np.asarray(payload["semantic_group_ids"], dtype=np.int64)
        )
        adaptive_keys = {
            "adaptive_mode_centers",
            "adaptive_mode_groups",
            "adaptive_slot_groups",
            "adaptive_slot_to_mode",
        }
        has_adaptive = bool(adaptive_keys.issubset(payload.files))
        if has_adaptive:
            stored_slot_groups = tuple(
                int(value)
                for value in np.asarray(payload["adaptive_slot_groups"], dtype=np.int64).tolist()
            )
            if stored_slot_groups != tuple(groups):
                raise ValueError(
                    f"Adaptive center slot groups {stored_slot_groups} do not match requested "
                    f"groups {tuple(groups)}."
                )
            mode_groups = tuple(
                int(value)
                for value in np.asarray(payload["adaptive_mode_groups"], dtype=np.int64).tolist()
            )
            centers = torch.from_numpy(
                np.asarray(payload["adaptive_mode_centers"], dtype=np.float32)
            )
            slot_to_mode = torch.from_numpy(
                np.asarray(payload["adaptive_slot_to_mode"], dtype=np.int64)
            )
            group_reliability = (
                torch.from_numpy(
                    np.asarray(payload["adaptive_group_reliability"], dtype=np.float32)
                )
                if "adaptive_group_reliability" in payload.files
                else None
            )
            radius_keys = {
                "adaptive_mode_radii",
                "adaptive_mode_radius_member_counts",
            }
            present_radius_keys = radius_keys.intersection(payload.files)
            if present_radius_keys and present_radius_keys != radius_keys:
                raise ValueError(
                    "Adaptive center teacher has an incomplete full-pool radius calibration."
                )
            if present_radius_keys:
                precomputed_radii = torch.from_numpy(
                    np.asarray(payload["adaptive_mode_radii"], dtype=np.float32)
                )
                precomputed_member_counts = torch.from_numpy(
                    np.asarray(
                        payload["adaptive_mode_radius_member_counts"], dtype=np.int64
                    )
                )
        else:
            mode_groups = tuple(groups)
            centers = None
            slot_to_mode = None
            group_reliability = None
    if centers is None:
        mode_teacher = load_frozen_memory_mode_teacher(path, groups)
        centers = mode_teacher.centers
    if precomputed_radii is None:
        radii, member_counts = _calibrate_center6_radii(
            bank,
            semantic_group_ids,
            centers,
            mode_groups,
            quantile=radius_quantile,
            radius_floor=radius_floor,
        )
    else:
        radii = precomputed_radii
        member_counts = precomputed_member_counts
    return FrozenCenter6Teacher(
        centers,
        radii,
        member_counts,
        groups,
        mode_groups=mode_groups,
        slot_to_mode=slot_to_mode,
        group_reliability=group_reliability,
    )


def configure_guided_prototypes(model: torch.nn.Module, args, architecture: str) -> None:
    from .context_normal_prototype import configure_context_normal_prototype

    configure_context_normal_prototype(model, args, architecture)
    context_requested = bool(
        getattr(args, "guided_prototype_context_transport", False)
        or getattr(args, "guided_prototype_context_adaptive_scale", False)
        or getattr(args, "guided_prototype_free_prototypes", False)
        or getattr(args, "guided_prototype_context_variant", "off") != "off"
        or getattr(args, "guided_prototype_descriptor_variant", "off") != "off"
    )
    hard_roi_attention = bool(
        getattr(args, "prototype_hard_roi_attention", False)
    )
    transition_roi_attention = bool(
        getattr(args, "prototype_transition_roi_attention", False)
    )
    if hard_roi_attention and transition_roi_attention:
        raise ValueError(
            "Select only one of hard or transition prototype ROI attention."
        )
    if (hard_roi_attention or transition_roi_attention) and architecture != "inpformer":
        raise ValueError(
            "Prototype ROI attention is only implemented for --architecture inpformer."
        )
    transition_roi_floor = float(
        getattr(args, "prototype_transition_roi_floor", 1e-6)
    )
    if not 0.0 < transition_roi_floor <= 1.0:
        raise ValueError("Transition ROI floor must be in (0,1].")
    model._prototype_hard_roi_attention = hard_roi_attention
    model._prototype_transition_roi_attention = transition_roi_attention
    model._prototype_transition_roi_floor = transition_roi_floor
    model._guided_roi_aware_loss = bool(
        getattr(args, "guided_prototype_roi_aware_loss", False)
    )
    if not getattr(args, "guided_prototype", False):
        if model._guided_roi_aware_loss:
            raise ValueError(
                "--guided-prototype-roi-aware-loss requires --guided-prototype."
            )
        if context_requested:
            raise ValueError("Context-familiarity options require --guided-prototype.")
        if hard_roi_attention or transition_roi_attention:
            _enable_roi_aware_inpformer_forward(model)
        return
    if architecture != "inpformer":
        raise ValueError("--guided-prototype is only implemented for --architecture inpformer.")
    config = guided_config_from_args(args)
    historical_compatible_raw = bool(
        getattr(args, "rgpl_historical_compatible_raw", False)
    )
    if historical_compatible_raw:
        if hard_roi_attention or transition_roi_attention:
            raise ValueError(
                "Historical-compatible RGPL Raw cannot use hard/transition ROI attention."
            )
        required = {
            "center6_balanced": config.center6_balanced,
            "aggregation_gate": config.aggregation_gate,
            "aggregation_familiarity_gate": config.aggregation_familiarity_gate,
            "familiarity_write_enabled": config.familiarity_write_enabled,
        }
        disabled = [name for name, enabled in required.items() if not enabled]
        if disabled:
            raise ValueError(
                "Historical-compatible RGPL Raw requires: " + ", ".join(disabled)
            )
        incompatible = {
            "mode_routing": config.mode_specific_routing,
            "decoder_read": config.decoder_read_gate,
            "target_gate": config.target_gate_calibration,
            "target_reject": config.target_reject_gate,
            "target_mode": config.target_mode_gate,
            "context_transport": config.context_transport,
            "descriptor": config.descriptor_variant != "off",
            "guided_distill": config.guided_distill_weight != 0.0,
            "semantic_coverage": config.semantic_coverage_weight != 0.0,
        }
        active = [name for name, enabled in incompatible.items() if enabled]
        if active:
            raise ValueError(
                "Historical-compatible RGPL Raw rejects extra branches: "
                + ", ".join(active)
            )
        if config.groups != (2, 2, 2):
            raise ValueError("Historical-compatible RGPL Raw requires groups 2,2,2.")
        if config.familiarity_memory_size != 176 or config.familiarity_min_count != 176:
            raise ValueError("Historical-compatible RGPL Raw requires memory/min-count 176.")
        if config.familiarity_tokens_per_image != 4:
            raise ValueError("Historical-compatible RGPL Raw requires four tokens per image.")
        if config.native_anchor_alpha != 1.0:
            raise ValueError("Historical-compatible RGPL Raw requires native alpha 1.0.")
        if config.aggregation_alpha not in (None, 1.0):
            raise ValueError("Historical-compatible RGPL Raw requires aggregation alpha 1.0.")
        if config.aggregation_min_weight != 0.05 or config.aggregation_power != 1.0:
            raise ValueError(
                "Historical-compatible RGPL Raw requires gate floor 0.05 and power 1.0."
            )
    model._rgpl_historical_compatible_raw = historical_compatible_raw
    if config.context_transport and bool(getattr(args, "masked_recon", False)):
        raise ValueError(
            "Context-familiarity transport cannot yet be combined with masked reconstruction: "
            "the mask-specific prototype context has no defined transport semantics."
        )
    prototype_token = getattr(model, "prototype_token", None)
    if prototype_token is None:
        raise ValueError("INP-Former model does not expose prototype_token.")
    prototype_count = int(prototype_token.shape[0])
    if not config.free_prototypes and sum(config.groups) != prototype_count:
        raise ValueError(
            f"Guided prototype groups {config.groups} sum to {sum(config.groups)}, "
            f"but model has {prototype_count} prototype tokens."
        )
    if config.group_contrastive and config.trainable_prior:
        raise ValueError("Group-contrastive guidance currently requires a fixed prior.")
    if config.vlm_prior_checkpoint and config.trainable_prior:
        raise ValueError("The frozen VLM semantic prior cannot be combined with the legacy trainable prior.")
    if config.aggregation_gate and config.trainable_prior:
        raise ValueError("Aggregation gating currently requires a fixed prior.")
    if config.aggregation_familiarity_gate and not config.aggregation_gate:
        raise ValueError("Familiarity gating requires --guided-prototype-aggregation-gate.")
    if config.mode_specific_routing and not config.center6_balanced:
        raise ValueError("Mode-specific prototype routing requires Center6 guidance.")
    if config.mode_specific_routing and not config.aggregation_gate:
        raise ValueError(
            "Mode-specific prototype routing must be layered on an explicit "
            "prototype aggregation gate."
        )
    if config.mode_specific_routing and config.descriptor_variant != "off":
        raise ValueError(
            "Mode-specific prototype routing is defined for the Center6 aggregation "
            "path, not descriptor B/C/D."
        )
    if (
        config.target_gate_calibration
        and not config.aggregation_familiarity_gate
    ):
        raise ValueError(
            "Target-gate calibration requires familiarity risk. It may be "
            "combined with familiarity read-only for a write-gate ablation."
        )
    enabled_target_gates = sum(
        int(enabled)
        for enabled in (
            config.target_gate_calibration,
            config.target_reject_gate,
            config.target_mode_gate,
        )
    )
    if enabled_target_gates > 1:
        raise ValueError(
            "Select exactly one target-gate implementation."
        )
    if (config.target_reject_gate or config.target_mode_gate) and not (
        config.aggregation_familiarity_gate and config.familiarity_write_enabled
    ):
        raise ValueError(
            "Target reject/mode gate requires the familiarity aggregation write gate."
        )
    if (
        config.target_gate_risk_source == "mode_normalized"
        or config.target_reject_gate
        or config.target_mode_gate
    ) and not config.center6_balanced:
        raise ValueError("Mode-normalized target gating requires Center6 guidance.")
    if (
        config.aggregation_alpha is not None
        and not config.aggregation_gate
        and config.descriptor_variant == "off"
    ):
        raise ValueError(
            "An explicit aggregation alpha requires the legacy aggregation gate or descriptor B/C/D."
        )
    if (
        not config.familiarity_write_enabled
        and not config.aggregation_familiarity_gate
        and not config.decoder_read_gate
    ):
        raise ValueError(
            "Familiarity read-only mode requires an aggregation or decoder familiarity gate."
        )
    if config.free_prototypes and config.group_contrastive:
        raise ValueError("Free prototypes cannot use the group-contrastive gather objective.")
    if config.semantic_coverage_weight > 0.0:
        if not config.center6_balanced:
            raise ValueError(
                "Semantic prototype coverage requires the adaptive Center6 normal-role teacher."
            )
        if not config.aggregation_familiarity_gate:
            raise ValueError(
                "Risk-aware semantic prototype coverage requires the familiarity gate."
            )
        if config.native_anchor_alpha != 0.0 or config.guided_distill_weight != 0.0:
            raise ValueError(
                "Semantic prototype coverage is an auxiliary term on full native coherence; "
                "set native-anchor-alpha and guided-distill-weight to zero."
            )
        if config.mode_specific_routing:
            raise ValueError(
                "Semantic prototype coverage uses dynamic set matching and cannot be combined "
                "with fixed mode-to-slot routing."
            )
        if config.aggregation_alpha is None:
            raise ValueError(
                "Semantic prototype coverage requires an explicit aggregation alpha so its "
                "loss does not implicitly change the risk gate."
            )
    teacher_path = getattr(args, "guided_prototype_memory_mode_teacher", None)
    center6_teacher_path = getattr(args, "guided_prototype_center6_teacher", None)
    if config.center6_balanced:
        if center6_teacher_path is None:
            raise ValueError(
                "Center6 balanced supervision requires --guided-prototype-center6-teacher."
            )
        if len(config.groups) != 3 or any(int(count) <= 0 for count in config.groups):
            raise ValueError(
                "Center-balanced supervision requires three non-empty semantic "
                "prototype groups."
            )
        if config.group_contrastive or config.memory_mode_weight > 0.0 or teacher_path is not None:
            raise ValueError(
                "Center6 replaces group-contrastive and memory-mode objectives; do not combine them."
            )
        read_only_physical_prior = bool(
            config.decoder_read_gate
            and config.decoder_read_risk_source in {"physical", "center6_hybrid"}
        )
        if (
            config.trainable_prior
            or config.scale_consistent
            or (config.multiscale_direct and not read_only_physical_prior)
        ):
            raise ValueError(
                "Center6 replaces the old semantic prior; a multiscale physical prior is "
                "allowed only as a decoder-read-only risk source."
            )
        if config.decoder_read_risk_source == "center6_hybrid" and not config.decoder_read_gate:
            raise ValueError("Center6 hybrid risk requires the decoder read gate.")
        if config.context_transport:
            raise ValueError("Center6 does not support context transport in this ablation.")
        if (
            config.separation_weight != 0.0
            or config.intra_group_repulsion_weight != 0.0
            or config.intra_group_balance_weight != 0.0
        ):
            raise ValueError(
                "The Center6 clean ablation requires every explicit balance/separation weight to be zero."
            )
        if config.native_anchor_alpha != 1.0 and config.aggregation_alpha is None:
            raise ValueError(
                "Center6 guided/coherence mixing requires an explicit aggregation-alpha "
                "so the loss mixture does not also change prototype aggregation."
            )
        if config.guided_distill_weight > 0.0 and not config.center6_balanced:
            raise ValueError("Guided distillation requires Center6 supervision.")
    elif center6_teacher_path is not None:
        raise ValueError(
            "A Center6 teacher was provided but --guided-prototype-center6-balanced is disabled."
        )
    if config.memory_mode_weight > 0.0:
        if teacher_path is None:
            raise ValueError(
                "P2 memory-mode supervision requires --guided-prototype-memory-mode-teacher."
            )
        if not config.group_contrastive or config.free_prototypes or config.trainable_prior:
            raise ValueError(
                "P2 memory-mode supervision requires fixed-prior group-contrastive guidance."
            )
    elif teacher_path is not None:
        raise ValueError(
            "A P2 memory-mode teacher was provided but --guided-prototype-memory-mode-weight is zero."
        )
    if not hasattr(model, "_original_gather_loss"):
        model._original_gather_loss = model.gather_loss
    model._guided_prototype_config = config
    model._guided_prototype_enabled = True
    model._guided_prototype_diag = {}
    model._guided_last_prior_state = None
    if config.vlm_prior_checkpoint:
        from .vlm_semantic_prior import load_vlm_prior_checkpoint

        sidecar = load_vlm_prior_checkpoint(
            config.vlm_prior_checkpoint,
            prototype_token.device,
        )
        sidecar_dim = getattr(sidecar.head, "dim", None)
        if sidecar_dim is not None and int(sidecar_dim) != int(prototype_token.shape[-1]):
            raise ValueError(
                f"VLM prior dimension {sidecar_dim} does not match token dimension "
                f"{int(prototype_token.shape[-1])}."
            )
        # The sidecar has its own checkpoint lineage and must not alter strict
        # loading/saving of reconstruction checkpoints.
        object.__setattr__(model, "_guided_vlm_prior_sidecar", sidecar)
    if (
        config.multiscale_direct
        and (not config.free_prototypes or config.descriptor_variant == "d")
        and not hasattr(model, "guided_prior_normalizer")
    ):
        model.guided_prior_normalizer = RobustPriorNormalizer(
            config.robust_quantile_low,
            config.robust_quantile_high,
            config.robust_quantile_momentum,
        ).to(prototype_token.device)
    if config.trainable_prior and not hasattr(model, "guided_prior_head"):
        dim = int(prototype_token.shape[-1])
        model.guided_prior_head = PrototypePriorHead(dim, config.prior_hidden_dim).to(prototype_token.device)
    if config.center6_balanced and not hasattr(model, "guided_center6_teacher"):
        teacher = load_frozen_center6_teacher(
            center6_teacher_path,
            config.groups,
            radius_quantile=config.center6_radius_quantile,
            radius_floor=config.center6_radius_floor,
        )
        if teacher.centers.shape[-1] != int(prototype_token.shape[-1]):
            raise ValueError(
                f"Center6 teacher dimension {teacher.centers.shape[-1]} does not match "
                f"prototype dimension {int(prototype_token.shape[-1])}."
            )
        model.guided_center6_teacher = teacher.to(prototype_token.device)
        if (
            config.center6_hierarchical_reliability
            and not teacher.group_reliability_provided
        ):
            raise ValueError(
                "Hierarchical Center6 supervision requires adaptive_group_reliability "
                "in the teacher sidecar."
            )
    if config.memory_mode_weight > 0.0 and not hasattr(model, "guided_memory_mode_teacher"):
        teacher = load_frozen_memory_mode_teacher(teacher_path, config.groups)
        if teacher.centers.shape[-1] != int(prototype_token.shape[-1]):
            raise ValueError(
                f"P2 teacher dimension {teacher.centers.shape[-1]} does not match "
                f"prototype dimension {int(prototype_token.shape[-1])}."
            )
        if config.memory_mode_margin_weighting and not teacher.has_confidence_calibration:
            raise ValueError(
                "Mode-margin weighting requires a stable_mode_teacher.npz with "
                "Normal-only reliability and margin calibration."
            )
        model.guided_memory_mode_teacher = teacher.to(prototype_token.device)
    if (
        config.aggregation_familiarity_gate or config.decoder_read_gate
    ) and not hasattr(model, "guided_normal_object_memory"):
        model.guided_normal_object_memory = NormalObjectTokenMemory(
            dim=int(prototype_token.shape[-1]),
            size=config.familiarity_memory_size,
            tokens_per_image=config.familiarity_tokens_per_image,
            min_count=config.familiarity_min_count,
            novelty_floor=config.familiarity_novelty_floor,
            temperature=config.familiarity_temperature,
            novelty_mapping=config.familiarity_novelty_mapping,
            calibration_mode=config.familiarity_calibration_mode,
            calibration_quantile=config.familiarity_calibration_quantile,
        ).to(prototype_token.device)
    if config.target_gate_calibration and not hasattr(
        model, "guided_target_gate_calibrator"
    ):
        model.guided_target_gate_calibrator = FamiliarityRiskCalibrator().to(
            prototype_token.device
        )
    if config.target_reject_gate and not hasattr(model, "guided_target_reject_gate"):
        model.guided_target_reject_gate = TargetRejectGate().to(prototype_token.device)
    if config.target_mode_gate and not hasattr(model, "guided_target_mode_gate"):
        teacher = getattr(model, "guided_center6_teacher", None)
        if not isinstance(teacher, FrozenCenter6Teacher):
            raise RuntimeError("Per-mode target gate requires a configured Center6 teacher.")
        model.guided_target_mode_gate = PerModeTargetGate(
            int(teacher.centers.shape[0]),
            initial_temperature=config.target_mode_novelty_temperature,
            minimum_positive_support=config.target_mode_gate_min_support,
        ).to(prototype_token.device)
    if config.context_transport and not hasattr(model, "guided_context_transport_memory"):
        from .context_familiarity_transport import ContextFamiliarityTransportMemory

        configured_size = (
            getattr(args, "model_input_size", 0)
            or getattr(args, "patch_output_size", 0)
            or getattr(args, "image_size", 0)
            or getattr(args, "crop_size", 0)
        )
        input_size = int(configured_size or config.image_reference_size or 448)
        token_side = max(1, input_size // 14)
        kernels, _, _ = resolve_spatial_prior_parameters(config, token_side)
        radii = tuple((kernel - 1) // 2 for kernel in kernels)
        mode_ids = {
            "off": 0.0,
            "custom": 1.0,
            "a1": 2.0,
            "a2": 3.0,
            "a3": 4.0,
            "a2_safe": 5.0,
        }
        mode_signature = (
            1.0,
            mode_ids[config.context_variant],
            float(config.context_adaptive_scale),
            float(config.free_prototypes),
            float(config.multiscale_direct),
            float(config.spatial_scale_mode == "integer"),
            float(config.spatial_reference_side),
            float(config.scale_consistent),
            float(config.prior_reference_side),
            float(config.image_reference_size),
            *tuple(float(value) for value in config.groups),
            *tuple(float(value) for value in config.group_weights),
            *tuple(float(value) for value in config.object_kernel_weights),
            *tuple(float(value) for value in config.texture_offsets),
            *tuple(float(value) for value in config.texture_offset_weights),
            float(config.prior_floor),
            float(config.texture_power),
            float(config.object_power),
            float(config.feature_texture_weight),
            float(config.image_texture_weight),
            float(config.feature_object_weight),
            float(config.image_object_weight),
            float(config.object_texture_suppress),
            float(config.group_contrastive),
        )
        model.guided_context_transport_memory = ContextFamiliarityTransportMemory(
            dim=int(prototype_token.shape[-1]),
            radii=radii,
            size=config.context_memory_size,
            topk=config.context_topk,
            temperature=config.context_temperature,
            query_chunk_size=config.context_query_chunk_size,
            key_dim=config.context_key_dim,
            mode_signature=mode_signature,
        ).to(prototype_token.device)
    if config.descriptor_variant != "off" and not hasattr(
        model, "guided_normal_descriptor_memory"
    ):
        from .normal_descriptor_gate import NormalDescriptorMemory

        descriptor_ids = {"b": 1.0, "c": 2.0, "d": 3.0}
        mode_signature = (
            1.0,
            descriptor_ids[config.descriptor_variant],
            float(config.descriptor_memory_size),
            float(config.descriptor_topk),
            float(config.descriptor_temperature),
            float(config.descriptor_key_dim),
            *tuple(float(radius) for radius in config.descriptor_radii),
        )
        model.guided_normal_descriptor_memory = NormalDescriptorMemory(
            dim=int(prototype_token.shape[-1]),
            radii=config.descriptor_radii,
            size=config.descriptor_memory_size,
            topk=config.descriptor_topk,
            temperature=config.descriptor_temperature,
            query_chunk_size=config.descriptor_query_chunk_size,
            key_dim=config.descriptor_key_dim,
            mode_signature=mode_signature,
        ).to(prototype_token.device)
    if (
        config.aggregation_gate
        or config.context_transport
        or config.decoder_read_gate
        or config.descriptor_variant != "off"
    ) and not hasattr(model, "_original_guided_forward"):
        model._original_guided_forward = model.forward
        model.forward = types.MethodType(_guided_inpformer_forward, model)
        model.aggregate_guided_prototypes = types.MethodType(_aggregate_guided_prototypes, model)
    model.gather_loss = types.MethodType(_guided_gather_loss, model)


def set_guided_prototype_image(
    model: torch.nn.Module,
    images: torch.Tensor | None,
    update_prior_stats: bool | None = None,
) -> None:
    if getattr(model, "_guided_prototype_enabled", False):
        model._guided_prototype_image = None if images is None else images.detach()
        if update_prior_stats is not None:
            model._guided_update_prior_stats = bool(update_prior_stats)


def set_guided_prototype_source_groups(
    model: torch.nn.Module,
    source_group_ids: torch.Tensor | None,
) -> None:
    if getattr(model, "_guided_prototype_enabled", False):
        model._guided_source_group_ids = (
            None if source_group_ids is None else source_group_ids.detach()
        )


def set_guided_prototype_valid_roi(
    model: torch.nn.Module,
    valid_roi_mask: torch.Tensor | None,
) -> None:
    """Set per-image spatial support for Normal prototype aggregation/memory."""

    model._guided_valid_roi_mask = (
        None if valid_roi_mask is None else valid_roi_mask.detach()
    )


def set_guided_prototype_alpha(model: torch.nn.Module, alpha: float) -> None:
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Guided prototype alpha must be in [0,1], got {alpha}.")
    if getattr(model, "_guided_prototype_enabled", False):
        model._guided_runtime_alpha = alpha


def _guided_alpha(model: torch.nn.Module, config: GuidedPrototypeConfig) -> float:
    return float(getattr(model, "_guided_runtime_alpha", config.native_anchor_alpha))


def _guided_aggregation_alpha(
    model: torch.nn.Module, config: GuidedPrototypeConfig
) -> float:
    if config.aggregation_alpha is not None:
        return float(config.aggregation_alpha)
    return _guided_alpha(model, config)


def _familiarity_memory_update_enabled(
    config: GuidedPrototypeConfig,
    *,
    training: bool,
    update_prior_stats: bool,
) -> bool:
    """Keep Normal-memory construction independent of aggregation writing."""

    return bool(
        config.aggregation_familiarity_gate
        and training
        and update_prior_stats
    )


def _descriptor_write_weights(
    risk: torch.Tensor,
    *,
    minimum: float,
    power: float,
    alpha: float,
) -> torch.Tensor:
    """Convert descriptor pollution risk to a Native-anchored write weight."""

    if not 0.0 < float(minimum) <= 1.0:
        raise ValueError("Descriptor write minimum must be in (0,1].")
    if float(power) <= 0.0:
        raise ValueError("Descriptor write power must be positive.")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("Descriptor write alpha must be in [0,1].")
    guided = float(minimum) + (1.0 - float(minimum)) * (
        1.0 - risk.clamp(0.0, 1.0)
    ).pow(float(power))
    return 1.0 + float(alpha) * (guided - 1.0)


def _descriptor_aux_objectness(
    risk: torch.Tensor,
    objectness: torch.Tensor,
    *,
    floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """D-only objectness modulation without assigning prototype identities."""

    if risk.shape != objectness.shape:
        raise ValueError("Descriptor risk and auxiliary objectness shapes disagree.")
    if not 0.0 <= float(floor) <= 1.0:
        raise ValueError("Descriptor objectness floor must be in [0,1].")
    factor = float(floor) + (1.0 - float(floor)) * objectness.clamp(0.0, 1.0)
    return risk * factor, factor


def get_guided_prototype_diag(model: torch.nn.Module) -> dict[str, float]:
    return dict(getattr(model, "_guided_prototype_diag", {}) or {})


def guided_prototype_trainable_modules(model: torch.nn.Module) -> list[nn.Module]:
    modules = []
    for name in (
        "guided_prior_head",
        "guided_target_gate_calibrator",
        "guided_target_reject_gate",
        "guided_target_mode_gate",
    ):
        module = getattr(model, name, None)
        if isinstance(module, nn.Module):
            modules.append(module)
    return modules


def guided_target_gate_normal_anchor_loss(
    model: torch.nn.Module,
    valid_roi_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Keep target-supervised risk calibration close to Normal risk geometry."""

    raw_risk = getattr(model, "_guided_target_gate_raw_risk", None)
    gate_input = getattr(model, "_guided_target_gate_input_risk", None)
    mode_novelty = getattr(model, "_guided_target_gate_mode_novelty", None)
    mode_distance_ratio = getattr(
        model, "_guided_target_gate_mode_distance_ratio", None
    )
    mode_assignment = getattr(model, "_guided_target_gate_mode_assignment", None)
    objectness = getattr(model, "_guided_target_gate_objectness", None)
    mode_uncertainty = getattr(model, "_guided_target_gate_mode_uncertainty", None)
    calibrator = getattr(model, "guided_target_gate_calibrator", None)
    reject_gate = getattr(model, "guided_target_reject_gate", None)
    mode_gate = getattr(model, "guided_target_mode_gate", None)
    if raw_risk is None:
        reference = next(model.parameters())
        zero = reference.new_tensor(0.0)
        return zero, {"l_target_gate_normal_anchor": 0.0}
    valid_tokens = None
    if valid_roi_mask is not None:
        token_count = int(raw_risk.shape[1])
        side = int(math.isqrt(token_count))
        if side * side != token_count:
            raise ValueError(f"TargetGate token count must be square, got {token_count}.")
        valid_tokens = (
            F.adaptive_avg_pool2d(valid_roi_mask.float(), (side, side)).flatten(1)
            >= (1.0 - 1e-6)
        )
        if not bool(valid_tokens.any()):
            raise ValueError("TargetGate Normal anchor received an empty valid ROI.")
    if isinstance(mode_gate, PerModeTargetGate):
        if mode_distance_ratio is None or mode_assignment is None:
            raise RuntimeError("Per-mode TargetGate Normal anchor is missing mode features.")
        calibrated, logits = mode_gate(
            mode_distance_ratio.detach(),
            mode_assignment.detach(),
        )
        values = F.softplus(logits)
        loss = values.mean() if valid_tokens is None else values[valid_tokens].mean()
        raw_mean = raw_risk.mean() if valid_tokens is None else raw_risk[valid_tokens].mean()
        calibrated_mean = calibrated.mean() if valid_tokens is None else calibrated[valid_tokens].mean()
        return loss, {
            "l_target_gate_normal_anchor": float(loss.detach().cpu()),
            "target_gate_normal_raw_mean": float(raw_mean.detach().cpu()),
            "target_gate_normal_calibrated_mean": float(
                calibrated_mean.detach().cpu()
            ),
        }
    if isinstance(reject_gate, TargetRejectGate):
        if mode_novelty is None or objectness is None or mode_uncertainty is None:
            raise RuntimeError("Target-reject Normal anchor is missing gate features.")
        calibrated, _ = reject_gate(
            raw_risk.detach(),
            mode_novelty.detach(),
            objectness.detach(),
            mode_uncertainty.detach(),
        )
    elif isinstance(calibrator, FamiliarityRiskCalibrator):
        calibrated, _ = calibrator(
            (gate_input if gate_input is not None else raw_risk).detach()
        )
    else:
        reference = next(model.parameters())
        zero = reference.new_tensor(0.0)
        return zero, {"l_target_gate_normal_anchor": 0.0}
    squared = (calibrated - raw_risk.detach()).square()
    loss = squared.mean() if valid_tokens is None else squared[valid_tokens].mean()
    raw_mean = raw_risk.mean() if valid_tokens is None else raw_risk[valid_tokens].mean()
    calibrated_mean = calibrated.mean() if valid_tokens is None else calibrated[valid_tokens].mean()
    return loss, {
        "l_target_gate_normal_anchor": float(loss.detach().cpu()),
        "target_gate_normal_raw_mean": float(raw_mean.detach().cpu()),
        "target_gate_normal_calibrated_mean": float(calibrated_mean.detach().cpu()),
    }


def guided_target_gate_supervision_loss(
    model: torch.nn.Module,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Balanced token BCE for pasted-target versus clean-background write risk."""

    logits = getattr(model, "_guided_target_gate_logits", None)
    if logits is None or logits.ndim != 2:
        raise RuntimeError("Target-gate supervision did not receive token logits.")
    token_count = int(logits.shape[1])
    token_side = int(math.isqrt(token_count))
    if token_side * token_side != token_count:
        raise ValueError(f"Target-gate token count must be square, got {token_count}.")
    if target_mask.ndim != 4 or target_mask.shape[0] != logits.shape[0]:
        raise ValueError(
            "Target-gate mask must be [B,1,H,W] and match the logit batch."
        )
    token_target = F.adaptive_max_pool2d(target_mask.float(), (token_side, token_side))
    token_target = token_target.flatten(1) > 0.05
    positive = token_target
    negative = ~token_target
    if not bool(positive.any()) or not bool(negative.any()):
        raise RuntimeError("Target-gate supervision requires both target and background tokens.")
    positive_loss = F.softplus(-logits[positive]).mean()
    mode_gate = getattr(model, "guided_target_mode_gate", None)
    mode_assignment = getattr(model, "_guided_target_gate_mode_assignment", None)
    if isinstance(mode_gate, PerModeTargetGate):
        if not isinstance(mode_assignment, torch.Tensor) or (
            mode_assignment.shape != logits.shape
        ):
            raise RuntimeError("Per-mode TargetGate supervision is missing assignments.")
        hard_negative_losses = []
        for mode_index in range(mode_gate.mode_count):
            mode_positive = positive & (mode_assignment == mode_index)
            mode_negative = negative & (mode_assignment == mode_index)
            candidates = F.softplus(logits[mode_negative])
            if candidates.numel() == 0:
                continue
            keep = min(
                int(candidates.numel()),
                max(int(mode_positive.sum().item()), mode_gate.minimum_positive_support),
            )
            hard_negative_losses.append(candidates.topk(keep).values)
        if not hard_negative_losses:
            raise RuntimeError("Per-mode TargetGate found no background hard negatives.")
        negative_loss = torch.cat(hard_negative_losses).mean()
        mode_gate.record_support(mode_assignment, positive, negative)
    else:
        negative_loss = F.softplus(logits[negative]).mean()
    loss = 0.5 * (positive_loss + negative_loss)
    probabilities = logits.sigmoid().detach()
    diagnostics = {
        "l_target_gate": float(loss.detach().cpu()),
        "target_gate_positive_fraction": float(positive.float().mean().cpu()),
        "target_gate_positive_risk": float(probabilities[positive].mean().cpu()),
        "target_gate_background_risk": float(probabilities[negative].mean().cpu()),
        "target_gate_balanced_accuracy": float(
            0.5
            * (
                (probabilities[positive] >= 0.5).float().mean()
                + (probabilities[negative] < 0.5).float().mean()
            ).cpu()
        ),
        "target_gate_hard_negative": float(
            isinstance(mode_gate, PerModeTargetGate)
        ),
    }
    for name, value in (
        ("input_risk", getattr(model, "_guided_target_gate_input_risk", None)),
        ("mode_novelty", getattr(model, "_guided_target_gate_mode_novelty", None)),
        ("objectness", getattr(model, "_guided_target_gate_objectness", None)),
        ("mode_uncertainty", getattr(model, "_guided_target_gate_mode_uncertainty", None)),
    ):
        if isinstance(value, torch.Tensor) and value.shape == logits.shape:
            detached = value.detach()
            diagnostics[f"target_gate_positive_{name}"] = float(
                detached[positive].mean().cpu()
            )
            diagnostics[f"target_gate_background_{name}"] = float(
                detached[negative].mean().cpu()
            )
    return loss, diagnostics


def _mode_specific_routing_weights(
    token_weights: torch.Tensor,
    mode_probability: torch.Tensor,
    slot_to_mode: torch.Tensor,
    *,
    floor: float,
    strength: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine scalar write reliability with frozen mode-to-slot routing.

    ``token_weights`` is the existing familiarity/objectness write gate
    ``[B,N]``. ``mode_probability`` is the frozen Center6 posterior ``[B,N,M]``.
    The returned gate is ``[B,P,N]`` and therefore changes the attention logits
    for each physical prototype separately. The posterior is detached by design:
    routing guides prototype construction, but the model cannot alter token
    features merely to game a frozen routing decision.
    """

    if token_weights.ndim != 2:
        raise ValueError("Mode routing token weights must have shape [B,N].")
    if mode_probability.ndim != 3 or mode_probability.shape[:2] != token_weights.shape:
        raise ValueError(
            "Mode routing probability must have shape [B,N,M] and match token weights."
        )
    if slot_to_mode.ndim != 1 or slot_to_mode.numel() == 0:
        raise ValueError("Mode routing slot_to_mode must be a non-empty vector.")
    if not 0.0 < float(floor) <= 1.0:
        raise ValueError("Mode routing floor must be in (0,1].")
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("Mode routing strength must be in [0,1].")
    slot_to_mode = slot_to_mode.to(device=mode_probability.device, dtype=torch.long)
    if int(slot_to_mode.min().item()) < 0 or int(slot_to_mode.max().item()) >= int(
        mode_probability.shape[-1]
    ):
        raise ValueError("Mode routing slot_to_mode indexes an unavailable mode.")

    slot_probability = mode_probability.detach().index_select(-1, slot_to_mode)
    full_factor = float(floor) + (1.0 - float(floor)) * slot_probability
    routing_factor = 1.0 + float(strength) * (full_factor - 1.0)
    combined = token_weights[:, None, :] * routing_factor.permute(0, 2, 1)
    return combined, routing_factor.permute(0, 2, 1)


def _aggregation_attention_with_gate(
    attention: nn.Module,
    prototype: torch.Tensor,
    tokens: torch.Tensor,
    token_weights: torch.Tensor,
    *,
    valid_token_mask: torch.Tensor | None = None,
    return_attention: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply INP-Former aggregation attention with an additive write gate.

    A scalar gate has shape ``[B,N]`` and preserves the historical behavior by
    broadcasting one reliability value to every prototype. A mode-specific
    gate has shape ``[B,P,N]`` and supplies a different token prior to every
    physical prototype slot.
    """

    batch, prototype_count, channels = prototype.shape
    token_count = tokens.shape[1]
    heads = int(attention.num_heads)
    query = attention.q(prototype).reshape(
        batch, prototype_count, heads, channels // heads
    ).permute(0, 2, 1, 3)
    key_value = attention.kv(tokens).reshape(
        batch, token_count, 2, heads, channels // heads
    ).permute(2, 0, 3, 1, 4)
    key, value = key_value[0], key_value[1]
    logits = (query @ key.transpose(-2, -1)) * attention.scale
    if token_weights.ndim == 2:
        if token_weights.shape != (batch, token_count):
            raise ValueError(
                "Scalar aggregation gate must have shape [batch,tokens]."
            )
        log_gate = token_weights.clamp_min(1e-6).log()[:, None, None, :]
    elif token_weights.ndim == 3:
        if token_weights.shape != (batch, prototype_count, token_count):
            raise ValueError(
                "Mode-specific aggregation gate must have shape "
                "[batch,prototypes,tokens]."
            )
        log_gate = token_weights.clamp_min(1e-6).log()[:, None, :, :]
    else:
        raise ValueError("Aggregation gate must have shape [B,N] or [B,P,N].")
    logits = logits + log_gate
    if valid_token_mask is not None:
        if valid_token_mask.shape != (batch, token_count):
            raise ValueError(
                "Prototype aggregation validity mask must have shape [batch,tokens]."
            )
        valid_token_mask = valid_token_mask.to(device=logits.device, dtype=torch.bool)
        if not bool(valid_token_mask.any(dim=1).all()):
            invalid_batches = (~valid_token_mask.any(dim=1)).nonzero(
                as_tuple=False
            ).flatten().tolist()
            raise ValueError(
                "Prototype aggregation requires at least one valid ROI token per "
                f"sample; empty samples={invalid_batches}."
            )
        logits = logits.masked_fill(
            ~valid_token_mask[:, None, None, :], float("-inf")
        )
    probabilities = logits.softmax(dim=-1)
    weights = attention.attn_drop(probabilities)
    output = (weights @ value).transpose(1, 2).reshape(batch, prototype_count, channels)
    output = attention.proj(output)
    output = attention.proj_drop(output)
    if return_attention:
        return output, probabilities
    return output


def _aggregation_block_with_gate(
    block: nn.Module,
    prototype: torch.Tensor,
    tokens: torch.Tensor,
    token_weights: torch.Tensor,
    *,
    valid_token_mask: torch.Tensor | None = None,
    return_attention: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    normalized_prototype = block.norm1(prototype)
    normalized_tokens = block.norm1(tokens)
    attention_output = _aggregation_attention_with_gate(
        block.attn,
        normalized_prototype,
        normalized_tokens,
        token_weights,
        valid_token_mask=valid_token_mask,
        return_attention=return_attention,
    )
    if return_attention:
        update, weights = attention_output
    else:
        update = attention_output
    prototype = prototype + block.drop_path(update)
    prototype = prototype + block.drop_path(block.mlp(block.norm2(prototype)))
    if return_attention:
        return prototype, weights
    return prototype


def _decoder_read_gate_activation(
    risk: torch.Tensor,
    mode: str,
    tail_threshold: float,
    tail_upper: float,
    tail_power: float,
) -> torch.Tensor:
    """Map raw familiarity risk to the read-gate activation."""

    if mode == "suppress":
        return risk.clamp(0.0, 1.0)
    if mode not in {"tail_suppress", "selective_route"}:
        raise ValueError(f"Unknown decoder read-gate mode: {mode!r}.")
    if not 0.0 <= float(tail_threshold) < float(tail_upper) <= 1.0:
        raise ValueError(
            "Decoder read-gate tail calibration must satisfy "
            "0 <= threshold < upper <= 1."
        )
    if float(tail_power) <= 0.0:
        raise ValueError(f"Decoder read-gate tail power must be positive, got {tail_power}.")
    return (
        (risk.clamp(0.0, 1.0) - float(tail_threshold))
        / (float(tail_upper) - float(tail_threshold))
    ).clamp(0.0, 1.0).pow(float(tail_power))


def _apply_decoder_read_gate(
    attention_weights: torch.Tensor,
    risk: torch.Tensor,
    groups: tuple[int, int, int],
    strength: float,
    scope: str,
    mode: str = "suppress",
    tail_threshold: float = 0.0,
    tail_upper: float = 1.0,
    tail_power: float = 1.0,
    attention_aware: bool = False,
    responsibility_aware: bool = False,
    adaptive_strength_power: float = 0.0,
) -> torch.Tensor:
    """Gate decoder reads, optionally preserving mass by routing object reads.

    INP-Former's decoder uses non-negative ReLU attention without a softmax.  In
    ``selective_route`` mode the sum over prototypes is therefore preserved for
    every batch/head/token entry instead of renormalizing the attention vector.
    """

    if attention_weights.ndim != 4:
        raise ValueError(
            "Decoder read-gate attention must have shape [B,H,N,P], got "
            f"{tuple(attention_weights.shape)}."
        )
    scalar_shape = (attention_weights.shape[0], attention_weights.shape[2])
    prototype_shape = (*scalar_shape, attention_weights.shape[3])
    if tuple(risk.shape) not in {scalar_shape, prototype_shape}:
        raise ValueError(
            "Decoder read-gate shape mismatch: "
            f"attention={tuple(attention_weights.shape)} risk={tuple(risk.shape)}"
        )
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError(f"Decoder read-gate strength must be in [0,1], got {strength}.")
    if float(adaptive_strength_power) < 0.0:
        raise ValueError("Decoder read adaptive-strength power must be non-negative.")
    if mode == "selective_route" and scope != "object":
        raise ValueError("Selective decoder read routing requires scope 'object'.")
    if mode == "tail_suppress" and scope == "all" and risk.ndim != 3:
        raise ValueError(
            "All-prototype tail suppression requires one calibrated risk per prototype."
        )
    if attention_aware and scope != "object":
        raise ValueError("Attention-aware decoder read gating requires scope 'object'.")
    if responsibility_aware and scope != "object":
        raise ValueError("Responsibility-aware decoder read gating requires scope 'object'.")
    if responsibility_aware and risk.ndim != 2:
        raise ValueError("Responsibility-aware decoder read gating requires scalar token risk.")
    calibrated_risk = _decoder_read_gate_activation(
        risk,
        mode,
        tail_threshold,
        tail_upper,
        tail_power,
    )

    activation = (
        calibrated_risk[:, None, :, None]
        if calibrated_risk.ndim == 2
        else calibrated_risk[:, None, :, :]
    )
    if attention_aware:
        object_slice = _group_slices(groups)[2]
        object_mass = attention_weights[:, :, :, object_slice].sum(dim=-1, keepdim=True)
        total_mass = attention_weights.sum(dim=-1, keepdim=True)
        object_share = object_mass / total_mass.clamp_min(
            torch.finfo(attention_weights.dtype).eps
        )
        uniform_object_share = float(groups[2]) / float(sum(groups))
        attention_confidence = (object_share / uniform_object_share).clamp(0.0, 1.0)
        # A head that does not read the object group should not be changed merely
        # because the token has high familiarity risk.  Group-size normalization
        # makes the rule parameter-free and leaves at/above-uniform reads untouched.
        activation = activation * attention_confidence
    if responsibility_aware:
        object_slice = _group_slices(groups)[2]
        object_weights = attention_weights[:, :, :, object_slice]
        object_mass = object_weights.sum(dim=-1, keepdim=True)
        responsibility = object_weights / object_mass.clamp_min(
            torch.finfo(object_weights.dtype).eps
        )
        activation = activation * responsibility
    effective_strength: float | torch.Tensor = float(strength)
    if float(adaptive_strength_power) > 0.0:
        effective_strength = float(strength) + (1.0 - float(strength)) * activation.pow(
            float(adaptive_strength_power)
        )
    factor = (1.0 - effective_strength * activation).clamp(0.0, 1.0)
    if scope == "all":
        return attention_weights * factor
    if scope != "object":
        raise ValueError(f"Unknown decoder read-gate scope: {scope!r}.")
    object_slice = _group_slices(groups)[2]
    gated = attention_weights.clone()
    # Clone before the slice assignment below: a view would observe the gated
    # values and underestimate the removed mass.
    object_weights = gated[:, :, :, object_slice].clone()
    object_factor = (
        factor
        if factor.shape[-1] in {1, int(groups[2])}
        else factor[:, :, :, object_slice]
    )
    gated[:, :, :, object_slice] = object_weights * object_factor
    if mode == "selective_route":
        non_object_count = int(groups[0] + groups[1])
        if non_object_count <= 0:
            raise ValueError("Selective read routing requires background/texture prototypes.")
        removed_mass = (object_weights * (1.0 - factor)).sum(dim=-1, keepdim=True)
        routing_weights = gated[:, :, :, :non_object_count]
        routing_sum = routing_weights.sum(dim=-1, keepdim=True)
        proportional = routing_weights / routing_sum.clamp_min(
            torch.finfo(routing_weights.dtype).eps
        )
        uniform = torch.full_like(routing_weights, 1.0 / float(non_object_count))
        routing_distribution = torch.where(routing_sum > 0.0, proportional, uniform)
        gated[:, :, :, :non_object_count] = (
            routing_weights + removed_mass * routing_distribution
        )
    return gated


def _cap_removed_decoder_update(
    ungated_update: torch.Tensor,
    gated_update: torch.Tensor,
    cap: float,
) -> torch.Tensor:
    """Cap the removed update norm relative to the ungated decoder update."""

    if ungated_update.shape != gated_update.shape:
        raise ValueError(
            "Decoder update-cap shape mismatch: "
            f"ungated={tuple(ungated_update.shape)} gated={tuple(gated_update.shape)}"
        )
    if not 0.0 < float(cap) <= 1.0:
        raise ValueError(f"Decoder update cap must be in (0,1], got {cap}.")
    removed = ungated_update - gated_update
    removed_norm = removed.norm(dim=-1, keepdim=True)
    reference_norm = ungated_update.norm(dim=-1, keepdim=True)
    allowed_norm = float(cap) * reference_norm
    scale = torch.minimum(
        torch.ones_like(removed_norm),
        allowed_norm / removed_norm.clamp_min(torch.finfo(removed_norm.dtype).eps),
    )
    return ungated_update - removed * scale


def _apply_descriptor_read_route(
    attention_weights: torch.Tensor,
    expected_normal: torch.Tensor,
    prototype: torch.Tensor,
    risk: torch.Tensor,
    *,
    strength: float,
    tail_threshold: float,
    tail_upper: float,
    tail_power: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route read mass toward prototypes compatible with context-predicted Normal.

    Unlike the legacy read gate, this operation has no fixed prototype slices.
    It preserves the total non-negative read mass for every head and token.
    """

    if attention_weights.ndim != 4:
        raise ValueError("Descriptor read attention must have shape [B,H,N,P].")
    if expected_normal.shape[:2] != risk.shape or expected_normal.ndim != 3:
        raise ValueError("Descriptor expected Normal and risk shapes disagree.")
    if prototype.ndim != 3 or prototype.shape[0] != expected_normal.shape[0]:
        raise ValueError("Descriptor prototype shape is incompatible with expected Normal.")
    if prototype.shape[1] != attention_weights.shape[-1]:
        raise ValueError("Descriptor prototype count does not match decoder attention.")
    if prototype.shape[-1] != expected_normal.shape[-1]:
        raise ValueError("Descriptor feature dimensions do not match.")
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("Descriptor read strength must be in [0,1].")

    activation = _decoder_read_gate_activation(
        risk,
        "tail_suppress",
        tail_threshold,
        tail_upper,
        tail_power,
    )
    activation = (float(strength) * activation).clamp(0.0, 1.0)
    compatibility_logits = F.cosine_similarity(
        expected_normal.unsqueeze(2).float(),
        prototype.unsqueeze(1).float(),
        dim=-1,
    )
    compatibility = F.softmax(compatibility_logits / 0.10, dim=-1).to(
        attention_weights.dtype
    )
    total_mass = attention_weights.sum(dim=-1, keepdim=True)
    compatible_attention = attention_weights * compatibility[:, None, :, :]
    compatible_sum = compatible_attention.sum(dim=-1, keepdim=True)
    fallback = total_mass * compatibility[:, None, :, :]
    routed = torch.where(
        compatible_sum > torch.finfo(attention_weights.dtype).eps,
        compatible_attention
        * (total_mass / compatible_sum.clamp_min(torch.finfo(attention_weights.dtype).eps)),
        fallback,
    )
    mixed = attention_weights + activation[:, None, :, None] * (
        routed - attention_weights
    )
    return mixed, activation


def _prototype_block_with_descriptor_read_route(
    block: nn.Module,
    tokens: torch.Tensor,
    prototype: torch.Tensor,
    expected_normal: torch.Tensor,
    risk: torch.Tensor,
    config: GuidedPrototypeConfig,
) -> torch.Tensor:
    """Run one decoder block with semantic-free context-conditioned routing."""

    attention = block.attn
    normalized_tokens = block.norm1(tokens)
    normalized_prototype = block.norm1(prototype)
    batch, token_count, channels = normalized_tokens.shape
    prototype_count = normalized_prototype.shape[1]
    heads = int(attention.num_heads)
    query = attention.q(normalized_tokens).reshape(
        batch, token_count, heads, channels // heads
    ).permute(0, 2, 1, 3)
    key_value = attention.kv(normalized_prototype).reshape(
        batch, prototype_count, 2, heads, channels // heads
    ).permute(2, 0, 3, 1, 4)
    key, value = key_value[0], key_value[1]
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    ungated_weights = F.relu(
        (query @ key.transpose(-2, -1)) * attention.learn_scale
    )
    routed_weights, _ = _apply_descriptor_read_route(
        ungated_weights,
        expected_normal,
        prototype,
        risk,
        strength=config.descriptor_read_strength,
        tail_threshold=config.descriptor_read_tail_threshold,
        tail_upper=config.descriptor_read_tail_upper,
        tail_power=config.descriptor_read_tail_power,
    )
    drop_probability = float(getattr(attention.attn_drop, "p", 0.0))
    if attention.training and drop_probability > 0.0:
        keep_probability = 1.0 - drop_probability
        shared_mask = torch.empty_like(routed_weights).bernoulli_(keep_probability)
        shared_mask = shared_mask / keep_probability
        ungated_weights = ungated_weights * shared_mask
        routed_weights = routed_weights * shared_mask
    ungated_update = (ungated_weights @ value).transpose(1, 2).reshape(
        batch, token_count, channels
    )
    routed_update = (routed_weights @ value).transpose(1, 2).reshape(
        batch, token_count, channels
    )
    ungated_update = attention.proj(ungated_update)
    routed_update = attention.proj(routed_update)
    if float(config.descriptor_read_update_cap) > 0.0:
        update = _cap_removed_decoder_update(
            ungated_update,
            routed_update,
            config.descriptor_read_update_cap,
        )
    else:
        update = routed_update
    update = attention.proj_drop(update)
    tokens = block.drop_path(update)
    return tokens + block.drop_path(block.mlp(block.norm2(tokens)))


def _decoder_read_layer_enabled(index: int, layer_count: int, start: int) -> bool:
    """Return whether a zero-based decoder block belongs to the gated suffix."""

    if layer_count <= 0:
        raise ValueError("Decoder layer count must be positive.")
    if not 0 <= int(index) < int(layer_count):
        raise ValueError(f"Decoder layer index {index} is outside [0,{layer_count}).")
    if not 0 <= int(start) < int(layer_count):
        raise ValueError(f"Decoder read layer start {start} is outside [0,{layer_count}).")
    return int(index) >= int(start)


def _prototype_block_with_read_gate(
    block: nn.Module,
    tokens: torch.Tensor,
    prototype: torch.Tensor,
    risk: torch.Tensor,
    config: GuidedPrototypeConfig,
    *,
    strength: float | None = None,
) -> torch.Tensor:
    """Run an INP-Former decoder block with a group-selective familiarity read gate."""

    attention = block.attn
    normalized_tokens = block.norm1(tokens)
    normalized_prototype = block.norm1(prototype)
    batch, token_count, channels = normalized_tokens.shape
    prototype_count = normalized_prototype.shape[1]
    heads = int(attention.num_heads)
    query = attention.q(normalized_tokens).reshape(
        batch, token_count, heads, channels // heads
    ).permute(0, 2, 1, 3)
    key_value = attention.kv(normalized_prototype).reshape(
        batch, prototype_count, 2, heads, channels // heads
    ).permute(2, 0, 3, 1, 4)
    key, value = key_value[0], key_value[1]
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    weights = F.relu((query @ key.transpose(-2, -1)) * attention.learn_scale)
    ungated_weights = weights
    weights = _apply_decoder_read_gate(
        ungated_weights,
        risk,
        config.groups,
        config.decoder_read_strength if strength is None else float(strength),
        config.decoder_read_scope,
        config.decoder_read_mode,
        config.decoder_read_tail_threshold,
        config.decoder_read_tail_upper,
        config.decoder_read_tail_power,
        config.decoder_read_attention_aware,
        config.decoder_read_responsibility_aware,
        config.decoder_read_adaptive_strength_power,
    )
    if float(config.decoder_read_update_cap) > 0.0:
        drop_probability = float(getattr(attention.attn_drop, "p", 0.0))
        if attention.training and drop_probability > 0.0:
            keep_probability = 1.0 - drop_probability
            shared_mask = torch.empty_like(weights).bernoulli_(keep_probability)
            shared_mask = shared_mask / keep_probability
            ungated_weights = ungated_weights * shared_mask
            weights = weights * shared_mask
        ungated_update = (ungated_weights @ value).transpose(1, 2).reshape(
            batch, token_count, channels
        )
        gated_update = (weights @ value).transpose(1, 2).reshape(
            batch, token_count, channels
        )
        ungated_update = attention.proj(ungated_update)
        gated_update = attention.proj(gated_update)
        update = _cap_removed_decoder_update(
            ungated_update,
            gated_update,
            config.decoder_read_update_cap,
        )
        update = attention.proj_drop(update)
    else:
        weights = attention.attn_drop(weights)
        update = (weights @ value).transpose(1, 2).reshape(batch, token_count, channels)
        update = attention.proj_drop(attention.proj(update))
    tokens = block.drop_path(update)
    return tokens + block.drop_path(block.mlp(block.norm2(tokens)))


def _aggregate_guided_prototypes(
    self,
    target_tokens: torch.Tensor,
    images: torch.Tensor,
    prototype_context: torch.Tensor | None = None,
) -> torch.Tensor:
    if bool(getattr(self, "_rgpl_historical_compatible_raw", False)):
        return _aggregate_historical_compatible_rgpl_raw(
            self,
            target_tokens,
            images,
            prototype_context=prototype_context,
        )
    config: GuidedPrototypeConfig = self._guided_prototype_config
    self._guided_decoder_target_tokens = target_tokens.detach()
    alpha = _guided_alpha(self, config)
    aggregation_alpha = _guided_aggregation_alpha(self, config)
    batch, token_count, _ = target_tokens.shape
    token_side = int(math.isqrt(token_count))
    if token_side * token_side != token_count:
        raise ValueError(f"Guided token count must be square, got {token_count}.")
    spatial_valid = getattr(self, "_guided_valid_roi_mask", None)
    roi_valid = _strict_token_roi_mask(
        spatial_valid,
        batch=batch,
        token_count=token_count,
    )
    roi_exclusion = None if roi_valid is None else ~roi_valid
    uniform_priors = target_tokens.new_full((batch, token_count, 3), 1.0 / 3.0)
    priors = uniform_priors
    fixed_priors = uniform_priors.detach()
    objectness = target_tokens.new_zeros((batch, token_count))
    novelty = torch.ones_like(objectness)
    write_risk = torch.zeros_like(objectness)
    distance = torch.zeros_like(objectness)
    mode_novelty = torch.zeros_like(objectness)
    mode_distance_ratio = torch.zeros_like(objectness)
    mode_uncertainty = torch.zeros_like(objectness)
    mode_routing_probability: torch.Tensor | None = None
    mode_routing_factor: torch.Tensor | None = None
    decoder_read_risk: torch.Tensor | None = None
    memory = getattr(self, "guided_normal_object_memory", None)
    transport_diag = {}
    descriptor_diag = {}
    descriptor_write_risk: torch.Tensor | None = None
    self._guided_descriptor_expected_normal = None
    self._guided_target_gate_raw_risk = None
    self._guided_target_gate_input_risk = None
    self._guided_target_gate_mode_novelty = None
    self._guided_target_gate_mode_distance_ratio = None
    self._guided_target_gate_mode_assignment = None
    self._guided_target_gate_objectness = None
    self._guided_target_gate_mode_uncertainty = None
    self._guided_target_gate_logits = None
    calibrated_write_risk: torch.Tensor | None = None
    if config.descriptor_variant != "off":
        from .normal_descriptor_gate import NormalDescriptorMemory

        descriptor_memory = getattr(self, "guided_normal_descriptor_memory", None)
        if not isinstance(descriptor_memory, NormalDescriptorMemory):
            raise RuntimeError("Normal descriptor memory was not configured.")
        evidence, descriptor_diag = descriptor_memory.describe(
            target_tokens,
            getattr(self, "_guided_source_group_ids", None),
        )
        descriptor_write_risk = evidence.write_risk
        decoder_read_risk = evidence.read_risk
        objectness = evidence.local_objectness
        novelty = evidence.appearance_novelty
        if config.descriptor_variant == "d":
            physical_priors = _build_priors(
                target_tokens,
                config,
                images,
                normalizer=getattr(self, "guided_prior_normalizer", None),
                update_stats=bool(
                    self.training and getattr(self, "_guided_update_prior_stats", True)
                ),
            )
            physical_objectness = physical_priors[:, :, 2].detach()
            descriptor_write_risk, objectness_factor = _descriptor_aux_objectness(
                descriptor_write_risk,
                physical_objectness,
                floor=config.descriptor_objectness_floor,
            )
            decoder_read_risk, _ = _descriptor_aux_objectness(
                decoder_read_risk,
                physical_objectness,
                floor=config.descriptor_objectness_floor,
            )
            objectness = physical_objectness
            descriptor_diag["guided_descriptor_objectness_factor"] = float(
                objectness_factor.mean().detach().cpu()
            )
        token_weights = _descriptor_write_weights(
            descriptor_write_risk,
            minimum=config.aggregation_min_weight,
            power=config.aggregation_power,
            alpha=aggregation_alpha,
        )
        context = target_tokens if prototype_context is None else prototype_context
        self._guided_descriptor_expected_normal = evidence.expected_normal.detach()
        priors = uniform_priors
        fixed_priors = uniform_priors.detach()
    elif config.context_transport:
        from .context_familiarity_transport import ContextFamiliarityTransportMemory

        transport_memory = getattr(self, "guided_context_transport_memory", None)
        if not isinstance(transport_memory, ContextFamiliarityTransportMemory):
            raise RuntimeError("Context familiarity transport memory was not configured.")
        if config.context_adaptive_scale:
            transport_objectness = torch.ones_like(objectness)
        else:
            priors, fixed_priors = _build_trainable_priors(
                self, target_tokens, config, images
            )
            transport_objectness = priors[:, :, 2].detach()
        transported, adaptive_objectness, transport_diag = transport_memory.transport(
            target_tokens,
            transport_objectness,
            fixed_scale_weights=config.object_kernel_weights,
            adaptive_scale=config.context_adaptive_scale,
            source_group_ids=getattr(self, "_guided_source_group_ids", None),
        )
        if config.context_adaptive_scale:
            objectness = adaptive_objectness.detach()
            if not config.free_prototypes:
                priors, fixed_priors = _build_trainable_priors(
                    self,
                    target_tokens,
                    config,
                    images,
                    objectness_override=objectness,
                )
        else:
            objectness = transport_objectness
        context = torch.lerp(target_tokens, transported, alpha)
        token_weights = torch.ones_like(objectness)
    else:
        if config.center6_balanced:
            teacher = getattr(self, "guided_center6_teacher", None)
            if not isinstance(teacher, FrozenCenter6Teacher):
                raise RuntimeError("Center6 teacher is not configured on the model.")
            priors = _center6_group_priors(
                target_tokens,
                teacher,
                config.groups,
                temperature=config.center6_teacher_temperature,
                mapping=config.center6_prior_mapping,
            )
            fixed_priors = priors.detach()
        else:
            priors, fixed_priors = _build_trainable_priors(
                self, target_tokens, config, images
            )
        if getattr(self, "_guided_vlm_prior_sidecar", None) is not None:
            objectness = _build_guided_objectness(
                self,
                target_tokens,
                config,
                images,
            ).detach()
        else:
            objectness = priors[:, :, 2].detach()
        guided_token_weights = torch.ones_like(objectness)
        if config.aggregation_familiarity_gate or config.decoder_read_gate:
            if not isinstance(memory, NormalObjectTokenMemory):
                raise RuntimeError("Normal object-token memory was not configured.")
            novelty, distance = memory.novelty(target_tokens)
            risk = _compose_aggregation_risk(
                objectness,
                novelty,
                config.aggregation_risk_composition,
            )
            write_risk = risk
            if (
                config.target_gate_risk_source == "mode_normalized"
                or config.target_reject_gate
                or config.target_mode_gate
            ):
                teacher = getattr(self, "guided_center6_teacher", None)
                if not isinstance(teacher, FrozenCenter6Teacher):
                    raise RuntimeError("Mode-normalized target gate requires Center6 teacher.")
                (
                    mode_novelty,
                    mode_distance_ratio,
                    mode_uncertainty,
                    mode_assignment,
                ) = (
                    center6_mode_normalized_features(
                        target_tokens,
                        teacher,
                        temperature=config.target_mode_novelty_temperature,
                        return_assignment=True,
                    )
                )
            gate_input_risk = risk
            if config.target_gate_risk_source == "mode_normalized":
                gate_input_risk = objectness * mode_novelty
            if config.target_mode_gate:
                mode_gate = getattr(self, "guided_target_mode_gate", None)
                if not isinstance(mode_gate, PerModeTargetGate):
                    raise RuntimeError("Per-mode target gate was not configured.")
                self._guided_target_gate_raw_risk = risk.detach()
                self._guided_target_gate_input_risk = mode_distance_ratio.detach()
                self._guided_target_gate_mode_novelty = mode_novelty.detach()
                self._guided_target_gate_mode_distance_ratio = (
                    mode_distance_ratio.detach()
                )
                self._guided_target_gate_mode_assignment = mode_assignment.detach()
                self._guided_target_gate_objectness = objectness.detach()
                self._guided_target_gate_mode_uncertainty = mode_uncertainty.detach()
                write_risk, calibrated_logits = mode_gate(
                    mode_distance_ratio,
                    mode_assignment,
                    detach_parameters=bool(
                        self.training
                        and getattr(self, "_guided_update_prior_stats", True)
                    ),
                )
                self._guided_target_gate_logits = calibrated_logits
                calibrated_write_risk = write_risk
            elif config.target_reject_gate:
                reject_gate = getattr(self, "guided_target_reject_gate", None)
                if not isinstance(reject_gate, TargetRejectGate):
                    raise RuntimeError("Target-reject gate was not configured.")
                self._guided_target_gate_raw_risk = risk.detach()
                self._guided_target_gate_input_risk = gate_input_risk.detach()
                self._guided_target_gate_mode_novelty = mode_novelty.detach()
                self._guided_target_gate_objectness = objectness.detach()
                self._guided_target_gate_mode_uncertainty = mode_uncertainty.detach()
                write_risk, calibrated_logits = reject_gate(
                    risk,
                    mode_novelty,
                    objectness,
                    mode_uncertainty,
                    detach_parameters=bool(
                        self.training
                        and getattr(self, "_guided_update_prior_stats", True)
                    ),
                )
                self._guided_target_gate_logits = calibrated_logits
                calibrated_write_risk = write_risk
            elif config.target_gate_calibration:
                calibrator = getattr(self, "guided_target_gate_calibrator", None)
                if not isinstance(calibrator, FamiliarityRiskCalibrator):
                    raise RuntimeError("Target-gate calibrator was not configured.")
                self._guided_target_gate_raw_risk = risk.detach()
                self._guided_target_gate_input_risk = gate_input_risk.detach()
                self._guided_target_gate_mode_novelty = mode_novelty.detach()
                self._guided_target_gate_objectness = objectness.detach()
                self._guided_target_gate_mode_uncertainty = mode_uncertainty.detach()
                write_risk, calibrated_logits = calibrator(
                    gate_input_risk,
                    detach_parameters=bool(
                        self.training
                        and getattr(self, "_guided_update_prior_stats", True)
                    ),
                )
                self._guided_target_gate_logits = calibrated_logits
                calibrated_write_risk = write_risk
            if config.aggregation_familiarity_gate and config.familiarity_write_enabled:
                guided_token_weights = float(config.aggregation_min_weight) + (
                    1.0 - float(config.aggregation_min_weight)
                ) * (1.0 - write_risk).pow(float(config.aggregation_power))
            should_update = _familiarity_memory_update_enabled(
                config,
                training=self.training,
                update_prior_stats=getattr(
                    self, "_guided_update_prior_stats", True
                ),
            )
            if should_update:
                memory.update(
                    target_tokens,
                    objectness,
                    getattr(self, "_guided_source_group_ids", None),
                    valid_mask=None if roi_exclusion is None else ~roi_exclusion,
                )
        elif config.aggregation_gate:
            guided_token_weights = (1.0 - objectness).pow(float(config.aggregation_power))
            guided_token_weights = guided_token_weights.clamp_min(
                float(config.aggregation_min_weight)
            )
        if (
            config.aggregation_familiarity_gate
            and config.familiarity_write_enabled
            and float(config.aggregation_power) == 1.0
        ):
            # Active Adaptive Raw settings collapse exactly to one linear rule:
            # 1 + alpha * ([m + (1-m)(1-risk)] - 1)
            # = 1 - alpha * (1-m) * risk.
            token_weights = 1.0 - (
                aggregation_alpha
                * (1.0 - float(config.aggregation_min_weight))
                * write_risk
            )
        else:
            token_weights = 1.0 + aggregation_alpha * (guided_token_weights - 1.0)
        context = target_tokens if prototype_context is None else prototype_context

    decoder_objectness = objectness
    if config.decoder_read_gate:
        if (
            config.decoder_read_risk_source == "target_gate"
            and calibrated_write_risk is not None
        ):
            decoder_read_risk = calibrated_write_risk.detach()
        elif config.target_reject_gate and calibrated_write_risk is not None:
            decoder_read_risk = calibrated_write_risk
        else:
            decoder_read_risk, decoder_objectness = _build_decoder_read_risk(
                self,
                target_tokens,
                images,
                config,
                novelty,
                aggregation_priors=priors,
            )
    if decoder_read_risk is None:
        decoder_read_risk = objectness * novelty

    # Keep the historical scalar diagnostics stable while exposing the actual
    # decoder input separately when Center6 uses prototype-wise risks.
    scalar_decoder_risk = (
        decoder_read_risk
        if decoder_read_risk.ndim == 2
        else decoder_read_risk.max(dim=-1).values
    )
    self._guided_decoder_objectness = decoder_objectness.detach()
    self._guided_decoder_risk = scalar_decoder_risk.detach()
    self._guided_decoder_read_risk = decoder_read_risk.detach()

    self._guided_pending_prior_state = {"prior": priors, "fixed": fixed_priors}
    self._guided_last_aggregation_risk = write_risk.detach()
    external_exclusion = getattr(self, "_guided_external_aggregation_exclusion", None)
    hard_roi_attention = bool(
        getattr(self, "_prototype_hard_roi_attention", False)
    )
    transition_roi_attention = bool(
        getattr(self, "_prototype_transition_roi_attention", False)
    )
    transition_roi_prior = None
    if transition_roi_attention:
        transition_roi_prior = _transition_token_roi_prior(
            spatial_valid,
            batch=batch,
            token_count=token_count,
            floor=float(getattr(self, "_prototype_transition_roi_floor", 1e-6)),
        )
        if transition_roi_prior is not None:
            if token_weights.ndim == 2:
                token_weights = token_weights * transition_roi_prior.to(token_weights)
            else:
                token_weights = token_weights * transition_roi_prior[:, None, :].to(
                    token_weights
                )
    valid_token_mask = None if roi_exclusion is None else ~roi_exclusion
    if (
        roi_exclusion is not None
        and not hard_roi_attention
        and not transition_roi_attention
    ):
        external_exclusion = (
            roi_exclusion
            if external_exclusion is None
            else torch.logical_or(external_exclusion.bool(), roi_exclusion)
        )
    if external_exclusion is not None:
        if external_exclusion.shape != token_weights.shape:
            raise ValueError(
                "External prototype exclusion/gate shape mismatch: "
                f"exclusion={tuple(external_exclusion.shape)} "
                f"gate={tuple(token_weights.shape)}"
            )
        token_weights = token_weights * (
            1.0 - external_exclusion.to(device=token_weights.device, dtype=token_weights.dtype)
        )
    if config.mode_specific_routing:
        teacher = getattr(self, "guided_center6_teacher", None)
        if not isinstance(teacher, FrozenCenter6Teacher):
            raise RuntimeError("Mode-specific routing requires a configured Center6 teacher.")
        # Route according to the exact key/value context consumed by the
        # aggregation block. This matters when a masked-reconstruction caller
        # supplies prototype_context distinct from the decoder target tokens.
        mode_routing_probability = _center6_teacher_probability(
            context,
            teacher,
            temperature=config.center6_teacher_temperature,
            hierarchical=config.center6_hierarchical_teacher,
        )
        token_weights, mode_routing_factor = _mode_specific_routing_weights(
            token_weights,
            mode_routing_probability,
            teacher.slot_to_mode,
            floor=config.mode_routing_floor,
            strength=config.mode_routing_strength,
        )
    expected_gate_shape = (
        (context.shape[0], int(self.prototype_token.shape[0]), context.shape[1])
        if token_weights.ndim == 3
        else context.shape[:2]
    )
    if tuple(token_weights.shape) != tuple(expected_gate_shape):
        raise ValueError(
            f"Prototype context/gate shape mismatch: context={tuple(context.shape)} "
            f"gate={tuple(token_weights.shape)}"
        )

    batch = target_tokens.shape[0]
    prototype = self.prototype_token.unsqueeze(0).repeat(batch, 1, 1)
    capture_attention = bool(getattr(self, "_guided_capture_aggregation_attention", False))
    captured_attention = []
    for block in self.aggregation:
        block_output = _aggregation_block_with_gate(
            block,
            prototype,
            context,
            token_weights,
            valid_token_mask=valid_token_mask if hard_roi_attention else None,
            return_attention=capture_attention,
        )
        if capture_attention:
            prototype, block_attention = block_output
            captured_attention.append(block_attention)
        else:
            prototype = block_output
    self._guided_last_aggregate_prototype = prototype
    self._guided_last_aggregation_attention = tuple(captured_attention)
    self._guided_aggregation_diag = {
        "guided_gate_mean": float(token_weights.mean().detach().cpu()),
        "guided_gate_min": float(token_weights.min().detach().cpu()),
        "guided_gate_suppressed": float(
            (token_weights < 0.5).float().mean().detach().cpu()
        ),
        "guided_hard_roi_attention": float(hard_roi_attention),
        "guided_transition_roi_attention": float(transition_roi_attention),
        "guided_transition_roi_prior_mean": float(
            transition_roi_prior.mean().detach().cpu()
        ) if transition_roi_prior is not None else 1.0,
        "guided_roi_valid_fraction": float(
            valid_token_mask.float().mean().detach().cpu()
        ) if valid_token_mask is not None else 1.0,
        "guided_native_anchor_alpha": float(alpha),
        "guided_aggregation_alpha": float(aggregation_alpha),
        "guided_mode_routing_active": float(config.mode_specific_routing),
        "guided_mode_routing_strength": float(config.mode_routing_strength),
        "guided_mode_routing_floor": float(config.mode_routing_floor),
        "guided_mode_routing_factor_mean": float(
            mode_routing_factor.mean().detach().cpu()
        ) if mode_routing_factor is not None else 1.0,
        "guided_mode_routing_factor_min": float(
            mode_routing_factor.min().detach().cpu()
        ) if mode_routing_factor is not None else 1.0,
        "guided_mode_routing_factor_max": float(
            mode_routing_factor.max().detach().cpu()
        ) if mode_routing_factor is not None else 1.0,
        "guided_mode_routing_factor_std": float(
            mode_routing_factor.float().std(unbiased=False).detach().cpu()
        ) if mode_routing_factor is not None else 0.0,
        "guided_familiarity_count": float(memory.count.item())
        if isinstance(memory, NormalObjectTokenMemory)
        else 0.0,
        "guided_novelty_mean": float(novelty.mean().detach().cpu()),
        "guided_novelty_distance": float(distance.mean().detach().cpu()),
        "guided_pollution_risk": float(
            (
                descriptor_write_risk
                if descriptor_write_risk is not None
                else objectness * novelty
            ).mean().detach().cpu()
        ),
        "guided_target_gate_calibrated_risk": float(
            calibrated_write_risk.mean().detach().cpu()
        ) if calibrated_write_risk is not None else 0.0,
        "guided_target_gate_scale": float(
            self.guided_target_gate_calibrator.log_scale.exp().detach().cpu()
        ) if config.target_gate_calibration else 1.0,
        "guided_target_gate_bias": float(
            self.guided_target_gate_calibrator.bias.detach().cpu()
        ) if config.target_gate_calibration else 0.0,
        "guided_target_mode_gate_global_slope": float(
            self.guided_target_mode_gate.global_log_slope.exp().detach().cpu()
        ) if config.target_mode_gate else 0.0,
        "guided_target_mode_gate_global_boundary": float(
            self.guided_target_mode_gate.global_boundary.detach().cpu()
        ) if config.target_mode_gate else 0.0,
        "guided_target_mode_gate_supported_modes": float(
            (
                self.guided_target_mode_gate.positive_support
                >= self.guided_target_mode_gate.minimum_positive_support
            ).sum().detach().cpu()
        ) if config.target_mode_gate else 0.0,
        "guided_target_mode_novelty": float(mode_novelty.mean().detach().cpu()),
        "guided_target_mode_distance_ratio": float(
            mode_distance_ratio.mean().detach().cpu()
        ),
        "guided_target_mode_uncertainty": float(
            mode_uncertainty.mean().detach().cpu()
        ),
        "guided_target_reject_weight_mode": float(
            self.guided_target_reject_gate.weights()[0].detach().cpu()
        ) if config.target_reject_gate else 0.0,
        "guided_target_reject_weight_objectness": float(
            self.guided_target_reject_gate.weights()[1].detach().cpu()
        ) if config.target_reject_gate else 0.0,
        "guided_target_reject_weight_uncertainty": float(
            self.guided_target_reject_gate.weights()[2].detach().cpu()
        ) if config.target_reject_gate else 0.0,
        "guided_target_reject_bias": float(
            self.guided_target_reject_gate.bias.detach().cpu()
        ) if config.target_reject_gate else 0.0,
        "guided_decoder_read_risk_mean": float(
            scalar_decoder_risk.mean().detach().cpu()
        ),
        "guided_decoder_read_prototypewise": float(decoder_read_risk.ndim == 3),
        "guided_novelty_threshold": float(memory.novelty_threshold.item())
        if isinstance(memory, NormalObjectTokenMemory)
        else 0.0,
        "guided_calibration_groups": float(memory.calibration_group_count.item())
        if isinstance(memory, NormalObjectTokenMemory)
        else 0.0,
        "guided_calibrated_count": float(memory.calibrated_count.item())
        if isinstance(memory, NormalObjectTokenMemory)
        else 0.0,
        **transport_diag,
        **descriptor_diag,
    }
    return prototype


def _aggregate_historical_compatible_rgpl_raw(
    self,
    target_tokens: torch.Tensor,
    images: torch.Tensor,
    *,
    prototype_context: torch.Tensor | None = None,
) -> torch.Tensor:
    """Audited single path for the historical memory-176 RGPL Raw model.

    The function intentionally contains only the operations active in the
    2026-07-29 best run: Center6-derived objectness, frozen-backbone familiarity,
    a scalar risk gate, strict normal-memory eligibility, and epsilon-weighted
    ROI exclusion inside prototype attention.
    """

    config: GuidedPrototypeConfig = self._guided_prototype_config
    self._guided_decoder_target_tokens = target_tokens.detach()
    batch, token_count, _ = target_tokens.shape
    token_side = int(math.isqrt(token_count))
    if token_side * token_side != token_count:
        raise ValueError(f"Guided token count must be square, got {token_count}.")

    spatial_valid = getattr(self, "_guided_valid_roi_mask", None)
    valid_tokens = _strict_token_roi_mask(
        spatial_valid,
        batch=batch,
        token_count=token_count,
    )
    roi_exclusion = None if valid_tokens is None else ~valid_tokens

    teacher = getattr(self, "guided_center6_teacher", None)
    if not isinstance(teacher, FrozenCenter6Teacher):
        raise RuntimeError("Historical-compatible RGPL Raw requires Center6 teacher.")
    memory = getattr(self, "guided_normal_object_memory", None)
    if not isinstance(memory, NormalObjectTokenMemory):
        raise RuntimeError("Historical-compatible RGPL Raw requires familiarity memory.")

    priors = _center6_group_priors(
        target_tokens,
        teacher,
        config.groups,
        temperature=config.center6_teacher_temperature,
        mapping=config.center6_prior_mapping,
    )
    fixed_priors = priors.detach()
    objectness = priors[:, :, 2].detach()
    novelty, distance = memory.novelty(target_tokens)
    risk = _compose_aggregation_risk(
        objectness,
        novelty,
        config.aggregation_risk_composition,
    )
    aggregation_alpha = _guided_aggregation_alpha(self, config)
    token_weights = 1.0 - (
        aggregation_alpha
        * (1.0 - float(config.aggregation_min_weight))
        * risk
    )

    if _familiarity_memory_update_enabled(
        config,
        training=self.training,
        update_prior_stats=getattr(self, "_guided_update_prior_stats", True),
    ):
        memory.update(
            target_tokens,
            objectness,
            getattr(self, "_guided_source_group_ids", None),
            valid_mask=valid_tokens,
        )

    self._guided_descriptor_expected_normal = None
    self._guided_target_gate_raw_risk = None
    self._guided_target_gate_input_risk = None
    self._guided_target_gate_mode_novelty = None
    self._guided_target_gate_mode_distance_ratio = None
    self._guided_target_gate_mode_assignment = None
    self._guided_target_gate_objectness = None
    self._guided_target_gate_mode_uncertainty = None
    self._guided_target_gate_logits = None
    self._guided_decoder_objectness = objectness.detach()
    self._guided_decoder_risk = risk.detach()
    self._guided_decoder_read_risk = risk.detach()
    self._guided_pending_prior_state = {"prior": priors, "fixed": fixed_priors}
    self._guided_last_aggregation_risk = risk.detach()

    external_exclusion = getattr(self, "_guided_external_aggregation_exclusion", None)
    if roi_exclusion is not None:
        external_exclusion = (
            roi_exclusion
            if external_exclusion is None
            else torch.logical_or(external_exclusion.bool(), roi_exclusion)
        )
    if external_exclusion is not None:
        if external_exclusion.shape != token_weights.shape:
            raise ValueError(
                "Historical RGPL exclusion/gate shape mismatch: "
                f"exclusion={tuple(external_exclusion.shape)} "
                f"gate={tuple(token_weights.shape)}"
            )
        token_weights = token_weights * (
            1.0
            - external_exclusion.to(
                device=token_weights.device,
                dtype=token_weights.dtype,
            )
        )

    context = target_tokens if prototype_context is None else prototype_context
    prototype = self.prototype_token.unsqueeze(0).repeat(batch, 1, 1)
    capture_attention = bool(getattr(self, "_guided_capture_aggregation_attention", False))
    captured_attention = []
    for block in self.aggregation:
        block_output = _aggregation_block_with_gate(
            block,
            prototype,
            context,
            token_weights,
            return_attention=capture_attention,
        )
        if capture_attention:
            prototype, block_attention = block_output
            captured_attention.append(block_attention)
        else:
            prototype = block_output

    self._guided_last_aggregate_prototype = prototype
    self._guided_last_aggregation_attention = tuple(captured_attention)
    self._guided_aggregation_diag = {
        "guided_gate_mean": float(token_weights.mean().detach().cpu()),
        "guided_gate_min": float(token_weights.min().detach().cpu()),
        "guided_gate_suppressed": float(
            (token_weights < 0.5).float().mean().detach().cpu()
        ),
        "guided_native_anchor_alpha": float(_guided_alpha(self, config)),
        "guided_aggregation_alpha": float(aggregation_alpha),
        "guided_familiarity_count": float(memory.count.item()),
        "guided_novelty_mean": float(novelty.mean().detach().cpu()),
        "guided_novelty_distance": float(distance.mean().detach().cpu()),
        "guided_pollution_risk": float(risk.mean().detach().cpu()),
        "guided_novelty_threshold": float(memory.novelty_threshold.item()),
        "guided_calibration_groups": float(memory.calibration_group_count.item()),
        "guided_calibrated_count": float(memory.calibrated_count.item()),
        "guided_historical_compatible_raw": 1.0,
    }
    return prototype


def _compose_aggregation_risk(
    objectness: torch.Tensor,
    novelty: torch.Tensor,
    composition: str,
) -> torch.Tensor:
    """Compose the scalar RGPL risk, including controlled single-factor ablations."""

    if objectness.shape != novelty.shape:
        raise ValueError(
            "RGPL objectness/novelty shape mismatch: "
            f"{tuple(objectness.shape)} versus {tuple(novelty.shape)}"
        )
    if composition == "product":
        return objectness * novelty
    if composition == "objectness_only":
        return objectness
    if composition == "novelty_only":
        return novelty
    raise ValueError(f"Unknown RGPL aggregation-risk composition: {composition!r}.")


def _strict_token_roi_mask(
    spatial_mask: torch.Tensor | None,
    *,
    batch: int,
    token_count: int,
) -> torch.Tensor | None:
    """Project a pixel ROI to tokens, retaining only fully covered tokens."""

    coverage = _token_roi_coverage(
        spatial_mask,
        batch=batch,
        token_count=token_count,
    )
    if coverage is None:
        return None
    return coverage >= (1.0 - 1e-6)


def _token_roi_coverage(
    spatial_mask: torch.Tensor | None,
    *,
    batch: int,
    token_count: int,
) -> torch.Tensor | None:
    """Project a pixel ROI to the fraction covered by each encoder token."""

    if spatial_mask is None:
        return None
    if spatial_mask.ndim != 4 or spatial_mask.shape[0] != batch:
        raise ValueError("Valid ROI must have shape [B,1,H,W] and match the batch.")
    token_side = int(math.isqrt(token_count))
    if token_side * token_side != token_count:
        raise ValueError(f"ROI-aware token count must be square, got {token_count}.")
    return F.adaptive_avg_pool2d(
        spatial_mask.float(), (token_side, token_side)
    ).flatten(1).clamp(0.0, 1.0)


def _transition_token_roi_prior(
    spatial_mask: torch.Tensor | None,
    *,
    batch: int,
    token_count: int,
    floor: float = 1e-6,
) -> torch.Tensor | None:
    """Linearly map token ROI coverage to a strictly positive attention prior."""

    if not 0.0 < float(floor) <= 1.0:
        raise ValueError("Transition ROI floor must be in (0,1].")
    coverage = _token_roi_coverage(
        spatial_mask,
        batch=batch,
        token_count=token_count,
    )
    if coverage is None:
        return None
    return float(floor) + (1.0 - float(floor)) * coverage


def _roi_aware_inpformer_forward(self, images: torch.Tensor) -> tuple:
    """Native INP-Former forward with hard or coverage-transition ROI attention."""

    x = self.encoder.prepare_tokens(images)
    encoder_features = []
    for index, block in enumerate(self.encoder.blocks):
        if index > self.target_layers[-1]:
            continue
        if index in self.encoder_require_grad_layer:
            x = block(x)
        else:
            with torch.no_grad():
                x = block(x)
        if index in self.target_layers:
            encoder_features.append(x)
    if not encoder_features:
        raise RuntimeError("No INP-Former encoder features were captured.")

    token_start = 1 + self.encoder.num_register_tokens
    side = int(math.sqrt(encoder_features[0].shape[1] - token_start))
    batch = images.shape[0]
    if self.remove_class_token:
        encoder_features = [feature[:, token_start:, :] for feature in encoder_features]
    target_tokens = self.fuse_feature(encoder_features)
    spatial_valid = getattr(self, "_guided_valid_roi_mask", None)
    hard_roi_attention = bool(getattr(self, "_prototype_hard_roi_attention", False))
    transition_roi_attention = bool(
        getattr(self, "_prototype_transition_roi_attention", False)
    )
    valid_token_mask = _strict_token_roi_mask(
        spatial_valid,
        batch=batch,
        token_count=target_tokens.shape[1],
    ) if hard_roi_attention else None
    transition_prior = _transition_token_roi_prior(
        spatial_valid,
        batch=batch,
        token_count=target_tokens.shape[1],
        floor=float(getattr(self, "_prototype_transition_roi_floor", 1e-6)),
    ) if transition_roi_attention else None
    unit_gate = (
        transition_prior.to(target_tokens)
        if transition_prior is not None
        else target_tokens.new_ones(target_tokens.shape[:2])
    )
    prototype = self.prototype_token.unsqueeze(0).repeat(batch, 1, 1)
    for block in self.aggregation:
        prototype = _aggregation_block_with_gate(
            block,
            prototype,
            target_tokens,
            unit_gate,
            valid_token_mask=valid_token_mask,
        )
    gather_loss = self.gather_loss(target_tokens, prototype)

    x = target_tokens
    for block in self.bottleneck:
        x = block(x)
    decoder_features = []
    for block in self.decoder:
        x = block(x, prototype)
        decoder_features.append(x)
    decoder_features = decoder_features[::-1]

    encoder_output = [
        self.fuse_feature([encoder_features[index] for index in indices])
        for indices in self.fuse_layer_encoder
    ]
    decoder_output = [
        self.fuse_feature([decoder_features[index] for index in indices])
        for indices in self.fuse_layer_decoder
    ]
    if not self.remove_class_token:
        encoder_output = [feature[:, token_start:, :] for feature in encoder_output]
        decoder_output = [feature[:, token_start:, :] for feature in decoder_output]
    encoder_maps = [
        feature.permute(0, 2, 1).reshape(batch, -1, side, side).contiguous()
        for feature in encoder_output
    ]
    decoder_maps = [
        feature.permute(0, 2, 1).reshape(batch, -1, side, side).contiguous()
        for feature in decoder_output
    ]
    self._hard_roi_aggregation_diag = {
        "hard_roi_attention": float(hard_roi_attention),
        "transition_roi_attention": float(transition_roi_attention),
        "roi_valid_fraction": float(
            valid_token_mask.float().mean().detach().cpu()
        ) if valid_token_mask is not None else 1.0,
        "transition_roi_prior_mean": float(
            transition_prior.mean().detach().cpu()
        ) if transition_prior is not None else 1.0,
    }
    return encoder_maps, decoder_maps, gather_loss


def _enable_roi_aware_inpformer_forward(model: torch.nn.Module) -> None:
    if not hasattr(model, "prototype_token") or not hasattr(model, "aggregation"):
        raise ValueError("ROI-aware aggregation requires an INP-Former model.")
    if not hasattr(model, "_original_hard_roi_forward"):
        model._original_hard_roi_forward = model.forward
        model.forward = types.MethodType(_roi_aware_inpformer_forward, model)


def _guided_inpformer_forward(self, images: torch.Tensor) -> tuple:
    """INP-Former forward with guidance restricted to prototype aggregation."""

    config: GuidedPrototypeConfig = self._guided_prototype_config
    x = self.encoder.prepare_tokens(images)
    encoder_features = []
    for index, block in enumerate(self.encoder.blocks):
        if index > self.target_layers[-1]:
            continue
        if index in self.encoder_require_grad_layer:
            x = block(x)
        else:
            with torch.no_grad():
                x = block(x)
        if index in self.target_layers:
            encoder_features.append(x)
    if not encoder_features:
        raise RuntimeError("No INP-Former encoder features were captured.")

    token_start = 1 + self.encoder.num_register_tokens
    side = int(math.sqrt(encoder_features[0].shape[1] - token_start))
    batch = images.shape[0]
    if self.remove_class_token:
        encoder_features = [feature[:, token_start:, :] for feature in encoder_features]

    target_tokens = self.fuse_feature(encoder_features)
    self._guided_prototype_image = images.detach()
    prototype = self.aggregate_guided_prototypes(target_tokens, images)
    gather_loss = self.gather_loss(target_tokens, prototype)

    x = target_tokens
    for block in self.bottleneck:
        x = block(x)
    decoder_features = []
    decoder_risk = getattr(
        self,
        "_guided_decoder_read_risk",
        getattr(self, "_guided_decoder_risk", None),
    )
    descriptor_read_enabled = config.descriptor_variant in {"c", "d"}
    descriptor_expected_normal = getattr(
        self, "_guided_descriptor_expected_normal", None
    )
    decoder_layer_count = len(self.decoder)
    if config.decoder_read_gate and not 0 <= int(config.decoder_read_layer_start) < decoder_layer_count:
        raise ValueError(
            "Decoder read layer start must select at least one decoder block, got "
            f"start={config.decoder_read_layer_start} count={decoder_layer_count}."
        )
    if descriptor_read_enabled and not 0 <= int(
        config.descriptor_read_layer_start
    ) < decoder_layer_count:
        raise ValueError(
            "Descriptor read layer start must select at least one decoder block, got "
            f"start={config.descriptor_read_layer_start} count={decoder_layer_count}."
        )
    if config.decoder_read_layer_strengths and len(config.decoder_read_layer_strengths) != decoder_layer_count:
        raise ValueError(
            "Decoder read layer strengths must provide one value per decoder block, got "
            f"{len(config.decoder_read_layer_strengths)} for {decoder_layer_count} blocks."
        )
    for block_index, block in enumerate(self.decoder):
        descriptor_layer_enabled = bool(
            descriptor_read_enabled
            and _decoder_read_layer_enabled(
                block_index,
                decoder_layer_count,
                config.descriptor_read_layer_start,
            )
        )
        layer_gate_enabled = bool(
            config.decoder_read_gate
            and _decoder_read_layer_enabled(
                block_index,
                decoder_layer_count,
                config.decoder_read_layer_start,
            )
        )
        if descriptor_layer_enabled:
            if decoder_risk is None or descriptor_expected_normal is None:
                raise RuntimeError(
                    "Descriptor read route did not receive Normal evidence."
                )
            x = _prototype_block_with_descriptor_read_route(
                block,
                x,
                prototype,
                descriptor_expected_normal.to(device=x.device, dtype=x.dtype),
                decoder_risk.to(device=x.device, dtype=x.dtype),
                config,
            )
        elif layer_gate_enabled:
            if decoder_risk is None:
                raise RuntimeError("Decoder read gate did not receive familiarity risk.")
            x = _prototype_block_with_read_gate(
                block,
                x,
                prototype,
                decoder_risk.to(device=x.device, dtype=x.dtype),
                config,
                strength=(
                    config.decoder_read_layer_strengths[block_index]
                    if config.decoder_read_layer_strengths
                    else config.decoder_read_strength
                ),
            )
        else:
            x = block(x, prototype)
        decoder_features.append(x)
    decoder_features = decoder_features[::-1]

    if config.decoder_read_gate and decoder_risk is not None:
        read_activation = _decoder_read_gate_activation(
            decoder_risk,
            config.decoder_read_mode,
            config.decoder_read_tail_threshold,
            config.decoder_read_tail_upper,
            config.decoder_read_tail_power,
        )
        enabled_strengths = (
            config.decoder_read_layer_strengths[config.decoder_read_layer_start :]
            if config.decoder_read_layer_strengths
            else (float(config.decoder_read_strength),)
        )
        mean_read_strength = float(sum(enabled_strengths) / len(enabled_strengths))
        read_factor = 1.0 - mean_read_strength * read_activation
        self._guided_aggregation_diag.update(
            {
                "guided_decoder_read_risk_mean": float(decoder_risk.mean().cpu()),
                "guided_decoder_read_activation_mean": float(read_activation.mean().cpu()),
                "guided_decoder_read_factor_mean": float(read_factor.mean().cpu()),
                "guided_decoder_read_factor_min": float(read_factor.min().cpu()),
                "guided_decoder_read_layer_start": float(config.decoder_read_layer_start),
                "guided_decoder_read_layer_count": float(
                    decoder_layer_count - config.decoder_read_layer_start
                ),
                "guided_decoder_read_strength_mean": mean_read_strength,
            }
        )
    if descriptor_read_enabled and decoder_risk is not None:
        descriptor_activation = _decoder_read_gate_activation(
            decoder_risk,
            "tail_suppress",
            config.descriptor_read_tail_threshold,
            config.descriptor_read_tail_upper,
            config.descriptor_read_tail_power,
        )
        self._guided_aggregation_diag.update(
            {
                "guided_descriptor_read_activation": float(
                    descriptor_activation.mean().cpu()
                ),
                "guided_descriptor_read_layer_start": float(
                    config.descriptor_read_layer_start
                ),
                "guided_descriptor_read_layer_count": float(
                    decoder_layer_count - config.descriptor_read_layer_start
                ),
                "guided_descriptor_read_strength": float(
                    config.descriptor_read_strength
                ),
            }
        )

    encoder_output = [
        self.fuse_feature([encoder_features[index] for index in indices])
        for indices in self.fuse_layer_encoder
    ]
    decoder_output = [
        self.fuse_feature([decoder_features[index] for index in indices])
        for indices in self.fuse_layer_decoder
    ]
    if not self.remove_class_token:
        encoder_output = [feature[:, token_start:, :] for feature in encoder_output]
        decoder_output = [feature[:, token_start:, :] for feature in decoder_output]
    encoder_maps = [
        feature.permute(0, 2, 1).reshape(batch, -1, side, side).contiguous()
        for feature in encoder_output
    ]
    decoder_maps = [
        feature.permute(0, 2, 1).reshape(batch, -1, side, side).contiguous()
        for feature in decoder_output
    ]
    return encoder_maps, decoder_maps, gather_loss


def inpformer_forward_with_prototype_context(model: torch.nn.Module, images: torch.Tensor) -> tuple:
    """Run INP-Former and expose fused tokens plus image-conditioned prototypes."""

    required = ("encoder", "target_layers", "fuse_feature", "bottleneck", "decoder")
    if not all(hasattr(model, name) for name in required):
        raise ValueError("Prototype-context scoring is only implemented for INP-Former-like models.")

    x = model.encoder.prepare_tokens(images)
    en_list = []
    for i, blk in enumerate(model.encoder.blocks):
        if i <= model.target_layers[-1]:
            if i in model.encoder_require_grad_layer:
                x = blk(x)
            else:
                with torch.no_grad():
                    x = blk(x)
        else:
            continue
        if i in model.target_layers:
            en_list.append(x)
    if not en_list:
        raise RuntimeError("No INP-Former encoder features were captured.")

    original_start = 1 + model.encoder.num_register_tokens
    side = int(math.sqrt(en_list[0].shape[1] - original_start))
    batch = images.shape[0]
    if model.remove_class_token:
        en_list = [item[:, original_start:, :] for item in en_list]

    target_tokens = model.fuse_feature(en_list)
    if hasattr(model, "aggregate_guided_prototypes"):
        current_prototype = model.aggregate_guided_prototypes(target_tokens, images)
    else:
        current_prototype = model.prototype_token
        for blk in model.aggregation:
            current_prototype = blk(current_prototype.unsqueeze(0).repeat((batch, 1, 1)), target_tokens)
    g_loss = model.gather_loss(target_tokens, current_prototype)

    x = target_tokens
    for blk in model.bottleneck:
        x = blk(x)

    de_list = []
    for blk in model.decoder:
        x = blk(x, current_prototype)
        de_list.append(x)
    de_list = de_list[::-1]

    en = [model.fuse_feature([en_list[idx] for idx in idxs]) for idxs in model.fuse_layer_encoder]
    de = [model.fuse_feature([de_list[idx] for idx in idxs]) for idxs in model.fuse_layer_decoder]

    if not model.remove_class_token:
        en = [item[:, original_start:, :] for item in en]
        de = [item[:, original_start:, :] for item in de]

    en_maps = [item.permute(0, 2, 1).reshape([batch, -1, side, side]).contiguous() for item in en]
    de_maps = [item.permute(0, 2, 1).reshape([batch, -1, side, side]).contiguous() for item in de]
    context = {
        "target_tokens": target_tokens,
        "agg_prototype": current_prototype,
        "side": side,
        "g_loss": g_loss,
    }
    return en_maps, de_maps, context


def prototype_objectness_mismatch_map(
    target_tokens: torch.Tensor,
    agg_prototype: torch.Tensor,
    images: torch.Tensor | None,
    config: GuidedPrototypeConfig,
    margin: float = 0.0,
    temperature: float = 0.08,
    normalize: bool = True,
) -> torch.Tensor:
    priors = _build_priors(target_tokens, config, images)
    distribution = 1.0 - F.cosine_similarity(target_tokens.unsqueeze(2), agg_prototype.unsqueeze(1), dim=-1)
    similarity = 1.0 - distribution
    bg_slice, texture_slice, object_slice = _group_slices(config.groups)
    bg_sim = similarity[:, :, bg_slice].max(dim=2).values
    texture_sim = similarity[:, :, texture_slice].max(dim=2).values
    object_sim = similarity[:, :, object_slice].max(dim=2).values
    non_object_sim = torch.maximum(bg_sim, texture_sim)
    mismatch = torch.sigmoid((non_object_sim - object_sim - float(margin)) / max(float(temperature), 1e-6))
    objectness = priors[:, :, 2]
    score = objectness * mismatch
    side = int(math.sqrt(target_tokens.shape[1]))
    score = score.view(target_tokens.shape[0], 1, side, side)
    if normalize:
        score = _norm01(score)
    return score.to(dtype=target_tokens.dtype)


def _norm01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    flat = x.flatten(1)
    min_v = flat.min(dim=1).values.view(-1, 1, 1, 1)
    max_v = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return ((x - min_v) / (max_v - min_v).clamp_min(eps)).clamp(0.0, 1.0)


def _feature_neighbor_change(feat: torch.Tensor, offset: int = 1) -> torch.Tensor:
    b, _, h, w = feat.shape
    offset = int(offset)
    if offset <= 0 or offset >= min(h, w):
        raise ValueError(f"Neighbor offset must be in [1, {min(h, w) - 1}], got {offset}.")
    out = feat.new_zeros((b, 1, h, w))
    count = feat.new_zeros((b, 1, h, w))
    right = (
        1.0 - (feat[:, :, :, :-offset] * feat[:, :, :, offset:]).sum(dim=1, keepdim=True)
    ).clamp_min(0.0)
    down = (
        1.0 - (feat[:, :, :-offset, :] * feat[:, :, offset:, :]).sum(dim=1, keepdim=True)
    ).clamp_min(0.0)
    out[:, :, :, :-offset] += right
    out[:, :, :, offset:] += right
    count[:, :, :, :-offset] += 1.0
    count[:, :, :, offset:] += 1.0
    out[:, :, :-offset, :] += down
    out[:, :, offset:, :] += down
    count[:, :, :-offset, :] += 1.0
    count[:, :, offset:, :] += 1.0
    return out / count.clamp_min(1.0)


def _feature_local_contrast(feat: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    pad = kernel_size // 2
    local = F.avg_pool2d(feat, kernel_size=kernel_size, stride=1, padding=pad)
    local = F.normalize(local, dim=1)
    return (1.0 - (feat * local).sum(dim=1, keepdim=True)).clamp_min(0.0)


def _image_edge_texture(
    images: torch.Tensor | None,
    side: int,
    device: torch.device,
    reference_size: int = 0,
    physical_rgb: bool = False,
    replicate_padding: bool = False,
    normalize: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if images is None:
        return None, None
    img = images.to(device=device, dtype=torch.float32)
    if reference_size > 0 and img.shape[-2:] != (reference_size, reference_size):
        img = F.interpolate(img, size=(reference_size, reference_size), mode="bilinear", align_corners=False)

    if physical_rgb:
        mean = img.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = img.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        rgb = (img * std + mean).clamp(0.0, 1.0)
        luminance = img.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        gray = (rgb * luminance).sum(dim=1, keepdim=True)
    else:
        gray = img.mean(dim=1, keepdim=True)
    sobel_x = gray.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    sobel_y = gray.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    lap = gray.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
    if replicate_padding:
        padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
        gx = F.conv2d(padded, sobel_x)
        gy = F.conv2d(padded, sobel_y)
        texture = F.conv2d(padded, lap).abs()
    else:
        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        texture = F.conv2d(gray, lap, padding=1).abs()
    edge = torch.sqrt((gx * gx + gy * gy).clamp_min(1e-12))
    edge = F.interpolate(edge, size=(side, side), mode="bilinear", align_corners=False)
    texture = F.interpolate(texture, size=(side, side), mode="bilinear", align_corners=False)
    if normalize:
        return _norm01(edge), _norm01(texture)
    return edge, texture


def _weighted_sum(items: Sequence[tuple[torch.Tensor | None, float]], fallback: torch.Tensor) -> torch.Tensor:
    acc = None
    total = 0.0
    for tensor, weight in items:
        if tensor is None or weight <= 0.0:
            continue
        acc = tensor * weight if acc is None else acc + tensor * weight
        total += weight
    if acc is None or total <= 0.0:
        return fallback
    return acc / total


def _scale_odd_kernel(kernel: int, scale: float) -> int:
    """Scale an odd kernel by its radius and keep the result odd."""

    radius = (int(kernel) - 1) // 2
    scaled_radius = max(1, math.floor(radius * float(scale) + 0.5))
    return 2 * scaled_radius + 1


def _scale_offset(offset: int, scale: float) -> int:
    return max(1, math.floor(int(offset) * float(scale) + 0.5))


def resolve_spatial_prior_parameters(
    config: GuidedPrototypeConfig,
    token_side: int,
) -> tuple[tuple[int, ...], tuple[int, ...], float]:
    """Resolve integer token windows for the current grid without feature interpolation."""

    if config.spatial_scale_mode == "legacy":
        return config.object_kernels, config.texture_offsets, 1.0
    if config.spatial_scale_mode != "integer":
        raise ValueError(f"Unknown spatial scale mode: {config.spatial_scale_mode!r}.")
    if config.scale_consistent:
        raise ValueError("Integer spatial scaling cannot be combined with reference-grid interpolation.")
    if config.spatial_reference_side <= 0:
        raise ValueError("Spatial reference token side must be positive.")

    scale = float(token_side) / float(config.spatial_reference_side)
    kernels = tuple(_scale_odd_kernel(kernel, scale) for kernel in config.object_kernels)
    offsets = tuple(_scale_offset(offset, scale) for offset in config.texture_offsets)
    return kernels, offsets, scale


def _build_priors(
    query: torch.Tensor,
    config: GuidedPrototypeConfig,
    image: torch.Tensor | None,
    normalizer: RobustPriorNormalizer | None = None,
    update_stats: bool = False,
    objectness_override: torch.Tensor | None = None,
) -> torch.Tensor:
    b, n, c = query.shape
    output_side = int(math.sqrt(n))
    if output_side * output_side != n:
        base = query.new_full((b, n, 3), 1.0 / 3.0)
        return base

    prior_side = (
        int(config.prior_reference_side)
        if config.scale_consistent and config.prior_reference_side > 0
        else output_side
    )
    feat = query.detach().transpose(1, 2).reshape(b, c, output_side, output_side).float()
    if feat.shape[-2:] != (prior_side, prior_side):
        feat = F.interpolate(feat, size=(prior_side, prior_side), mode="bilinear", align_corners=False)
    feat = F.normalize(feat, dim=1)
    if objectness_override is not None:
        if objectness_override.shape != (b, n):
            raise ValueError(
                f"Objectness override must have shape {(b, n)}, got "
                f"{tuple(objectness_override.shape)}."
            )
        override_map = objectness_override.detach().float().reshape(
            b, 1, output_side, output_side
        )
        if override_map.shape[-2:] != (prior_side, prior_side):
            override_map = F.interpolate(
                override_map,
                size=(prior_side, prior_side),
                mode="bilinear",
                align_corners=False,
            )
    else:
        override_map = None
    if config.multiscale_direct:
        object_kernels, texture_offsets, _ = resolve_spatial_prior_parameters(config, prior_side)
        feat_texture = sum(
            weight * _feature_neighbor_change(feat, offset=offset)
            for offset, weight in zip(texture_offsets, config.texture_offset_weights)
        )
        feat_object = (
            None
            if override_map is not None
            else sum(
                weight * _feature_local_contrast(feat, kernel_size=kernel)
                for kernel, weight in zip(object_kernels, config.object_kernel_weights)
            )
        )
    else:
        feat_texture = _feature_neighbor_change(feat)
        feat_object = (
            None if override_map is not None else _feature_local_contrast(feat, kernel_size=5)
        )
    needs_image_prior = (
        float(config.image_texture_weight) > 0.0
        or (override_map is None and float(config.image_object_weight) > 0.0)
    )
    if needs_image_prior:
        img_edge, img_texture = _image_edge_texture(
            image,
            side=prior_side,
            device=query.device,
            reference_size=int(config.image_reference_size) if config.scale_consistent else 0,
            physical_rgb=config.scale_consistent or config.multiscale_direct,
            replicate_padding=config.scale_consistent or config.multiscale_direct,
            normalize=not config.multiscale_direct,
        )
    else:
        img_edge, img_texture = None, None

    if config.multiscale_direct:
        batch_normalizer = normalizer or RobustPriorNormalizer(
            config.robust_quantile_low,
            config.robust_quantile_high,
            momentum=0.0,
        ).to(query.device)
        feat_texture = batch_normalizer.normalize(feat_texture, "feature_texture", update=update_stats)
        if feat_object is not None:
            feat_object = batch_normalizer.normalize(
                feat_object, "feature_object", update=update_stats
            )
        if img_texture is not None:
            img_texture = batch_normalizer.normalize(img_texture, "image_texture", update=update_stats)
        if img_edge is not None and override_map is None:
            img_edge = batch_normalizer.normalize(img_edge, "image_edge", update=update_stats)
    else:
        feat_texture = _norm01(feat_texture)
        if feat_object is not None:
            feat_object = _norm01(feat_object)

    texture = _weighted_sum(
        [
            (feat_texture, config.feature_texture_weight),
            (img_texture, config.image_texture_weight),
        ],
        fallback=feat_texture,
    )
    texture = _norm01(texture).pow(config.texture_power)
    if override_map is not None:
        objectness = override_map.clamp(0.0, 1.0).pow(config.object_power)
    else:
        if feat_object is None:
            raise RuntimeError("Fixed objectness features were not constructed.")
        objectness = _weighted_sum(
            [
                (feat_object, config.feature_object_weight),
                (img_edge, config.image_object_weight),
            ],
            fallback=feat_object,
        )
        objectness = _norm01(objectness).pow(config.object_power)
    if config.object_texture_suppress > 0.0:
        objectness = objectness * (1.0 - config.object_texture_suppress * texture).clamp_min(0.0)

    bg = (1.0 - torch.maximum(texture, objectness)).clamp_min(0.0)
    scores = torch.cat([bg, texture, objectness], dim=1)
    floor = max(0.0, float(config.prior_floor))
    if floor > 0.0:
        scores = scores + floor
    scores = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-6)
    if prior_side != output_side:
        scores = F.interpolate(scores, size=(output_side, output_side), mode="bilinear", align_corners=False)
        scores = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-6)
    effective_group_weights = (1.0, 1.0, 1.0) if config.free_prototypes else config.group_weights
    group_weights = scores.new_tensor(effective_group_weights).view(1, 3, 1, 1)
    scores = scores * group_weights
    scores = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return scores.permute(0, 2, 3, 1).reshape(b, n, 3).to(dtype=query.dtype)


def _build_trainable_priors(
    model: torch.nn.Module,
    query: torch.Tensor,
    config: GuidedPrototypeConfig,
    image: torch.Tensor | None,
    objectness_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if config.multiscale_direct:
        output_side = int(math.sqrt(query.shape[1]))
        kernels, offsets, scale = resolve_spatial_prior_parameters(config, output_side)
        model._guided_spatial_diag = {
            "guided_spatial_scale": float(scale),
            **{f"guided_object_kernel_{index}": float(value) for index, value in enumerate(kernels)},
            **{f"guided_texture_offset_{index}": float(value) for index, value in enumerate(offsets)},
        }
    normalizer = getattr(model, "guided_prior_normalizer", None)
    fixed = _build_priors(
        query,
        config,
        image,
        normalizer=normalizer if isinstance(normalizer, RobustPriorNormalizer) else None,
        update_stats=bool(
            model.training
            and config.multiscale_direct
            and getattr(model, "_guided_update_prior_stats", True)
        ),
        objectness_override=objectness_override,
    )
    vlm_sidecar = getattr(model, "_guided_vlm_prior_sidecar", None)
    if vlm_sidecar is not None:
        from .vlm_semantic_prior import (
            correct_physical_prior_with_vlm,
            fuse_vlm_and_physical_prior,
            predict_vlm_prior,
        )

        vlm_prior = predict_vlm_prior(vlm_sidecar, query.detach(), image)
        correction_weight = None
        if config.vlm_prior_fusion_mode == "low_confidence_correction":
            fixed, correction_weight = correct_physical_prior_with_vlm(
                vlm_prior,
                fixed,
                max_weight=config.vlm_prior_correction_max_weight,
                physical_margin_threshold=config.vlm_prior_correction_physical_margin,
                vlm_margin_threshold=config.vlm_prior_correction_vlm_margin,
                return_weight=True,
            )
        else:
            fixed = fuse_vlm_and_physical_prior(
                vlm_prior,
                fixed,
                config.vlm_prior_physical_weight,
            )
        model._guided_vlm_prior_diag = {
            "fusion_mode": config.vlm_prior_fusion_mode,
            "physical_weight": float(config.vlm_prior_physical_weight),
            "background_mean": float(fixed[..., 0].detach().float().mean()),
            "texture_mean": float(fixed[..., 1].detach().float().mean()),
            "object_mean": float(fixed[..., 2].detach().float().mean()),
        }
        if correction_weight is not None:
            model._guided_vlm_prior_diag.update(
                correction_weight_mean=float(correction_weight.detach().float().mean()),
                correction_active_fraction=float(
                    (correction_weight.detach() > 0.0).float().mean()
                ),
            )
    head = getattr(model, "guided_prior_head", None)
    if not config.trainable_prior or head is None:
        return fixed, fixed.detach()
    head_query = query.detach().float()
    output_side = int(math.sqrt(query.shape[1]))
    reference_side = int(config.prior_reference_side)
    use_reference_grid = (
        config.scale_consistent
        and reference_side > 0
        and output_side * output_side == query.shape[1]
        and output_side != reference_side
    )
    if use_reference_grid:
        feature_map = head_query.transpose(1, 2).reshape(
            query.shape[0], query.shape[2], output_side, output_side
        )
        feature_map = F.interpolate(
            feature_map,
            size=(reference_side, reference_side),
            mode="bilinear",
            align_corners=False,
        )
        reference_query = feature_map.flatten(2).transpose(1, 2).contiguous()
        delta = head(reference_query)
        delta_map = delta.transpose(1, 2).reshape(query.shape[0], 3, reference_side, reference_side)
        delta = F.interpolate(
            delta_map,
            size=(output_side, output_side),
            mode="bilinear",
            align_corners=False,
        ).flatten(2).transpose(1, 2).contiguous()
    else:
        delta = head(head_query)
    delta = delta.to(dtype=query.dtype)
    logits = fixed.clamp_min(1e-6).log() + delta
    prior = torch.softmax(logits, dim=-1)
    return prior, fixed.detach()


def _build_guided_objectness(
    model: torch.nn.Module,
    query: torch.Tensor,
    config: GuidedPrototypeConfig,
    image: torch.Tensor | None,
) -> torch.Tensor:
    """Object score decoupled from the three normal semantic memory groups."""

    physical_prior = _build_priors(query, config, image)
    physical = physical_prior[..., 2]
    vlm_sidecar = getattr(model, "_guided_vlm_prior_sidecar", None)
    if vlm_sidecar is None:
        return physical
    from .vlm_semantic_prior import (
        correct_physical_prior_with_vlm,
        fuse_vlm_objectness,
        predict_vlm_objectness,
        predict_vlm_prior,
    )

    if config.vlm_prior_fusion_mode == "low_confidence_correction":
        vlm_prior = predict_vlm_prior(vlm_sidecar, query.detach(), image)
        corrected = correct_physical_prior_with_vlm(
            vlm_prior,
            physical_prior,
            max_weight=config.vlm_prior_correction_max_weight,
            physical_margin_threshold=config.vlm_prior_correction_physical_margin,
            vlm_margin_threshold=config.vlm_prior_correction_vlm_margin,
        )
        return corrected[..., 2]

    learned_novelty = predict_vlm_objectness(vlm_sidecar, query.detach(), image)
    return fuse_vlm_objectness(
        learned_novelty,
        physical,
        config.vlm_prior_physical_weight,
    )


def _prior_anchor_loss(prior: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
    return (fixed * (fixed.clamp_min(1e-6).log() - prior.clamp_min(1e-6).log())).sum(dim=-1).mean()


def _residual_tokens(en: Sequence[torch.Tensor], de: Sequence[torch.Tensor], side: int) -> torch.Tensor:
    residuals = []
    for target, pred in zip(en, de):
        residual = (1.0 - F.cosine_similarity(target.detach(), pred, dim=1)).clamp_min(0.0).unsqueeze(1)
        if residual.shape[-2:] != (side, side):
            residual = F.interpolate(residual, size=(side, side), mode="bilinear", align_corners=False)
        residuals.append(residual.flatten(1))
    if not residuals:
        raise ValueError("No feature pairs were provided for guided normal guard.")
    return torch.stack(residuals, dim=0).mean(dim=0)


def guided_prototype_extra_loss(
    model: torch.nn.Module,
    en: Sequence[torch.Tensor],
    de: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    config = getattr(model, "_guided_prototype_config", None)
    state = getattr(model, "_guided_last_prior_state", None)
    if config is None or state is None:
        device = en[0].device if en else torch.device("cpu")
        return torch.tensor(0.0, device=device), {}

    prior = state["prior"]
    fixed = state["fixed"].to(device=prior.device, dtype=prior.dtype)
    loss = prior.new_tensor(0.0)
    diag: dict[str, float] = {}

    if config.trainable_prior and config.prior_anchor_weight > 0.0:
        anchor = _prior_anchor_loss(prior, fixed)
        loss = loss + float(config.prior_anchor_weight) * anchor
        diag["guided_prior_anchor"] = float(anchor.detach().cpu())

    if config.trainable_prior and config.normal_guard_weight > 0.0:
        n = prior.shape[1]
        side = int(math.sqrt(n))
        if side * side == n:
            texture = fixed[:, :, 1]
            k = max(1, min(n, int(round(n * float(config.normal_guard_top_frac)))))
            indices = texture.topk(k, dim=1).indices
            mask = torch.zeros_like(texture)
            mask.scatter_(1, indices, 1.0)
            denom = mask.sum(dim=1).clamp_min(1.0)
            residual = _residual_tokens(en, de, side=side)
            rec_guard = (residual * mask).sum(dim=1).div(denom).mean()
            obj_guard = (prior[:, :, 2] * mask).sum(dim=1).div(denom).mean()
            guard = rec_guard + float(config.normal_guard_beta) * obj_guard
            loss = loss + float(config.normal_guard_weight) * guard
            diag["guided_normal_guard"] = float(guard.detach().cpu())
            diag["guided_guard_rec"] = float(rec_guard.detach().cpu())
            diag["guided_guard_obj"] = float(obj_guard.detach().cpu())

    return loss, diag


def _group_slices(groups: tuple[int, int, int]) -> tuple[slice, slice, slice]:
    bg, texture, obj = groups
    return (slice(0, bg), slice(bg, bg + texture), slice(bg + texture, bg + texture + obj))


def _prototype_separation(keys: torch.Tensor, groups: tuple[int, int, int], margin: float) -> torch.Tensor:
    slices = _group_slices(groups)
    means = [F.normalize(keys[:, group_slice, :].mean(dim=1), dim=-1) for group_slice in slices]
    losses = []
    for idx in range(len(means)):
        for jdx in range(idx + 1, len(means)):
            distance = 1.0 - (means[idx] * means[jdx]).sum(dim=-1)
            losses.append(F.relu(float(margin) - distance).mean())
    return torch.stack(losses).mean() if losses else keys.new_tensor(0.0)


def _group_contrastive_loss(
    distribution: torch.Tensor,
    priors: torch.Tensor,
    groups: tuple[int, int, int],
    temperature: float,
    confidence_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Class-balanced group contrastive loss for confident fixed-prior assignments."""

    similarity = 1.0 - distribution
    logits = torch.stack(
        [similarity[:, :, group_slice].max(dim=2).values for group_slice in _group_slices(groups)],
        dim=2,
    )
    top = priors.topk(2, dim=2)
    target = top.indices[:, :, 0]
    confidence = top.values[:, :, 0] - top.values[:, :, 1]
    valid = confidence >= float(confidence_margin)
    token_loss = F.cross_entropy(
        (logits / float(temperature)).reshape(-1, 3),
        target.reshape(-1),
        reduction="none",
    ).reshape_as(target)

    group_losses = []
    for group_index in range(3):
        selected = valid & (target == group_index)
        weights = confidence * selected.to(confidence.dtype)
        if bool(selected.any()):
            group_losses.append((token_loss * weights).sum() / weights.sum().clamp_min(1e-6))
    loss = torch.stack(group_losses).mean() if group_losses else distribution.sum() * 0.0
    return loss, target, confidence, valid


def _intra_group_balance_loss(
    distribution: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    valid: torch.Tensor,
    groups: tuple[int, int, int],
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Balance confident tokens across prototypes without forcing per-image 50/50 usage."""

    similarity = 1.0 - distribution
    losses = []
    minimum_usage = []
    for group_index, group_slice in enumerate(_group_slices(groups)):
        group_size = int(group_slice.stop - group_slice.start)
        if group_size <= 1:
            continue
        selected = valid & (target == group_index)
        if not bool(selected.any()):
            continue
        assignment = F.softmax(
            similarity[:, :, group_slice] / float(temperature),
            dim=-1,
        )
        token_weight = (confidence * selected.to(confidence.dtype)).unsqueeze(-1)
        mean_assignment = (assignment * token_weight).sum(dim=(0, 1))
        mean_assignment = mean_assignment / token_weight.sum().clamp_min(1e-6)
        mean_assignment = mean_assignment.clamp_min(1e-8)
        uniform = 1.0 / float(group_size)
        losses.append((mean_assignment * (mean_assignment / uniform).log()).sum())
        minimum_usage.append(mean_assignment.min())
    zero = distribution.sum() * 0.0
    loss = torch.stack(losses).mean() if losses else zero
    min_usage = torch.stack(minimum_usage).mean() if minimum_usage else zero.detach()
    return loss, min_usage


def _intra_group_repulsion_loss(
    keys: torch.Tensor,
    groups: tuple[int, int, int],
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize overly similar image-conditioned prototypes within each group."""

    normalized = F.normalize(keys, dim=-1)
    losses = []
    similarities = []
    for group_slice in _group_slices(groups):
        group = normalized[:, group_slice, :]
        count = int(group.shape[1])
        if count <= 1:
            continue
        cosine = group @ group.transpose(1, 2)
        upper = torch.triu(
            torch.ones(count, count, dtype=torch.bool, device=keys.device),
            diagonal=1,
        )
        pair_cosine = cosine[:, upper]
        losses.append(F.relu(pair_cosine - float(margin)).mean())
        similarities.append(pair_cosine.mean())
    zero = keys.sum() * 0.0
    loss = torch.stack(losses).mean() if losses else zero
    similarity = torch.stack(similarities).mean() if similarities else zero.detach()
    return loss, similarity


def _memory_mode_supervision_loss(
    query: torch.Tensor,
    distribution: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    valid: torch.Tensor,
    teacher: FrozenMemoryModeTeacher,
    groups: tuple[int, int, int],
    temperature: float,
    *,
    margin_weighting: bool = False,
    soft_targets: bool = False,
    teacher_temperature: float = 0.10,
    semantic_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Teach each within-group prototype a distinct P1 normal-memory mode.

    Losses are averaged first per observed mode and then per semantic group, so
    a frequent normal mode cannot erase a rarer one.
    """

    if tuple(teacher.groups) != tuple(groups):
        raise ValueError(f"Teacher groups {teacher.groups} do not match model groups {groups}.")
    if semantic_weights is not None and semantic_weights.shape != (*query.shape[:2], 3):
        raise ValueError("Soft semantic mode weights must have shape [batch,tokens,3].")
    similarity = 1.0 - distribution
    normalized_query = F.normalize(query.float(), dim=-1)
    centers = teacher.centers.to(normalized_query)
    group_losses = []
    group_accuracies = []
    group_min_usage = []
    for group_index, group_slice in enumerate(_group_slices(groups)):
        if semantic_weights is None:
            semantic_weight = confidence * (valid & (target == group_index)).to(
                confidence.dtype
            )
        else:
            semantic_weight = semantic_weights[:, :, group_index].to(confidence.dtype)
        selected = semantic_weight > 0.0
        if not bool(selected.any()):
            continue
        teacher_logits = normalized_query[:, :, :] @ centers[group_slice].T
        teacher_label = teacher_logits.argmax(dim=-1)
        student_logits = similarity[:, :, group_slice] / float(temperature)
        if soft_targets:
            teacher_probability = F.softmax(
                teacher_logits / float(teacher_temperature), dim=-1
            ).detach()
            token_loss = -(
                teacher_probability * F.log_softmax(student_logits, dim=-1)
            ).sum(dim=-1)
        else:
            token_loss = F.cross_entropy(
                student_logits.reshape(-1, student_logits.shape[-1]),
                teacher_label.reshape(-1),
                reduction="none",
            ).reshape_as(teacher_label)
        mode_confidence = torch.ones_like(token_loss)
        if margin_weighting:
            if not teacher.has_confidence_calibration:
                raise ValueError("Mode-margin weighting requires calibrated teacher metadata.")
            top_two = teacher_logits.topk(min(2, teacher_logits.shape[-1]), dim=-1).values
            if teacher_logits.shape[-1] == 1:
                teacher_margin = torch.ones_like(top_two[..., 0])
            else:
                teacher_margin = top_two[..., 0] - top_two[..., 1]
            floor = teacher.margin_floor[group_index].to(teacher_margin)
            scale = teacher.margin_scale[group_index].to(teacher_margin)
            mode_confidence = (
                (teacher_margin - floor) / (scale - floor).clamp_min(1e-6)
            ).clamp(0.0, 1.0)
            mode_confidence = mode_confidence * teacher.group_reliability[group_index].to(
                mode_confidence
            )
        mode_losses = []
        usage = []
        for mode_index in range(int(group_slice.stop - group_slice.start)):
            mode_selected = selected & (teacher_label == mode_index)
            usage.append((semantic_weight * mode_selected.to(semantic_weight.dtype)).sum())
            if bool(mode_selected.any()):
                weights = (
                    semantic_weight
                    * mode_confidence
                    * mode_selected.to(confidence.dtype)
                )
                if bool((weights.sum() > 0).item()):
                    mode_losses.append(
                        (token_loss * weights).sum() / weights.sum().clamp_min(1e-6)
                    )
        if mode_losses:
            group_losses.append(torch.stack(mode_losses).mean())
        prediction = student_logits.argmax(dim=-1)
        correct = (prediction == teacher_label).to(semantic_weight.dtype)
        group_accuracies.append(
            (correct * semantic_weight).sum() / semantic_weight.sum().clamp_min(1e-6)
        )
        usage_tensor = torch.stack(usage)
        group_min_usage.append(usage_tensor.min() / usage_tensor.sum().clamp_min(1.0))
    zero = distribution.sum() * 0.0
    loss = torch.stack(group_losses).mean() if group_losses else zero
    accuracy = torch.stack(group_accuracies).mean() if group_accuracies else zero.detach()
    min_usage = torch.stack(group_min_usage).mean() if group_min_usage else zero.detach()
    return loss, accuracy, min_usage


def _center6_teacher_probability(
    query: torch.Tensor,
    teacher: FrozenCenter6Teacher,
    *,
    temperature: float,
    hierarchical: bool = False,
) -> torch.Tensor:
    """Return the frozen radius-normalized soft six-center prior."""

    _, probability = _center6_teacher_distance_probability(
        query,
        teacher,
        temperature=temperature,
        hierarchical=hierarchical,
    )
    return probability


def _center6_teacher_distance_probability(
    query: torch.Tensor,
    teacher: FrozenCenter6Teacher,
    *,
    temperature: float,
    hierarchical: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return radius-normalized distances and their soft Center6 probabilities."""

    if query.ndim != 3:
        raise ValueError("Center6 query must have shape [batch,tokens,channels].")
    if temperature <= 0.0:
        raise ValueError("Center6 teacher temperature must be positive.")
    normalized_query = F.normalize(query.float(), dim=-1)
    centers = teacher.centers.to(normalized_query)
    radii = teacher.radii.to(normalized_query).clamp_min(1e-8)
    teacher_distance = (1.0 - normalized_query @ centers.T).clamp_min(0.0)
    normalized_distance = teacher_distance / radii
    teacher_logits = -normalized_distance / float(temperature)
    if not hierarchical:
        probability = F.softmax(teacher_logits, dim=-1)
    else:
        group_energies = []
        conditional_probabilities = []
        for group_slice in _group_slices(teacher.mode_groups):
            group_logits = teacher_logits[:, :, group_slice]
            mode_count = int(group_slice.stop - group_slice.start)
            conditional_probabilities.append(F.softmax(group_logits, dim=-1))
            group_energies.append(
                -float(temperature)
                * (
                    torch.logsumexp(group_logits, dim=-1)
                    - math.log(float(mode_count))
                )
            )
        group_probability = F.softmax(
            -torch.stack(group_energies, dim=-1) / float(temperature),
            dim=-1,
        )
        probability = torch.cat(
            [
                group_probability[:, :, group_index : group_index + 1]
                * conditional
                for group_index, conditional in enumerate(conditional_probabilities)
            ],
            dim=-1,
        )
    return normalized_distance.detach(), probability.detach()


def _center6_group_priors(
    query: torch.Tensor,
    teacher: FrozenCenter6Teacher,
    groups: tuple[int, int, int],
    *,
    temperature: float,
    mapping: str = "soft",
    hierarchical: bool = False,
) -> torch.Tensor:
    """Aggregate the six-center teacher prior into semantic group probabilities."""

    if tuple(groups) != tuple(teacher.groups):
        raise ValueError(
            f"Center6 prior groups {tuple(groups)} do not match teacher groups {teacher.groups}."
        )
    if mapping == "hard":
        normalized_distance, _ = _center6_teacher_distance_probability(
            query,
            teacher,
            temperature=temperature,
            hierarchical=hierarchical,
        )
        probability = F.one_hot(
            normalized_distance.argmin(dim=-1),
            num_classes=int(teacher.centers.shape[0]),
        ).to(dtype=query.dtype)
    elif mapping == "soft":
        probability = _center6_teacher_probability(
            query,
            teacher,
            temperature=temperature,
            hierarchical=hierarchical,
        )
    else:
        raise ValueError(f"Unknown Center6 prior mapping: {mapping!r}.")
    mode_group_ids = teacher.mode_group_ids.to(probability.device)
    return torch.stack(
        [
            probability[:, :, mode_group_ids == group_index].sum(dim=-1)
            for group_index in range(len(groups))
        ],
        dim=-1,
    )


def _center6_mode_values_to_slots(
    values: torch.Tensor,
    teacher: FrozenCenter6Teacher,
) -> torch.Tensor:
    """Expand one value per effective Normal mode to physical prototype slots."""

    if values.shape[-1] != teacher.centers.shape[0]:
        raise ValueError("Center mode values do not match the teacher mode count.")
    return values.index_select(-1, teacher.slot_to_mode.to(values.device))


def _build_decoder_read_risk(
    model: torch.nn.Module,
    query: torch.Tensor,
    images: torch.Tensor | None,
    config: GuidedPrototypeConfig,
    novelty: torch.Tensor,
    *,
    aggregation_priors: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the inference-only decoder risk without changing prototype supervision.

    The returned risk is normally ``[B,N]``.  Center6 prototype-wise reads return
    ``[B,N,P]`` so each normal mode can be attenuated independently.  The second
    return value is the scalar objectness used for diagnostics.
    """

    if novelty.shape != query.shape[:2]:
        raise ValueError(
            f"Decoder novelty must have shape {tuple(query.shape[:2])}, got "
            f"{tuple(novelty.shape)}."
        )
    source = config.decoder_read_risk_source
    if source == "aggregation":
        if aggregation_priors is None:
            if config.center6_balanced:
                teacher = getattr(model, "guided_center6_teacher", None)
                if not isinstance(teacher, FrozenCenter6Teacher):
                    raise RuntimeError("Center6 decoder risk requires its frozen teacher.")
                aggregation_priors = _center6_group_priors(
                    query,
                    teacher,
                    config.groups,
                    temperature=config.center6_teacher_temperature,
                    mapping=config.center6_prior_mapping,
                )
            else:
                aggregation_priors, _ = _build_trainable_priors(
                    model,
                    query,
                    config,
                    images,
                )
        objectness = aggregation_priors[:, :, 2].detach()
        return (objectness * novelty).detach(), objectness

    if source in {"center6_radius", "center6_global", "center6_mode_novelty"}:
        if not config.center6_balanced:
            raise ValueError("Center6-native decoder risk requires Center6 supervision.")
        teacher = getattr(model, "guided_center6_teacher", None)
        if not isinstance(teacher, FrozenCenter6Teacher):
            raise RuntimeError("Center6-native decoder risk requires its frozen teacher.")
        normalized_distance, probability = _center6_teacher_distance_probability(
            query,
            teacher,
            temperature=config.center6_teacher_temperature,
        )
        support_temperature = float(config.decoder_read_center6_support_temperature)
        support_violation = torch.sigmoid(
            (normalized_distance - 1.0) / support_temperature
        )
        object_modes = teacher.mode_group_ids.to(probability.device) == 2
        objectness = probability[:, :, object_modes].sum(dim=-1)
        if source == "center6_mode_novelty":
            nearest_mode = normalized_distance.argmin(dim=-1)
            slot_thresholds = novelty.new_tensor(
                config.decoder_read_center6_mode_tail_thresholds
            )
            slot_uppers = novelty.new_tensor(config.decoder_read_center6_mode_tail_uppers)
            mode_count = int(teacher.centers.shape[0])
            thresholds = novelty.new_zeros(mode_count)
            uppers = novelty.new_zeros(mode_count)
            slot_to_mode = teacher.slot_to_mode.to(novelty.device)
            for mode_index in range(mode_count):
                selected_slots = slot_to_mode == mode_index
                thresholds[mode_index] = slot_thresholds[selected_slots].mean()
                uppers[mode_index] = slot_uppers[selected_slots].mean()
            selected_threshold = thresholds[nearest_mode]
            selected_upper = uppers[nearest_mode]
            calibrated = (
                (novelty - selected_threshold)
                / (selected_upper - selected_threshold).clamp_min(1e-8)
            ).clamp(0.0, 1.0)
            return calibrated.detach(), objectness.detach()
        if source == "center6_global":
            nearest_violation = torch.sigmoid(
                (normalized_distance.min(dim=-1).values - 1.0) / support_temperature
            )
            alpha = float(config.decoder_read_center6_support_alpha)
            risk = novelty * (alpha + (1.0 - alpha) * nearest_violation)
            return risk.detach(), objectness.detach()
        # A Center6-native read risk: Atlas/parent-memory novelty answers whether
        # the token is unfamiliar, while each absolute radius violation answers
        # which of the six corresponding normal prototypes should not be read.
        # Relative six-way probabilities and the legacy physical prior are
        # deliberately excluded from this ablation.
        risk = novelty.unsqueeze(-1) * _center6_mode_values_to_slots(
            support_violation, teacher
        )
        return risk.detach(), objectness.detach()

    normalizer = getattr(model, "guided_prior_normalizer", None)
    physical_priors = _build_priors(
        query,
        config,
        images,
        normalizer=normalizer,
        update_stats=False,
    )
    physical_objectness = physical_priors[:, :, 2].detach()
    base_risk = physical_objectness * novelty
    if source == "physical":
        return base_risk.detach(), physical_objectness
    if source != "center6_hybrid":
        raise ValueError(f"Unknown decoder read risk source: {source!r}.")
    if not config.center6_balanced:
        raise ValueError("Center6 hybrid decoder risk requires Center6 supervision.")
    teacher = getattr(model, "guided_center6_teacher", None)
    if not isinstance(teacher, FrozenCenter6Teacher):
        raise RuntimeError("Center6 hybrid decoder risk requires its frozen teacher.")
    normalized_distance, probability = _center6_teacher_distance_probability(
        query,
        teacher,
        temperature=config.center6_teacher_temperature,
    )
    support_temperature = float(config.decoder_read_center6_support_temperature)
    support_violation = torch.sigmoid((normalized_distance - 1.0) / support_temperature)
    alpha = float(config.decoder_read_center6_support_alpha)
    if not config.decoder_read_prototypewise:
        nearest_violation = torch.sigmoid(
            (normalized_distance.min(dim=-1).values - 1.0) / support_temperature
        )
        risk = base_risk * (alpha + (1.0 - alpha) * nearest_violation)
        return risk.detach(), physical_objectness

    prototype_count = sum(config.groups)
    prototype_risk = base_risk.new_zeros((*base_risk.shape, prototype_count))
    object_slice = _group_slices(config.groups)[2]
    slot_probability = _center6_mode_values_to_slots(probability, teacher)
    slot_support_violation = _center6_mode_values_to_slots(support_violation, teacher)
    object_probability = slot_probability[:, :, object_slice]
    responsibility = object_probability / object_probability.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(object_probability.dtype).eps
    )
    responsibility_floor = float(config.decoder_read_prototype_responsibility_floor)
    responsibility_factor = responsibility_floor + (
        1.0 - responsibility_floor
    ) * responsibility
    mode_factor = alpha + (1.0 - alpha) * slot_support_violation[:, :, object_slice]
    prototype_risk[:, :, object_slice] = (
        base_risk.unsqueeze(-1) * mode_factor * responsibility_factor
    )
    return prototype_risk.detach(), physical_objectness


def _center6_balanced_loss(
    query: torch.Tensor,
    keys: torch.Tensor,
    teacher: FrozenCenter6Teacher,
    *,
    student_temperature: float,
    teacher_temperature: float,
    teacher_mapping: str = "soft",
    reduction: str = "equal_mode",
    hierarchical_teacher: bool = False,
    novelty_veto: bool = False,
    novelty_threshold: float = 1.0,
    novelty_temperature: float = 0.10,
    novelty_min_weight: float = 0.05,
    hierarchical_reliability: bool = False,
    collapsed_diversity_weight: float = 0.0,
    collapsed_diversity_margin: float = 0.90,
    valid_mask: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Match tokens to radius-calibrated effective Normal modes.

    The teacher may contain fewer modes than the six physical prototype slots.
    Student slot probabilities that share one parent mode are summed before KL,
    so unused capacity does not create fake Normal modes. ``equal_mode`` first
    averages inside each observed hard teacher mode and then across modes;
    ``token_mean`` preserves natural occupancy; ``sqrt_balanced`` gives a mode
    total weight proportional to the square root of its occupancy.
    """

    if query.ndim != 3 or keys.ndim != 3:
        raise ValueError("Center6 query and keys must have shape [batch,tokens,channels].")
    if keys.shape[1] != teacher.slot_to_mode.numel() or query.shape[0] != keys.shape[0]:
        raise ValueError("Center6 query, prototype and teacher dimensions are incompatible.")
    if reduction not in {"equal_mode", "token_mean", "sqrt_balanced"}:
        raise ValueError(f"Unsupported Center6 reduction: {reduction!r}")
    if collapsed_diversity_weight < 0.0 or not -1.0 <= collapsed_diversity_margin <= 1.0:
        raise ValueError("Collapsed-slot diversity settings are invalid.")
    if novelty_threshold <= 0.0 or novelty_temperature <= 0.0:
        raise ValueError("Center6 novelty threshold and temperature must be positive.")
    if not 0.0 <= novelty_min_weight <= 1.0:
        raise ValueError("Center6 novelty minimum weight must be in [0,1].")
    if valid_mask is None:
        valid_mask = torch.ones(
            query.shape[:2], dtype=torch.bool, device=query.device
        )
    elif valid_mask.shape != query.shape[:2]:
        raise ValueError("Center6 valid mask must have shape [B,N].")
    else:
        valid_mask = valid_mask.to(device=query.device, dtype=torch.bool)
    if not bool(valid_mask.any()):
        raise ValueError("Center6 loss received an empty strict ROI mask.")
    normalized_query = F.normalize(query.float(), dim=-1)
    normalized_keys = F.normalize(keys.float(), dim=-1)
    centers = teacher.centers.to(normalized_query)
    radii = teacher.radii.to(normalized_query).clamp_min(1e-8)
    normalized_distance = (
        1.0 - normalized_query @ centers.T
    ).clamp_min(0.0) / radii

    if teacher_mapping == "hard":
        teacher_hard_assignment = normalized_distance.argmin(dim=-1)
        teacher_probability = F.one_hot(
            teacher_hard_assignment,
            num_classes=int(teacher.centers.shape[0]),
        ).to(dtype=normalized_query.dtype)
    elif teacher_mapping == "soft":
        teacher_probability = _center6_teacher_probability(
            query,
            teacher,
            temperature=teacher_temperature,
            hierarchical=hierarchical_teacher,
        )
    else:
        raise ValueError(f"Unknown Center6 teacher mapping: {teacher_mapping!r}.")
    student_logits = (
        normalized_query @ normalized_keys.transpose(1, 2)
    ) / float(student_temperature)
    student_log_probability = F.log_softmax(student_logits, dim=-1)
    student_probability = student_log_probability.exp()
    mode_count = int(teacher.centers.shape[0])
    slot_to_mode = teacher.slot_to_mode.to(student_probability.device)
    student_mode_probability = student_probability.new_zeros(
        (*student_probability.shape[:-1], mode_count)
    )
    student_mode_probability.scatter_add_(
        -1,
        slot_to_mode.view(1, 1, -1).expand_as(student_probability),
        student_probability,
    )
    probability_floor = torch.finfo(student_mode_probability.dtype).tiny
    student_mode_log_probability = student_mode_probability.clamp_min(
        probability_floor
    ).log()
    if teacher_mapping == "hard":
        group_token_loss = token_loss = -student_mode_log_probability.gather(
            -1,
            teacher_hard_assignment.unsqueeze(-1),
        ).squeeze(-1)
    else:
        group_token_loss = token_loss = F.kl_div(
            student_mode_log_probability,
            teacher_probability,
            reduction="none",
        ).sum(dim=-1)
    conditional_token_loss = torch.zeros_like(token_loss)
    if hierarchical_reliability:
        if mode_count != keys.shape[1] or not torch.equal(
            teacher.slot_to_mode,
            torch.arange(mode_count, device=teacher.slot_to_mode.device),
        ):
            raise ValueError(
                "Hierarchical Center6 reliability requires one physical slot per candidate mode."
            )
        mode_group_ids = teacher.mode_group_ids.to(teacher_probability.device)
        group_count = len(teacher.mode_groups)
        teacher_group_probability = teacher_probability.new_zeros(
            (*teacher_probability.shape[:-1], group_count)
        )
        student_group_probability = student_mode_probability.new_zeros(
            (*student_mode_probability.shape[:-1], group_count)
        )
        expanded_groups = mode_group_ids.view(1, 1, -1).expand_as(teacher_probability)
        teacher_group_probability.scatter_add_(
            -1, expanded_groups, teacher_probability
        )
        student_group_probability.scatter_add_(
            -1, expanded_groups, student_mode_probability
        )
        group_token_loss = F.kl_div(
            student_group_probability.clamp_min(probability_floor).log(),
            teacher_group_probability,
            reduction="none",
        ).sum(dim=-1)
        reliability = teacher.group_reliability.to(teacher_probability)
        for group_index in range(group_count):
            selected_modes = mode_group_ids == group_index
            teacher_group_mass = teacher_group_probability[:, :, group_index]
            student_group_mass = student_group_probability[:, :, group_index]
            teacher_conditional = teacher_probability[:, :, selected_modes] / (
                teacher_group_mass.unsqueeze(-1).clamp_min(probability_floor)
            )
            student_conditional = student_mode_probability[:, :, selected_modes] / (
                student_group_mass.unsqueeze(-1).clamp_min(probability_floor)
            )
            conditional_kl = F.kl_div(
                student_conditional.clamp_min(probability_floor).log(),
                teacher_conditional,
                reduction="none",
            ).sum(dim=-1)
            conditional_token_loss = conditional_token_loss + (
                teacher_group_mass * reliability[group_index] * conditional_kl
            )
        token_loss = group_token_loss + conditional_token_loss

    nearest_normal_distance = normalized_distance.min(dim=-1).values.detach()
    if novelty_veto:
        support_weight = torch.sigmoid(
            (float(novelty_threshold) - nearest_normal_distance)
            / float(novelty_temperature)
        )
        support_weight = float(novelty_min_weight) + (
            1.0 - float(novelty_min_weight)
        ) * support_weight
    else:
        support_weight = torch.ones_like(nearest_normal_distance)

    teacher_hard = teacher_probability.argmax(dim=-1)
    student_hard = student_mode_probability.argmax(dim=-1)
    student_slot_hard = student_probability.argmax(dim=-1)
    mode_losses = []
    mode_counts = []
    mode_loss_values = []
    teacher_hard_usage = []
    student_hard_usage = []
    token_count = int(valid_mask.sum().item())
    for mode_index in range(teacher.centers.shape[0]):
        selected = (teacher_hard == mode_index) & valid_mask
        if bool(selected.any()):
            selected_weight = support_weight[selected]
            mode_loss = (token_loss[selected] * selected_weight).sum() / (
                selected_weight.sum().clamp_min(1e-8)
            )
            mode_losses.append(mode_loss)
            mode_counts.append(selected_weight.sum().to(token_loss.dtype))
            mode_loss_values.append(mode_loss.detach())
        else:
            mode_loss_values.append(token_loss.new_tensor(0.0))
        teacher_hard_usage.append(selected.sum().to(token_loss.dtype) / float(token_count))
        student_hard_usage.append(
            ((student_hard == mode_index) & valid_mask).sum().to(token_loss.dtype)
            / float(token_count)
        )
    zero = token_loss.sum() * 0.0
    mode_weight_values = token_loss.new_zeros(teacher.centers.shape[0])
    if mode_losses:
        stacked_mode_losses = torch.stack(mode_losses)
        stacked_mode_counts = torch.stack(mode_counts)
        if reduction == "equal_mode":
            active_weights = torch.ones_like(stacked_mode_counts)
        elif reduction == "sqrt_balanced":
            active_weights = stacked_mode_counts.sqrt()
        else:
            active_weights = stacked_mode_counts
        active_weights = active_weights / active_weights.sum().clamp_min(1.0)
        # With token_mean, the mode-count-weighted mean of per-mode means is
        # exactly the ordinary token mean. Keep per-mode values only as
        # diagnostics and use the direct expression for the active Raw path.
        loss = (
            (token_loss * support_weight * valid_mask).sum()
            / (support_weight * valid_mask).sum().clamp_min(1e-8)
            if reduction == "token_mean"
            else (stacked_mode_losses * active_weights).sum()
        )
        active_index = 0
        for mode_index in range(teacher.centers.shape[0]):
            if bool(((teacher_hard == mode_index) & valid_mask).any()):
                mode_weight_values[mode_index] = active_weights[active_index]
                active_index += 1
    else:
        loss = zero
    teacher_usage = torch.stack(teacher_hard_usage)
    student_usage = torch.stack(student_hard_usage)
    student_slot_usage = torch.stack(
        [
            ((student_slot_hard == slot_index) & valid_mask).sum().to(token_loss.dtype)
            / float(token_count)
            for slot_index in range(student_probability.shape[-1])
        ]
    )

    eps = torch.finfo(student_probability.dtype).tiny
    teacher_entropy_map = -(
        teacher_probability * teacher_probability.clamp_min(eps).log()
    ).sum(dim=-1)
    student_entropy_map = -(
        student_mode_probability * student_mode_probability.clamp_min(eps).log()
    ).sum(dim=-1)
    student_slot_entropy_map = -(
        student_probability * student_probability.clamp_min(eps).log()
    ).sum(dim=-1)
    teacher_entropy = teacher_entropy_map[valid_mask].mean()
    student_entropy = student_entropy_map[valid_mask].mean()
    student_slot_entropy = student_slot_entropy_map[valid_mask].mean()
    agreement = (teacher_hard[valid_mask] == student_hard[valid_mask]).float().mean()

    prototype_cosine = normalized_keys @ normalized_keys.transpose(1, 2)
    prototype_count = int(keys.shape[1])
    upper = torch.triu(
        torch.ones(
            prototype_count, prototype_count, dtype=torch.bool, device=keys.device
        ),
        diagonal=1,
    )
    pair_cosine = prototype_cosine[:, upper]
    shared_parent = upper & (
        teacher.slot_to_mode.to(keys.device).unsqueeze(0)
        == teacher.slot_to_mode.to(keys.device).unsqueeze(1)
    )
    if bool(shared_parent.any()):
        shared_parent_cosine = prototype_cosine[:, shared_parent]
        shared_parent_cosine_mean = shared_parent_cosine.mean()
        shared_parent_cosine_max = shared_parent_cosine.max()
        shared_parent_pair_count = token_loss.new_tensor(
            float(shared_parent.sum())
        )
        collapsed_diversity_loss = F.relu(
            shared_parent_cosine - float(collapsed_diversity_margin)
        ).mean()
    else:
        shared_parent_cosine_mean = token_loss.new_tensor(0.0)
        shared_parent_cosine_max = token_loss.new_tensor(0.0)
        shared_parent_pair_count = token_loss.new_tensor(0.0)
        collapsed_diversity_loss = keys.sum() * 0.0
    loss = loss + float(collapsed_diversity_weight) * collapsed_diversity_loss
    slot_centers = centers.index_select(0, teacher.slot_to_mode.to(centers.device))
    alignment = (normalized_keys * slot_centers.unsqueeze(0)).sum(dim=-1)

    diagnostics = {
        "loss": loss.detach(),
        "accuracy": agreement.detach(),
        "teacher_entropy": teacher_entropy.detach(),
        "student_entropy": student_entropy.detach(),
        "student_slot_entropy": student_slot_entropy.detach(),
        "effective_mode_count": token_loss.new_tensor(float(mode_count)).detach(),
        "prototype_slot_count": token_loss.new_tensor(float(prototype_count)).detach(),
        "teacher_confidence": teacher_probability.max(dim=-1).values[valid_mask].mean().detach(),
        "normal_support_weight": support_weight[valid_mask].mean().detach(),
        "normal_support_fraction": (
            nearest_normal_distance[valid_mask] <= float(novelty_threshold)
        ).float().mean().detach(),
        "nearest_normalized_distance": nearest_normal_distance[valid_mask].mean().detach(),
        "hierarchical_teacher": token_loss.new_tensor(
            float(hierarchical_teacher)
        ).detach(),
        "teacher_min_usage": teacher_usage.min().detach(),
        "student_min_usage": student_usage.min().detach(),
        "teacher_dead_modes": (teacher_usage == 0.0).sum().to(token_loss.dtype).detach(),
        "student_dead_modes": (student_usage == 0.0).sum().to(token_loss.dtype).detach(),
        "student_dead_slots": (student_slot_usage == 0.0)
        .sum()
        .to(token_loss.dtype)
        .detach(),
        "pair_cosine_mean": pair_cosine.mean().detach(),
        "pair_cosine_max": pair_cosine.max().detach(),
        "pair_cosine_min": pair_cosine.min().detach(),
        "pair_min_distance": (1.0 - pair_cosine.max()).detach(),
        "shared_parent_cosine_mean": shared_parent_cosine_mean.detach(),
        "shared_parent_cosine_max": shared_parent_cosine_max.detach(),
        "shared_parent_pair_count": shared_parent_pair_count.detach(),
        "collapsed_diversity_loss": collapsed_diversity_loss.detach(),
        "collapsed_diversity_weight": token_loss.new_tensor(
            float(collapsed_diversity_weight)
        ).detach(),
        "alignment_mean": alignment.mean().detach(),
        "alignment_min": alignment.min().detach(),
        "radius_min": radii.min().detach(),
        "radius_mean": radii.mean().detach(),
        "radius_max": radii.max().detach(),
        "member_count_min": teacher.member_counts.min().to(token_loss.dtype).detach(),
        "hierarchical_reliability": token_loss.new_tensor(
            float(hierarchical_reliability)
        ).detach(),
        "group_loss": group_token_loss[valid_mask].mean().detach(),
        "conditional_loss": conditional_token_loss[valid_mask].mean().detach(),
        "group_reliability_mean": teacher.group_reliability.mean()
        .to(token_loss.dtype)
        .detach(),
        "group_reliability_min": teacher.group_reliability.min()
        .to(token_loss.dtype)
        .detach(),
    }
    for mode_index in range(mode_count):
        diagnostics[f"mode_loss_{mode_index}"] = mode_loss_values[mode_index]
        diagnostics[f"mode_weight_{mode_index}"] = mode_weight_values[mode_index].detach()
        diagnostics[f"teacher_usage_{mode_index}"] = teacher_usage[mode_index].detach()
        diagnostics[f"student_usage_{mode_index}"] = student_usage[mode_index].detach()
    for slot_index in range(prototype_count):
        diagnostics[f"student_slot_usage_{slot_index}"] = student_slot_usage[
            slot_index
        ].detach()
    for group_index in range(len(teacher.mode_groups)):
        diagnostics[f"group_reliability_{group_index}"] = teacher.group_reliability[
            group_index
        ].to(token_loss.dtype).detach()
    return (
        loss,
        diagnostics,
        teacher_probability,
        student_mode_probability,
        teacher_hard,
        student_hard,
    )


def _semantic_prototype_coverage_loss(
    tokens: torch.Tensor,
    prototypes: torch.Tensor,
    role_priors: torch.Tensor,
    risk: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    min_role_mass: float = 0.05,
    variant: str = "set_match",
    selected_roles: tuple[int, ...] | None = None,
    margin: float = 0.80,
    min_confidence: float = 0.0,
    max_risk: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Cover supported image-conditioned normal roles with an unordered prototype set.

    The role anchors are detached, risk-weighted summaries of the current image.
    A small exhaustive assignment is used instead of fixed slot identities: with
    three roles and six prototypes there are at most 120 injective assignments.
    Only the selected prototype vectors receive gradients from this objective.
    """

    if tokens.ndim != 3 or prototypes.ndim != 3:
        raise ValueError("Semantic coverage expects token/prototype tensors [B,N,C]/[B,P,C].")
    if tokens.shape[0] != prototypes.shape[0] or tokens.shape[2] != prototypes.shape[2]:
        raise ValueError("Semantic coverage token/prototype batch or channel mismatch.")
    if role_priors.ndim != 3 or role_priors.shape[:2] != tokens.shape[:2]:
        raise ValueError("Semantic coverage role priors must have shape [B,N,G].")
    if risk.shape != tokens.shape[:2]:
        raise ValueError("Semantic coverage risk must have shape [B,N].")
    if not 0.0 < float(min_role_mass) <= 1.0:
        raise ValueError("Semantic coverage minimum role mass must be in (0,1].")
    if variant not in {"set_match", "selective_hinge"}:
        raise ValueError(f"Unknown semantic coverage variant: {variant!r}.")
    if valid_mask is None:
        valid_mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
    elif valid_mask.shape != tokens.shape[:2]:
        raise ValueError("Semantic coverage valid mask must have shape [B,N].")

    batch, _, _ = tokens.shape
    prototype_count = int(prototypes.shape[1])
    role_count = int(role_priors.shape[2])
    if role_count > prototype_count:
        raise ValueError("Semantic coverage needs at least as many prototypes as roles.")

    if selected_roles is None:
        selected_roles = tuple(range(role_count))
    if any(role < 0 or role >= role_count for role in selected_roles):
        raise ValueError("Selected semantic role is outside the supplied priors.")
    detached_priors = role_priors.detach().clamp_min(0.0)
    detached_risk = risk.detach().clamp(0.0, 1.0)
    safe_weight = 1.0 - detached_risk
    role_selector = torch.zeros(role_count, device=tokens.device, dtype=tokens.dtype)
    role_selector[list(selected_roles)] = 1.0
    eligibility = valid_mask
    if variant == "selective_hinge":
        confidence, hard_role = detached_priors.max(dim=-1)
        eligibility = (
            eligibility
            & (confidence >= float(min_confidence))
            & (detached_risk <= float(max_risk))
        )
        assignment_mask = F.one_hot(hard_role, num_classes=role_count).to(tokens.dtype)
    else:
        assignment_mask = torch.ones_like(detached_priors)
    weights = (
        detached_priors
        * assignment_mask
        * role_selector.view(1, 1, -1)
        * safe_weight.unsqueeze(-1)
        * eligibility.to(tokens.dtype).unsqueeze(-1)
    )
    role_mass = weights.sum(dim=1)
    valid_count = valid_mask.sum(dim=1, keepdim=True).clamp_min(1).to(tokens.dtype)
    role_fraction = role_mass / valid_count
    supported = role_fraction >= float(min_role_mass)
    anchors = torch.einsum("bng,bnc->bgc", weights, tokens.detach())
    anchors = F.normalize(anchors / role_mass.clamp_min(1e-8).unsqueeze(-1), dim=-1)
    normalized_prototypes = F.normalize(prototypes, dim=-1)
    similarity = torch.einsum("bpc,bgc->bpg", normalized_prototypes, anchors)

    image_losses: list[torch.Tensor] = []
    matched_similarities: list[torch.Tensor] = []
    slot_usage = prototypes.new_zeros(prototype_count)
    active_images = 0
    for batch_index in range(batch):
        role_indices = torch.nonzero(supported[batch_index], as_tuple=False).flatten()
        active_roles = int(role_indices.numel())
        if active_roles == 0:
            continue
        active_images += 1
        if variant == "selective_hinge":
            role_similarity = similarity[batch_index].index_select(1, role_indices)
            selected_similarity, selected_slots = role_similarity.max(dim=0)
            image_losses.append(F.relu(float(margin) - selected_similarity).mean())
        else:
            assignments = torch.tensor(
                list(itertools.permutations(range(prototype_count), active_roles)),
                device=prototypes.device,
                dtype=torch.long,
            )
            expanded_roles = role_indices.unsqueeze(0).expand(assignments.shape[0], -1)
            candidate_similarity = similarity[batch_index][assignments, expanded_roles]
            best_index = candidate_similarity.detach().sum(dim=1).argmax()
            selected_slots = assignments[best_index]
            selected_similarity = similarity[
                batch_index, selected_slots, role_indices
            ]
            image_losses.append((1.0 - selected_similarity).mean())
        matched_similarities.append(selected_similarity)
        slot_usage.scatter_add_(
            0, selected_slots, torch.ones_like(selected_slots, dtype=slot_usage.dtype)
        )

    zero = prototypes.sum() * 0.0
    if image_losses:
        loss = torch.stack(image_losses).mean()
        matched = torch.cat(matched_similarities)
        matched_mean = matched.mean()
        matched_min = matched.min()
    else:
        loss = zero
        matched_mean = zero.detach()
        matched_min = zero.detach()
    usage_probability = slot_usage / slot_usage.sum().clamp_min(1.0)
    usage_entropy = -(
        usage_probability
        * usage_probability.clamp_min(torch.finfo(usage_probability.dtype).tiny).log()
    ).sum()
    if prototype_count > 1:
        usage_entropy = usage_entropy / math.log(prototype_count)

    diagnostics = {
        "loss": loss.detach(),
        "active_roles": supported.sum(dim=1).float().mean().detach(),
        "active_image_fraction": prototypes.new_tensor(
            float(active_images) / float(max(batch, 1))
        ).detach(),
        "matched_similarity_mean": matched_mean.detach(),
        "matched_similarity_min": matched_min.detach(),
        "matched_slot_fraction": (slot_usage > 0).float().mean().detach(),
        "matched_slot_entropy": usage_entropy.detach(),
    }
    for role_index in range(role_count):
        diagnostics[f"role_mass_{role_index}"] = role_fraction[:, role_index].mean().detach()
        diagnostics[f"role_active_{role_index}"] = supported[:, role_index].float().mean().detach()
    return loss, diagnostics


def _combine_guided_gather_losses(
    native_gather: torch.Tensor,
    guided_gather: torch.Tensor,
    *,
    alpha: float,
    distill_weight: float,
) -> torch.Tensor:
    if distill_weight > 0.0:
        return native_gather + float(distill_weight) * guided_gather
    return (1.0 - float(alpha)) * native_gather + float(alpha) * guided_gather


def _guided_gather_loss(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    config: GuidedPrototypeConfig = self._guided_prototype_config
    distribution = 1.0 - F.cosine_similarity(query.unsqueeze(2), keys.unsqueeze(1), dim=-1)
    self.distribution = distribution
    self.distance, self.cluster_index = torch.min(distribution, dim=2)
    strict_roi_mask = None
    if bool(getattr(self, "_guided_roi_aware_loss", False)):
        strict_roi_mask = _strict_token_roi_mask(
            getattr(self, "_guided_valid_roi_mask", None),
            batch=query.shape[0],
            token_count=query.shape[1],
        )
        if strict_roi_mask is None:
            raise RuntimeError(
                "ROI-aware guided loss requires a spatial ROI mask for every batch."
            )
        if not bool(strict_roi_mask.any()):
            raise ValueError("ROI-aware guided loss received an empty strict ROI.")
        native_gather = self.distance[strict_roi_mask].mean()
    else:
        native_gather = self.distance.mean()

    center6_diag: dict[str, torch.Tensor] = {}
    if config.center6_balanced:
        teacher = getattr(self, "guided_center6_teacher", None)
        if not isinstance(teacher, FrozenCenter6Teacher):
            raise RuntimeError("Center6 teacher is not configured on the model.")
        (
            guided_gather,
            center6_diag,
            teacher_probability,
            _student_probability,
            teacher_hard,
            _student_hard,
        ) = _center6_balanced_loss(
            query,
            keys,
            teacher,
            student_temperature=config.center6_temperature,
            teacher_temperature=config.center6_teacher_temperature,
            teacher_mapping=config.center6_loss_mapping,
            reduction=config.center6_reduction,
            hierarchical_teacher=config.center6_hierarchical_teacher,
            novelty_veto=config.center6_novelty_veto,
            novelty_threshold=config.center6_novelty_threshold,
            novelty_temperature=config.center6_novelty_temperature,
            novelty_min_weight=config.center6_novelty_min_weight,
            hierarchical_reliability=config.center6_hierarchical_reliability,
            collapsed_diversity_weight=(
                config.center6_collapsed_slot_diversity_weight
            ),
            collapsed_diversity_margin=(
                config.center6_collapsed_slot_diversity_margin
            ),
            valid_mask=strict_roi_mask,
        )
        self._guided_pending_prior_state = None
        priors = _center6_group_priors(
            query,
            teacher,
            config.groups,
            temperature=config.center6_teacher_temperature,
            mapping=config.center6_prior_mapping,
            hierarchical=config.center6_hierarchical_teacher,
        )
        fixed_priors = priors.detach()
        mode_to_group = teacher.mode_group_ids.to(teacher_hard.device)
        hard_assign = mode_to_group[teacher_hard]
        top = teacher_probability.topk(2, dim=-1).values
        confidence = top[:, :, 0] - top[:, :, 1]
        valid = torch.ones_like(hard_assign, dtype=torch.bool)
        sep = keys.new_tensor(0.0)
        intra_balance = keys.new_tensor(0.0)
        intra_min_usage = keys.new_tensor(0.0)
        intra_repulsion = keys.new_tensor(0.0)
        intra_similarity = keys.new_tensor(0.0)
    else:
        pending = getattr(self, "_guided_pending_prior_state", None)
        if (
            isinstance(pending, dict)
            and pending.get("prior") is not None
            and pending["prior"].shape[:2] == query.shape[:2]
        ):
            priors = pending["prior"]
            fixed_priors = pending["fixed"]
            self._guided_pending_prior_state = None
        elif config.free_prototypes:
            priors = query.new_full((*query.shape[:2], 3), 1.0 / 3.0)
            fixed_priors = priors.detach()
        else:
            priors, fixed_priors = _build_trainable_priors(
                self,
                query,
                config,
                getattr(self, "_guided_prototype_image", None),
            )
        if config.free_prototypes:
            guided_gather = native_gather
            sep = keys.new_tensor(0.0)
            intra_balance = keys.new_tensor(0.0)
            intra_min_usage = keys.new_tensor(0.0)
            intra_repulsion = keys.new_tensor(0.0)
            intra_similarity = keys.new_tensor(0.0)
            hard_assign = priors.argmax(dim=2)
            confidence = priors.topk(2, dim=2).values
            confidence = confidence[:, :, 0] - confidence[:, :, 1]
            valid = torch.ones_like(hard_assign, dtype=torch.bool)
        elif config.group_contrastive:
            guided_gather, hard_assign, confidence, valid = _group_contrastive_loss(
                distribution,
                priors.detach(),
                config.groups,
                config.group_temperature,
                config.group_confidence_margin,
            )
            # Separation remains a diagnostic; contrastive classification supplies the repulsive gradient.
            sep = _prototype_separation(keys, config.groups, config.separation_margin)
            intra_balance, intra_min_usage = _intra_group_balance_loss(
                distribution,
                hard_assign,
                confidence,
                valid,
                config.groups,
                config.intra_group_temperature,
            )
            intra_repulsion, intra_similarity = _intra_group_repulsion_loss(
                keys,
                config.groups,
                config.intra_group_repulsion_margin,
            )
            guided_gather = (
                guided_gather
                + float(config.intra_group_balance_weight) * intra_balance
                + float(config.intra_group_repulsion_weight) * intra_repulsion
            )
        else:
            group_distances = []
            for group_slice in _group_slices(config.groups):
                group_distances.append(distribution[:, :, group_slice].min(dim=2).values)
            group_distance = torch.stack(group_distances, dim=2)
            guided_gather = (priors * group_distance).sum(dim=2).mean()
            sep = keys.new_tensor(0.0)
            if config.separation_weight > 0.0:
                sep = _prototype_separation(keys, config.groups, config.separation_margin)
                guided_gather = guided_gather + float(config.separation_weight) * sep
            hard_assign = priors.argmax(dim=2)
            confidence = priors.topk(2, dim=2).values
            confidence = confidence[:, :, 0] - confidence[:, :, 1]
            valid = torch.ones_like(hard_assign, dtype=torch.bool)
            intra_balance = keys.new_tensor(0.0)
            intra_min_usage = keys.new_tensor(0.0)
            intra_repulsion = keys.new_tensor(0.0)
            intra_similarity = keys.new_tensor(0.0)
    self._guided_last_prior_state = {"prior": priors, "fixed": fixed_priors}

    memory_mode_loss = keys.new_tensor(0.0)
    memory_mode_accuracy = keys.new_tensor(0.0)
    memory_mode_min_usage = keys.new_tensor(0.0)
    if config.memory_mode_weight > 0.0:
        teacher = getattr(self, "guided_memory_mode_teacher", None)
        if not isinstance(teacher, FrozenMemoryModeTeacher):
            raise RuntimeError("P2 memory-mode teacher is not configured on the model.")
        semantic_mode_weights = None
        if config.memory_mode_soft_semantic:
            semantic_mode_weights = boundary_soft_semantic_weights(
                priors.detach().reshape(-1, 3),
                config.memory_mode_semantic_margin,
            ).reshape_as(priors)
        memory_mode_loss, memory_mode_accuracy, memory_mode_min_usage = (
            _memory_mode_supervision_loss(
                query,
                distribution,
                hard_assign,
                confidence,
                valid,
                teacher,
                config.groups,
                config.memory_mode_temperature,
                margin_weighting=config.memory_mode_margin_weighting,
                soft_targets=config.memory_mode_soft_targets,
                teacher_temperature=config.memory_mode_teacher_temperature,
                semantic_weights=semantic_mode_weights,
            )
        )
        guided_gather = guided_gather + float(config.memory_mode_weight) * memory_mode_loss

    semantic_coverage_loss = keys.new_tensor(0.0)
    semantic_coverage_diag: dict[str, torch.Tensor] = {}
    semantic_coverage_effective_weight = float(config.semantic_coverage_weight)
    training_step = int(getattr(self, "_guided_training_step", 0))
    if training_step > 0 and config.semantic_coverage_warmup_steps > 0:
        if training_step <= config.semantic_coverage_warmup_steps:
            semantic_coverage_effective_weight = 0.0
        elif config.semantic_coverage_ramp_steps > 0:
            ramp_progress = min(
                1.0,
                float(training_step - config.semantic_coverage_warmup_steps)
                / float(config.semantic_coverage_ramp_steps),
            )
            semantic_coverage_effective_weight *= ramp_progress
    if config.semantic_coverage_weight > 0.0:
        aggregation_risk = getattr(self, "_guided_last_aggregation_risk", None)
        if not isinstance(aggregation_risk, torch.Tensor):
            raise RuntimeError(
                "Semantic prototype coverage requires risk from the preceding aggregation."
            )
        semantic_coverage_loss, semantic_coverage_diag = (
            _semantic_prototype_coverage_loss(
                query,
                keys,
                priors,
                aggregation_risk,
                valid_mask=strict_roi_mask,
                min_role_mass=config.semantic_coverage_min_role_mass,
                variant=config.semantic_coverage_variant,
                selected_roles=config.semantic_coverage_roles,
                margin=config.semantic_coverage_margin,
                min_confidence=config.semantic_coverage_min_confidence,
                max_risk=config.semantic_coverage_max_risk,
            )
        )

    alpha = _guided_alpha(self, config)
    gather = _combine_guided_gather_losses(
        native_gather,
        guided_gather,
        alpha=alpha,
        distill_weight=config.guided_distill_weight,
    )
    gather = gather + semantic_coverage_effective_weight * semantic_coverage_loss

    with torch.no_grad():
        valid_count = valid.sum().clamp_min(1)
        aggregation_diag = getattr(self, "_guided_aggregation_diag", {}) or {}
        spatial_diag = getattr(self, "_guided_spatial_diag", {}) or {}
        semantic_groups_active = config.descriptor_variant == "off"
        self._guided_prototype_diag = {
            "guided_proto_loss": float(gather.detach().cpu()),
            "guided_native_gather": float(native_gather.detach().cpu()),
            "guided_group_gather": float(guided_gather.detach().cpu()),
            "guided_native_anchor_alpha": float(alpha),
            "guided_distill_weight": float(config.guided_distill_weight),
            "guided_semantic_coverage_weight": float(config.semantic_coverage_weight),
            "roi_aware_guided_gather": float(strict_roi_mask is not None),
            "guided_semantic_coverage_effective_weight": float(
                semantic_coverage_effective_weight
            ),
            "guided_semantic_coverage_min_role_mass": float(
                config.semantic_coverage_min_role_mass
            ),
            **{
                f"guided_semantic_coverage_{name}": float(value.detach().cpu())
                for name, value in semantic_coverage_diag.items()
            },
            "guided_proto_sep": float(sep.detach().cpu()),
            "guided_intra_balance": float(intra_balance.detach().cpu()),
            "guided_intra_min_soft_usage": float(intra_min_usage.detach().cpu()),
            "guided_intra_repulsion": float(intra_repulsion.detach().cpu()),
            "guided_intra_similarity": float(intra_similarity.detach().cpu()),
            "guided_memory_mode_loss": float(memory_mode_loss.detach().cpu()),
            "guided_memory_mode_accuracy": float(memory_mode_accuracy.detach().cpu()),
            "guided_memory_mode_min_usage": float(memory_mode_min_usage.detach().cpu()),
            "guided_prior_bg": float(priors[:, :, 0].mean().detach().cpu()),
            "guided_prior_texture": float(priors[:, :, 1].mean().detach().cpu()),
            "guided_prior_object": float(priors[:, :, 2].mean().detach().cpu()),
            "guided_fixed_bg": float(fixed_priors[:, :, 0].mean().detach().cpu()),
            "guided_fixed_texture": float(fixed_priors[:, :, 1].mean().detach().cpu()),
            "guided_fixed_object": float(fixed_priors[:, :, 2].mean().detach().cpu()),
            "guided_assign_bg": float(
                (hard_assign == 0).float().mean().detach().cpu()
            ) if semantic_groups_active else 0.0,
            "guided_assign_texture": float(
                (hard_assign == 1).float().mean().detach().cpu()
            ) if semantic_groups_active else 0.0,
            "guided_assign_object": float(
                (hard_assign == 2).float().mean().detach().cpu()
            ) if semantic_groups_active else 0.0,
            "guided_valid_ratio": float(valid.float().mean().detach().cpu()),
            "guided_valid_bg": float(
                ((hard_assign == 0) & valid).sum().div(valid_count).detach().cpu()
            ) if semantic_groups_active else 0.0,
            "guided_valid_texture": float(
                ((hard_assign == 1) & valid).sum().div(valid_count).detach().cpu()
            ) if semantic_groups_active else 0.0,
            "guided_valid_object": float(
                ((hard_assign == 2) & valid).sum().div(valid_count).detach().cpu()
            ) if semantic_groups_active else 0.0,
            "guided_confidence": float(confidence.mean().detach().cpu()),
            "guided_semantic_groups_active": float(semantic_groups_active),
            "guided_descriptor_variant_id": float(
                {"off": 0, "b": 1, "c": 2, "d": 3}[config.descriptor_variant]
            ),
            **{
                f"guided_center6_{name}": float(value.detach().cpu())
                for name, value in center6_diag.items()
            },
            **spatial_diag,
            **aggregation_diag,
        }
    return gather
