from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Tuple

import torch
import torch.nn as nn


EXTERNAL_PREFIXES = ("models", "dataset", "utils", "optimizers", "dinov1", "dinov2", "beit")


def _clear_external_modules() -> None:
    for name in list(sys.modules):
        if name in EXTERNAL_PREFIXES or name.startswith(tuple(prefix + "." for prefix in EXTERNAL_PREFIXES)):
            del sys.modules[name]


@contextmanager
def external_repo_context(repo_root: str | Path) -> Iterator[Path]:
    repo_root = Path(repo_root).resolve()
    if not repo_root.exists():
        raise FileNotFoundError(repo_root)
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    _clear_external_modules()
    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)
    try:
        yield repo_root
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


def build_dinomaly_model(
    external_repo: str | Path,
    encoder: str,
    device: torch.device,
) -> Tuple[nn.Module, nn.Module]:
    """Build the Dinomaly ViTill reconstruction model from the local Dinomaly repo."""

    with external_repo_context(external_repo):
        module = importlib.import_module("dinomaly_fod_rgb")
        model, trainable = module.build_model(encoder, device)
    return model, trainable


def build_inpformer_model(
    external_repo: str | Path,
    encoder: str,
    inp_num: int,
    device: torch.device,
) -> Tuple[nn.Module, nn.Module]:
    """Build the INP-Former prototype-guided reconstruction model."""

    with external_repo_context(external_repo):
        from functools import partial

        from dinov1.utils import trunc_normal_
        from models import vit_encoder
        from models.uad import INP_Former
        from models.vision_transformer import Aggregation_Block, Mlp, Prototype_Block

        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
        fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

        encoder_model = vit_encoder.load(encoder)
        if "small" in encoder:
            embed_dim, num_heads = 384, 6
        elif "base" in encoder:
            embed_dim, num_heads = 768, 12
        elif "large" in encoder:
            embed_dim, num_heads = 1024, 16
            target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
        else:
            raise ValueError(f"Unsupported encoder: {encoder}")

        bottleneck = nn.ModuleList([Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.0)])
        inp_tokens = nn.ParameterList([nn.Parameter(torch.randn(int(inp_num), embed_dim))])
        aggregation = nn.ModuleList(
            [
                Aggregation_Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-8),
                )
            ]
        )
        decoder = nn.ModuleList(
            [
                Prototype_Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-8),
                )
                for _ in range(8)
            ]
        )
        model = INP_Former(
            encoder=encoder_model,
            bottleneck=bottleneck,
            aggregation=aggregation,
            decoder=decoder,
            target_layers=target_layers,
            remove_class_token=True,
            fuse_layer_encoder=fuse_layer_encoder,
            fuse_layer_decoder=fuse_layer_decoder,
            prototype_token=inp_tokens,
        ).to(device)

        trainable = nn.ModuleList([bottleneck, decoder, aggregation, inp_tokens])
        for module in trainable.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.01, a=-0.03, b=0.03)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    return model, trainable


def load_stable_adamw(inpformer_repo: str | Path):
    """Load the StableAdamW implementation shipped with INP-Former."""

    with external_repo_context(inpformer_repo):
        from optimizers import StableAdamW

    return StableAdamW


def build_reconstruction_model(
    architecture: str,
    dinomaly_repo: str | Path,
    inpformer_repo: str | Path,
    encoder: str,
    device: torch.device,
    inp_num: int = 6,
    mamba_layers: int = 4,
    mamba_scan: str = "hilbert",
    mamba_d_state: int = 16,
    mamba_d_conv: int = 4,
    mamba_expand: int = 2,
    mamba_bidirectional: bool = True,
    mamba_drop: float = 0.0,
    mamba_multi_output: bool = False,
) -> Tuple[nn.Module, nn.Module]:
    if architecture == "dinomaly":
        return build_dinomaly_model(dinomaly_repo, encoder, device)
    if architecture == "inpformer":
        return build_inpformer_model(inpformer_repo, encoder, inp_num, device)
    if architecture == "mamba":
        from .mamba_recon import build_mamba_model

        return build_mamba_model(
            dinomaly_repo=dinomaly_repo,
            encoder_name=encoder,
            device=device,
            layers=mamba_layers,
            scan=mamba_scan,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            bidirectional=mamba_bidirectional,
            drop=mamba_drop,
            multi_output=mamba_multi_output,
        )
    raise ValueError(f"Unsupported architecture: {architecture}")


def select_reconstruction_trainable_modules(
    model: nn.Module,
    default_trainable: nn.Module,
    *,
    scope: str,
) -> nn.Module:
    """Restrict warm-start adaptation to an explicitly named module scope."""

    if scope == "full":
        return default_trainable
    if scope == "reconstruction_only":
        bottleneck = getattr(model, "bottleneck", None)
        decoder = getattr(model, "decoder", None)
        if not isinstance(bottleneck, nn.Module) or not isinstance(decoder, nn.Module):
            raise ValueError(
                "reconstruction_only scope requires INP-Former bottleneck and decoder modules."
            )
        model.requires_grad_(False)
        selected = nn.Module()
        selected.add_module("bottleneck", bottleneck)
        selected.add_module("decoder", decoder)
        selected.requires_grad_(True)
        return selected
    if scope == "blindspot_context":
        head = getattr(model, "blindspot_context_head", None)
        if not isinstance(head, nn.Module):
            raise ValueError(
                "blindspot_context scope requires an attached blindspot_context_head."
            )
        model.requires_grad_(False)
        head.requires_grad_(True)
        return head
    if scope != "prototype_aggregation":
        raise ValueError(f"Unsupported reconstruction trainable scope: {scope}")

    prototype_token = getattr(model, "prototype_token", None)
    aggregation = getattr(model, "aggregation", None)
    missing = []
    if not isinstance(prototype_token, (nn.Parameter, nn.Module)):
        missing.append("prototype_token")
    if not isinstance(aggregation, nn.Module):
        missing.append("aggregation")
    if missing:
        raise ValueError(
            "prototype_aggregation scope requires INP-Former trainables: "
            + ", ".join(missing)
        )

    model.requires_grad_(False)
    selected = nn.Module()
    if isinstance(prototype_token, nn.Parameter):
        selected.register_parameter("prototype_token", prototype_token)
    else:
        selected.add_module("prototype_token", prototype_token)
    selected.add_module("aggregation", aggregation)
    selected.requires_grad_(True)
    return selected


def load_checkpoint(
    model: nn.Module,
    checkpoint: str | Path,
    device: torch.device,
    strict: bool = True,
    allowed_missing_prefixes: tuple[str, ...] = (),
) -> None:
    payload = torch.load(checkpoint, map_location=device)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    state = payload
    if isinstance(payload, dict):
        if "model" in payload:
            state = payload["model"]
        elif "state_dict" in payload:
            state = payload["state_dict"]

    omitted_prefixes = tuple(metadata.get("omitted_model_state_prefixes", ()))
    allowed_prefixes = tuple(omitted_prefixes) + tuple(allowed_missing_prefixes)
    if not allowed_prefixes:
        model.load_state_dict(state, strict=strict)
        return

    incompatible = model.load_state_dict(state, strict=False)
    if not strict:
        return
    allowed_missing = {
        key
        for key in model.state_dict()
        if any(key.startswith(prefix) for prefix in allowed_prefixes)
    }
    disallowed_missing = sorted(set(incompatible.missing_keys) - allowed_missing)
    unexpected = sorted(incompatible.unexpected_keys)
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "Partial checkpoint did not match the model outside its declared omitted "
            f"prefixes={allowed_prefixes}: missing={disallowed_missing}, "
            f"unexpected={unexpected}"
        )


def checkpoint_payload(
    model: nn.Module,
    *,
    omit_model_state_prefixes: tuple[str, ...] = (),
    **metadata,
):
    state = model.state_dict()
    if omit_model_state_prefixes:
        state = {
            key: value
            for key, value in state.items()
            if not any(key.startswith(prefix) for prefix in omit_model_state_prefixes)
        }
        metadata = {
            **metadata,
            "model_state_mode": "partial",
            "omitted_model_state_prefixes": list(omit_model_state_prefixes),
        }
    return {"model": state, "metadata": metadata}
