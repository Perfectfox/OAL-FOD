from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from .masking import PerspectiveMaskConfig, _masked_tokens, _tokens_to_maps, make_spatial_mask


def _next_power_of_two(value: int) -> int:
    return 1 << max(int(value - 1).bit_length(), 0)


def _hilbert_index_to_xy(index: int, order: int) -> tuple[int, int]:
    """Map a Hilbert curve index to ``(x, y)`` on an ``order x order`` grid.

    The standard Hilbert curve is defined for power-of-two side lengths.  For
    non-power-of-two token grids we generate the next larger curve and later
    filter coordinates that fall outside the actual grid.  This keeps locality
    while avoiding special-case curve construction.
    """

    x = 0
    y = 0
    t = int(index)
    step = 1
    while step < order:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x = step - 1 - x
                y = step - 1 - y
            x, y = y, x
        x += step * rx
        y += step * ry
        t //= 4
        step *= 2
    return x, y


def _part1by1(value: int) -> int:
    value &= 0x0000FFFF
    value = (value | (value << 8)) & 0x00FF00FF
    value = (value | (value << 4)) & 0x0F0F0F0F
    value = (value | (value << 2)) & 0x33333333
    value = (value | (value << 1)) & 0x55555555
    return value


def _zorder_key(x: int, y: int) -> int:
    return _part1by1(x) | (_part1by1(y) << 1)


@lru_cache(maxsize=32)
def scan_permutation(side: int, scan: str) -> tuple[int, ...]:
    """Return flattened token indices in the selected 2D locality-preserving order.

    ``hilbert`` tends to preserve local neighborhoods better than simple raster
    order. ``zorder`` is cheaper and deterministic, and is useful as an ablation
    when checking whether the exact sequence order matters.
    """

    scan = scan.lower()
    if side <= 0:
        raise ValueError(f"side must be positive, got {side}")
    if scan == "raster":
        return tuple(range(side * side))
    if scan == "snake":
        order = []
        for y in range(side):
            xs = range(side) if y % 2 == 0 else range(side - 1, -1, -1)
            order.extend(y * side + x for x in xs)
        return tuple(order)
    if scan == "zorder":
        coords = [(x, y) for y in range(side) for x in range(side)]
        coords.sort(key=lambda xy: _zorder_key(xy[0], xy[1]))
        return tuple(y * side + x for x, y in coords)
    if scan == "hilbert":
        power = _next_power_of_two(side)
        order = []
        for idx in range(power * power):
            x, y = _hilbert_index_to_xy(idx, power)
            if x < side and y < side:
                order.append(y * side + x)
        if len(order) != side * side:
            raise RuntimeError(f"Hilbert order produced {len(order)} tokens for side={side}.")
        return tuple(order)
    raise ValueError(f"Unsupported scan order: {scan}")


class MambaSequenceBlock(nn.Module):
    """A residual Mamba block for feature-token sequence reconstruction.

    The block is intentionally small and explicit: LayerNorm stabilizes the
    frozen-DINO feature scale, Mamba models long-range normal-token context, and
    an MLP refines the reconstructed feature.  The optional reverse Mamba path
    gives each token access to context from both sequence directions, which is
    useful for masked infilling rather than strictly causal forecasting.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        bidirectional: bool = True,
        mlp_ratio: float = 2.0,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as exc:  # pragma: no cover - exercised by env setup.
            raise ImportError(
                "The Mamba architecture requires `mamba-ssm`. Install a wheel "
                "matching the active PyTorch/CUDA environment before training."
            ) from exc

        self.bidirectional = bool(bidirectional)
        self.norm = nn.LayerNorm(dim)
        self.forward_mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.reverse_mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand) if bidirectional else None
        self.drop = nn.Dropout(drop)
        hidden = int(round(dim * mlp_ratio))
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.norm(x)
        fwd = self.forward_mamba(y)
        if self.reverse_mamba is not None:
            rev = torch.flip(self.reverse_mamba(torch.flip(y, dims=(1,))), dims=(1,))
            fwd = 0.5 * (fwd + rev)
        x = residual + self.drop(fwd)
        x = x + self.mlp(x)
        return x


class MambaReconstructionModel(nn.Module):
    """Frozen-DINO feature reconstruction with locality-preserving Mamba scans.

    The model follows the same anomaly principle as the current dual-anchor
    masked baseline: predict the normal feature for masked tokens from visible
    context, then use encoder-vs-reconstruction cosine residual as anomaly
    score.  The decoder is a bidirectional Mamba sequence model over a 2D scan
    order instead of a ViT attention decoder.
    """

    def __init__(
        self,
        encoder: nn.Module,
        target_layers: Sequence[int],
        embed_dim: int,
        layers: int = 4,
        scan: str = "hilbert",
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        bidirectional: bool = True,
        drop: float = 0.0,
        multi_output: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.target_layers = list(target_layers)
        self.multi_output = bool(multi_output)
        self.encoder_require_grad_layer: list[int] = []
        if not hasattr(self.encoder, "num_register_tokens"):
            self.encoder.num_register_tokens = 0
        self.remove_class_token = True
        self.scan = scan
        self.fuse_layer_encoder = self._default_fuse_groups(len(self.target_layers))
        self.fuse_layer_decoder = self._default_fuse_groups(int(layers))
        if self.multi_output and int(layers) < len(self.target_layers):
            raise ValueError(
                "Mamba multi-output alignment needs at least as many decoder "
                f"blocks as encoder target layers: layers={layers}, targets={len(self.target_layers)}"
            )
        self.pre_norm = nn.LayerNorm(embed_dim)
        self.blocks = nn.ModuleList(
            [
                MambaSequenceBlock(
                    dim=embed_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    bidirectional=bidirectional,
                    drop=drop,
                )
                for _ in range(int(layers))
            ]
        )
        self.out_norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    @staticmethod
    def _default_fuse_groups(count: int) -> list[list[int]]:
        if count >= 8:
            return [[0, 1, 2, 3], [4, 5, 6, 7]]
        midpoint = max(count // 2, 1)
        return [list(range(0, midpoint)), list(range(midpoint, count))]

    def fuse_feature(self, feat_list: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.stack(list(feat_list), dim=1).mean(dim=1)

    def _fuse_groups(self, feat_list: Sequence[torch.Tensor], groups: Sequence[Sequence[int]]) -> list[torch.Tensor]:
        fused: list[torch.Tensor] = []
        for group in groups:
            if not group:
                continue
            if max(group) >= len(feat_list):
                raise RuntimeError(f"Fuse group {list(group)} is out of range for {len(feat_list)} features.")
            fused.append(self.fuse_feature([feat_list[idx] for idx in group]))
        if not fused:
            raise RuntimeError("No feature groups were produced for Mamba reconstruction.")
        return fused

    def _encode(self, images: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor, int]:
        x = self.encoder.prepare_tokens(images)
        en_list: list[torch.Tensor] = []
        for idx, blk in enumerate(self.encoder.blocks):
            if idx <= self.target_layers[-1]:
                with torch.no_grad():
                    x = blk(x)
            else:
                continue
            if idx in self.target_layers:
                en_list.append(x)
        if not en_list:
            raise RuntimeError("No encoder target features were captured.")
        start = 1 + int(self.encoder.num_register_tokens)
        spatial_tokens = en_list[0].shape[1] - start
        side = int(math.sqrt(spatial_tokens))
        if side * side != spatial_tokens:
            raise RuntimeError(f"Cannot infer square token grid from shape {en_list[0].shape}.")
        en_list = [item[:, start:, :] for item in en_list]
        target_tokens = self.fuse_feature(en_list)
        return en_list, target_tokens, side

    def _sequence_reconstruct(
        self,
        tokens: torch.Tensor,
        side: int,
        return_intermediates: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        order = torch.as_tensor(scan_permutation(side, self.scan), dtype=torch.long, device=tokens.device)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=tokens.device)
        seq = self.pre_norm(tokens[:, order, :])
        intermediates: list[torch.Tensor] = []
        for block in self.blocks:
            seq = block(seq)
            if return_intermediates:
                pred = self.out_proj(self.out_norm(seq))
                intermediates.append(pred[:, inverse, :])
        final = self.out_proj(self.out_norm(seq))[:, inverse, :]
        return final, intermediates

    def _reconstruction_outputs(
        self,
        en_list: Sequence[torch.Tensor],
        target_tokens: torch.Tensor,
        decoder_input: torch.Tensor,
        side: int,
        batch: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        pred_tokens, pred_layers = self._sequence_reconstruct(
            decoder_input,
            side,
            return_intermediates=self.multi_output,
        )
        if not self.multi_output:
            en_tokens = [target_tokens]
            de_tokens = [pred_tokens]
        else:
            # Mirror Dinomaly's ViT decoder alignment: decoder outputs are
            # reversed before grouped comparison, then shallow/deep encoder
            # groups are matched against decoder groups.
            decoder_layers = pred_layers[::-1]
            en_tokens = self._fuse_groups(en_list, self.fuse_layer_encoder)
            de_tokens = self._fuse_groups(decoder_layers, self.fuse_layer_decoder)
        return _tokens_to_maps(en_tokens, batch, side), _tokens_to_maps(de_tokens, batch, side)

    def forward_tokens(
        self,
        images: torch.Tensor,
        spatial_mask: torch.Tensor | None = None,
        fill: str = "visible_mean",
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor | None]:
        en_list, target_tokens, side = self._encode(images)
        if spatial_mask is not None:
            decoder_input = _masked_tokens(target_tokens, spatial_mask, start=0, fill=fill)
        else:
            decoder_input = target_tokens
        en, de = self._reconstruction_outputs(en_list, target_tokens, decoder_input, side, images.shape[0])
        return en, de, spatial_mask

    def forward_masked(
        self,
        images: torch.Tensor,
        pattern: int,
        config: PerspectiveMaskConfig,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        en_list, target_tokens, side = self._encode(images)
        mask = make_spatial_mask(
            self,
            target_tokens,
            start=0,
            batch=images.shape[0],
            side=side,
            pattern=pattern,
            config=config,
            device=images.device,
        )
        decoder_input = _masked_tokens(target_tokens, mask, start=0, fill=config.fill)
        en, de = self._reconstruction_outputs(en_list, target_tokens, decoder_input, side, images.shape[0])
        return en, de, mask

    def forward(self, images: torch.Tensor):
        en, de, _ = self.forward_tokens(images, spatial_mask=None)
        return en, de


def build_mamba_model(
    dinomaly_repo: str | Path,
    encoder_name: str,
    device: torch.device,
    layers: int = 4,
    scan: str = "hilbert",
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    bidirectional: bool = True,
    drop: float = 0.0,
    multi_output: bool = False,
) -> tuple[nn.Module, nn.Module]:
    """Build the Mamba reconstruction model with the same DINO encoder loader."""

    from .ext import external_repo_context

    with external_repo_context(dinomaly_repo):
        from models import vit_encoder

        encoder = vit_encoder.load(encoder_name)
    if "small" in encoder_name:
        embed_dim = 384
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    elif "base" in encoder_name:
        embed_dim = 768
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    elif "large" in encoder_name:
        embed_dim = 1024
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError(f"Unsupported encoder: {encoder_name}")
    model = MambaReconstructionModel(
        encoder=encoder,
        target_layers=target_layers,
        embed_dim=embed_dim,
        layers=layers,
        scan=scan,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        bidirectional=bidirectional,
        drop=drop,
        multi_output=multi_output,
    ).to(device)
    trainable = nn.ModuleList([model.pre_norm, model.blocks, model.out_norm, model.out_proj])
    return model, trainable
