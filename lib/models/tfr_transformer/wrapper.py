"""
End-to-end TFR classifier: [B, C, F, T] → [B, num_classes] logits.

Pools per-timestep class logits with ``SeqPool`` (mean / softmax / …).
"""

from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn

from lib.models.tfr_transformer.core import TFRSequenceTransformer
from lib.models.tfr_transformer.preprocess import (
    PREPROCESS_BUILDERS,
    SeqPool,
    TFRToSeqFTPlaneConvCollapse,
)


def _build_preprocess(
    preprocess: Union[str, nn.Module],
    *,
    kernel_freq: int,
    kernel_time: int,
) -> nn.Module:
    if isinstance(preprocess, nn.Module):
        return preprocess
    if preprocess == "ft_plane_conv":
        return TFRToSeqFTPlaneConvCollapse(kernel_freq=kernel_freq, kernel_time=kernel_time)
    if preprocess not in PREPROCESS_BUILDERS:
        raise ValueError(
            f"Unknown preprocess={preprocess!r}. Use {set(PREPROCESS_BUILDERS)} or pass an nn.Module."
        )
    return PREPROCESS_BUILDERS[preprocess]()


class TFRTransformerWrapper(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.5,
        *,
        seq_len: int = 1001,
        pooling: Union[str, SeqPool] = "softmax",
        preprocess: Union[str, nn.Module] = "flatten",
        kernel_freq: int = 3,
        kernel_time: int = 3,
        embed_dim: int = 256,
        nhead: int = 8,
        dim_fc: int = 512,
        num_layers: int = 4,
    ):
        super().__init__()
        self.preprocess_mod = _build_preprocess(
            preprocess, kernel_freq=kernel_freq, kernel_time=kernel_time
        )
        pool = SeqPool(mode=pooling) if isinstance(pooling, str) else pooling
        self.pooling = pool
        self.transformer = TFRSequenceTransformer(
            seq_len=seq_len,
            embed_dim=embed_dim,
            nhead=nhead,
            dim_fc=dim_fc,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,F,T], got {tuple(x.shape)}")
        seq = self.preprocess_mod(x)
        logits_t = self.transformer(seq)
        return self.pooling(logits_t)
