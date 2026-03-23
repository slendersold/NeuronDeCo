"""Adaptive pooling + MLP classifier head."""

from __future__ import annotations

import torch.nn as nn


def build_alexnet_head(
    *,
    spatial_pooled_hw: tuple[int, int] = (4, 8),
    hidden_dim: int = 512,
    num_classes: int = 2,
    dropout: float = 0.5,
) -> tuple[nn.Module, nn.Module]:
    """
    Returns (adapt_pool, classifier).

    Feature maps after backbone are pooled to ``spatial_pooled_hw`` then flattened.
    """
    h, w = spatial_pooled_hw
    adapt = nn.AdaptiveAvgPool2d((h, w))
    in_flat = 256 * h * w
    classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_flat, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    )
    return adapt, classifier
