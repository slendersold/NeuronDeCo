"""Shared model typing hooks (optional)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch.nn as nn


@runtime_checkable
class TFRClassifier(Protocol):
    """Maps batch of TFR tensors [B,C,F,T] to logits [B, num_classes]."""

    def forward(self, x):  # noqa: ANN001
        ...
