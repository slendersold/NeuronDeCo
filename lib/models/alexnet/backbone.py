"""CNN feature extractor for TFR maps [B, C, F, T]."""

from __future__ import annotations

import torch.nn as nn


def build_alexnet_backbone(in_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=(5, 11), stride=(1, 4), padding=(2, 0)),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=(2, 3), stride=(2, 2)),
        nn.Conv2d(64, 192, kernel_size=5, padding=2),
        nn.BatchNorm2d(192),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=(2, 3), stride=(2, 2)),
        nn.Conv2d(192, 384, kernel_size=3, padding=1),
        nn.BatchNorm2d(384),
        nn.ReLU(inplace=True),
        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
    )
