"""Full AlexNet for TFR — composes backbone + adaptive pool + classifier."""

from __future__ import annotations

import torch
import torch.nn as nn

from lib.models.alexnet.backbone import build_alexnet_backbone
from lib.models.alexnet.head import build_alexnet_head


class AlexNetTFR(nn.Module):
    def __init__(self, in_channels: int = 7, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.features = build_alexnet_backbone(in_channels)
        self.adapt, self.classifier = build_alexnet_head(
            spatial_pooled_hw=(4, 8),
            hidden_dim=512,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.adapt(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
