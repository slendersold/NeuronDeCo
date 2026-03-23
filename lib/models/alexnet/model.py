"""
Полная модель AlexNet для TFR: backbone → adaptive pool → MLP.

Реализует контракт :class:`lib.core.contracts.TorchTFRClassifier` при типичных ``C,F,T``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lib.models.alexnet.backbone import build_alexnet_backbone
from lib.models.alexnet.head import build_alexnet_head
from lib.models.alexnet.typing import AlexNetBatchIn, AlexNetLogits


class AlexNetTFR(nn.Module):
    """
    Parameters
    ----------
    in_channels:
        ``C`` в форме входа ``(B, C, F, T)``.
    num_classes:
        ``K`` — число классов; выход ``(B, K)``.
    dropout:
        Dropout в MLP-голове.

    Forward
    -------
    * **Вход:** ``x`` форма ``(B, C, F, T)``, float.
    * **Выход:** логиты ``(B, K)``.
    """

    def __init__(self, in_channels: int = 7, num_classes: int = 2, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = build_alexnet_backbone(in_channels)
        self.adapt, self.classifier = build_alexnet_head(
            spatial_pooled_hw=(4, 8),
            hidden_dim=512,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: AlexNetBatchIn) -> AlexNetLogits:
        x = self.features(x)
        x = self.adapt(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
