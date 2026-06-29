"""
Голова AlexNet: адаптивное усреднение по пространству ``(F', T')`` + MLP до ``K`` классов.

**Вход:** карта признаков ``(B, 256, F', T')`` с последнего conv backbone.

**Выход классификатора:** ``(B, K)`` логитов — см. :data:`lib.models.alexnet.typing.AlexNetLogits`.
"""

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
    Parameters
    ----------
    spatial_pooled_hw:
        Целевой размер после ``AdaptiveAvgPool2d``; задаёт размерность flatten = ``256 * h * w``.
    hidden_dim:
        Размер скрытого слоя MLP.
    num_classes:
        ``K`` — число классов (размер последнего линейного слоя).
    dropout:
        Dropout после первого линейного слоя и перед выходом.

    Returns
    -------
    tuple
        ``(adapt, classifier)`` — оба ``nn.Module``.
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
