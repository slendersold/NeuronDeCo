"""
Типизация, специфичная для **AlexNet на TFR** (2D-свёртки по плоскости частота×время).

Архитектура
-----------
Вход трактуется как ``(B, C, F, T)`` с осями канал / частота / время (как у ``nn.Conv2d``
на ``(F, T)`` при ``channels_first``).

**Ограничения по размеру:** после двух ``MaxPool2d`` маленькие ``F`` или ``T`` могут
схлопнуться до нуля — это ограничение архитектуры, не типизации. Для малых карт
нужен другой backbone или другие kernel/stride.

Типы ниже — алиасы к ``torch.Tensor``; фактические формы фиксируются в докстрингах
классов :mod:`lib.models.alexnet`.
"""

from __future__ import annotations

from typing import TypeAlias

from torch import Tensor

AlexNetBatchIn: TypeAlias = Tensor
"""Мини-батч TFR, форма ``(B, C, F, T)``, ``dtype`` обычно ``float32``."""

AlexNetLogits: TypeAlias = Tensor
"""Логиты классов, форма ``(B, K)``, ``K = num_classes``."""

AlexNetFeatureMap: TypeAlias = Tensor
"""Промежуточная карта после ``features``, форма ``(B, 256, F', T')`` (зависит от входа)."""
