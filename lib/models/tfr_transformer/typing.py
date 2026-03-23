"""
Типизация для **Transformer по времени** над последовательностью, полученной из TFR.

Пайплайн
--------
1. Препроцессор (см. :mod:`lib.models.tfr_transformer.preprocess`) переводит
   ``(B, C, F, T)`` → ``(B, T, D)``, где ``D`` зависит от варианта:

   * ``flatten``: ``D = C * F``
   * ``channel_conv``: ``D = F`` (каналы схлопнуты обучаемыми весами)
   * ``ft_plane_conv``: ``D = F`` (общий FT-kernel по каналам)
   * ``pixel_weight``: ``D = F`` (вес на каждую ячейку ``(f,t)`` после суммы по ``C``)

2. :class:`~lib.models.tfr_transformer.core.TFRSequenceTransformer` карту
   ``(B, T, D)`` встраивает в ``embed_dim``, прогоняет ``TransformerEncoder`` и выдаёт
   **логиты по каждому шагу времени**: ``(B, T, K)``.

3. :class:`~lib.models.tfr_transformer.preprocess.SeqPool` агрегирует по оси времени →
   ``(B, K)``.

**Вариабельность:** ``T`` (число временных бинов) должно быть ``≤ seq_len`` (размер
буфера позиционного кодирования). ``seq_len`` задаётся при создании модели; для Optuna
обычно берут ``X.shape[3]`` после crop.
"""

from __future__ import annotations

from typing import TypeAlias

from torch import Tensor

TransformerBatchIn: TypeAlias = Tensor
"""Сырой TFR, форма ``(B, C, F, T)``."""

TransformerSequence: TypeAlias = Tensor
"""После препроцессора, форма ``(B, T, D)``; ``D`` зависит от режима препроцессинга."""

TransformerPerStepLogits: TypeAlias = Tensor
"""Выход стека Transformer до pooling, форма ``(B, T, K)``."""

TransformerPooledLogits: TypeAlias = Tensor
"""После ``SeqPool`` по времени, форма ``(B, K)`` — вход в ``cross_entropy``."""
