"""
Сборка: **TFR** → препроцессор → Transformer по времени → **логиты классов** ``(B, K)``.

Соответствует :class:`lib.core.contracts.TorchTFRClassifier` при корректном ``seq_len``
и режиме ``SeqPool``, дающем ``(B, K)`` (не ``none``).
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
from lib.models.tfr_transformer.typing import (
    TransformerBatchIn,
    TransformerPooledLogits,
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
    """
    Parameters
    ----------
    num_classes:
        ``K`` в выходе ``(B, K)``.
    dropout:
        Dropout внутри Transformer и FC-блоков.
    seq_len:
        Верхняя граница длины последовательности для PE; должно быть ``>= T`` после препроцессора
        (обычно ``T`` совпадает с числом временных бинов во входном TFR).
    pooling:
        Строка (режим :class:`SeqPool`) или готовый модуль пулинга по времени.
    preprocess:
        Ключ из ``PREPROCESS_BUILDERS`` или пользовательский ``nn.Module`` ``(B,C,F,T)→(B,T,D)``.
    kernel_freq, kernel_time:
        Для строки ``\"ft_plane_conv\"`` — размеры FT-kernel.
    embed_dim, nhead, dim_fc, num_layers:
        Гиперпараметры :class:`TFRSequenceTransformer`.

    Forward
    -------
    * **Вход:** ``(B, C, F, T)``.
    * **Выход:** логиты ``(B, K)``.
    """

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
    ) -> None:
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

    def forward(self, x: TransformerBatchIn) -> TransformerPooledLogits:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,F,T], got {tuple(x.shape)}")
        seq = self.preprocess_mod(x)
        logits_t = self.transformer(seq)
        return self.pooling(logits_t)
