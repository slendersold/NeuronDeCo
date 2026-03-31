"""
Стек Transformer по **оси времени** последовательности ``(B, T, D)``.

Здесь ``D`` — размер признака на один шаг времени после препроцессора TFR
(не путать с ``embed_dim`` внутри блока).

**Вход:** ``(B, T, D)`` — см. :class:`lib.models.tfr_transformer.typing.TransformerSequence`.

**Выход:** логиты **на каждом шаге** ``(B, T, K)`` — :class:`lib.models.tfr_transformer.typing.TransformerPerStepLogits`.

Агрегация по времени выполняется в :class:`~lib.models.tfr_transformer.wrapper.TFRTransformerWrapper`.
"""

from __future__ import annotations

import torch.nn as nn

from lib.models.tfr_transformer.preprocess import PositionalEncoding
from lib.models.tfr_transformer.typing import (
    TransformerPerStepLogits,
    TransformerSequence,
)


class TFRSequenceTransformer(nn.Module):
    """
    Parameters
    ----------
    seq_len:
        Максимальная длина для буфера позиционного кодирования; нужно ``T <= seq_len``.
    embed_dim:
        Размерность после ``LazyLinear`` и d_model энкодера.
    nhead:
        Число голов внимания; должно делить ``embed_dim``.
    dim_fc:
        ``dim_feedforward`` в ``TransformerEncoderLayer``.
    num_layers:
        Число слоёв энкодера.
    dropout:
        Dropout в encoder / residual блоках.
    num_classes:
        ``K`` — число классов на **каждом** временном шаге (до SeqPool).
    """

    def __init__(
        self,
        *,
        seq_len: int,
        embed_dim: int,
        nhead: int,
        dim_fc: int,
        num_layers: int,
        dropout: float,
        num_classes: int,
        use_conv: bool = False,
        conv_kernel_size: int = 3,
        encoder_dropout: float | None = None,
        mlp_dropout: float | None = None,
        conv_dropout: float | None = None,
    ) -> None:
        super().__init__()
        if conv_kernel_size < 1 or conv_kernel_size % 2 == 0:
            raise ValueError("conv_kernel_size must be an odd positive integer.")

        enc_dropout = dropout if encoder_dropout is None else encoder_dropout
        head_dropout = dropout if mlp_dropout is None else mlp_dropout
        conv_drop = dropout if conv_dropout is None else conv_dropout

        self.embed_layer = nn.LazyLinear(embed_dim)
        self.position_encoder = PositionalEncoding(embed_dim, seq_len, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_fc,
            dropout=enc_dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm_before_pe = nn.LayerNorm(embed_dim)
        self.norm_after_pe = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.fc3 = nn.Linear(embed_dim, embed_dim)
        self.fc4 = nn.Linear(embed_dim, embed_dim // 2)
        self.fc5 = nn.Linear(embed_dim // 2, embed_dim // 4)
        self.fc6 = nn.Linear(embed_dim // 4, num_classes)
        self.dropout = nn.Dropout(head_dropout)
        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.use_conv = use_conv
        self.norm_conv = nn.LayerNorm(embed_dim)
        self.conv_dropout = nn.Dropout(conv_drop)
        padding = conv_kernel_size // 2
        self.conv_layers = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=conv_kernel_size, padding=padding),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: TransformerSequence) -> TransformerPerStepLogits:
        out = self.embed_layer(x)
        if out.size(1) > self.position_encoder.pe.size(1):
            raise ValueError(
                f"Sequence length T={out.size(1)} exceeds positional encoding "
                f"buffer {self.position_encoder.pe.size(1)}; increase seq_len."
            )
        out = self.norm_before_pe(out)
        out = self.position_encoder(out)
        out = self.encoder(out)
        out = self.norm_after_pe(out)
        if self.use_conv:
            out_conv = out.transpose(1, 2)
            out_conv = self.conv_layers(out_conv)
            out_conv = out_conv.transpose(1, 2)
            out = self.norm_conv(out + self.conv_dropout(out_conv))
        residue = out
        out = self.dropout(self.activation(self.fc1(out)))
        out = self.norm1(out + residue)
        residue = out
        out = self.dropout(self.activation(self.fc2(out)))
        out = self.norm2(out + residue)
        out = self.dropout(self.activation(self.fc3(out)))
        out = self.dropout(self.activation(self.fc4(out)))
        out = self.dropout(self.activation(self.fc5(out)))
        return self.fc6(out)
