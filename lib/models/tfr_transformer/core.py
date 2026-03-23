"""
Temporal Transformer stack: [B, T, in_features] → [B, T, num_classes] (per-time logits).

Originally from ``transformer_17_03_26.ipynb``; ``num_classes`` was hard-coded as 2 — fixed.
"""

from __future__ import annotations

import torch.nn as nn

from lib.models.tfr_transformer.preprocess import PositionalEncoding


class TFRSequenceTransformer(nn.Module):
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
    ):
        super().__init__()
        self.embed_layer = nn.LazyLinear(embed_dim)
        self.position_encoder = PositionalEncoding(embed_dim, seq_len, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_fc,
            dropout=dropout,
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
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.embed_layer(x)
        out = self.norm_before_pe(out)
        out = self.position_encoder(out)
        out = self.encoder(out)
        out = self.norm_after_pe(out)
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
