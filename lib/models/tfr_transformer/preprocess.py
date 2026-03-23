"""
TFR [B, C, F, T] → sequence [B, T, feat] for the temporal Transformer.

Ported from ``notebooks/transformer_17_03_26.ipynb`` with shape checks kept explicit.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, timesteps: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, timesteps, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(timesteps, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class SeqPool(nn.Module):
    """Pool over time dimension of [B, T, D]."""

    def __init__(self, mode: str = "mean"):
        super().__init__()
        assert mode in ("mean", "max", "last", "softmax", "none"), f"Unknown mode {mode}"
        self.mode = mode
        if self.mode == "softmax":
            self.score = nn.LazyLinear(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "mean":
            return x.mean(dim=1)
        if self.mode == "max":
            return x.max(dim=1).values
        if self.mode == "last":
            return x[:, -1]
        if self.mode == "softmax":
            w = torch.softmax(self.score(x), dim=1)
            return (x * w).sum(dim=1)
        return x


class TFRToSeqFlatten(nn.Module):
    """[B, C, F, T] -> [B, T, C*F]"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,F,T], got {tuple(x.shape)}")
        b, c, freq, time = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        return x.reshape(b, time, c * freq)


class TFRToSeqChannelConvCollapse(nn.Module):
    """
    Per-channel weights, collapse channels → [B, T, F].

    Implemented as conv3d with kernel [C,1,1].
    """

    def __init__(self, bias: bool = True):
        super().__init__()
        self.bias_enabled = bias
        self._built = False
        self._channels: int | None = None
        self.channel_kernel: nn.Parameter | None = None
        self.channel_bias: nn.Parameter | None = None

    def _build(self, channels: int, device: torch.device, dtype: torch.dtype) -> None:
        self._channels = channels
        self.channel_kernel = nn.Parameter(torch.empty(channels, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.channel_kernel.view(1, 1, channels), a=math.sqrt(5))
        if self.bias_enabled:
            self.channel_bias = nn.Parameter(torch.zeros(1, device=device, dtype=dtype))
        else:
            self.register_parameter("channel_bias", None)
        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,F,T], got {tuple(x.shape)}")
        b, c, freq, time = x.shape
        if not self._built:
            self._build(c, x.device, x.dtype)
        if c != self._channels:
            raise ValueError(f"Channel count mismatch: expected C={self._channels}, got {c}")
        x5 = x.unsqueeze(1)
        kernel = self.channel_kernel.view(1, 1, c, 1, 1)
        out = F.conv3d(x5, weight=kernel, bias=self.channel_bias, stride=1, padding=0)
        out = out.squeeze(1).squeeze(1).permute(0, 2, 1).contiguous()
        return out


class TFRToSeqFTPlaneConvCollapse(nn.Module):
    """Shared (F×T) kernel across channels → [B, T, F]."""

    def __init__(self, kernel_freq: int = 3, kernel_time: int = 3, bias: bool = True):
        super().__init__()
        if kernel_freq <= 0 or kernel_time <= 0:
            raise ValueError("kernel_freq and kernel_time must be > 0")
        self.kernel_freq = kernel_freq
        self.kernel_time = kernel_time
        self.bias_enabled = bias
        self._built = False
        self._channels: int | None = None
        self.ft_kernel: nn.Parameter | None = None
        self.ft_bias: nn.Parameter | None = None

    def _build(self, channels: int, device: torch.device, dtype: torch.dtype) -> None:
        self._channels = channels
        self.ft_kernel = nn.Parameter(
            torch.empty(self.kernel_freq, self.kernel_time, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(
            self.ft_kernel.view(1, 1, self.kernel_freq, self.kernel_time), a=math.sqrt(5)
        )
        if self.bias_enabled:
            self.ft_bias = nn.Parameter(torch.zeros(1, device=device, dtype=dtype))
        else:
            self.register_parameter("ft_bias", None)
        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,F,T], got {tuple(x.shape)}")
        b, c, freq, time = x.shape
        if not self._built:
            self._build(c, x.device, x.dtype)
        if c != self._channels:
            raise ValueError(f"Channel count mismatch: expected C={self._channels}, got {c}")
        x5 = x.unsqueeze(1)
        kernel = self.ft_kernel.view(1, 1, 1, self.kernel_freq, self.kernel_time).repeat(1, 1, c, 1, 1)
        pad_f = self.kernel_freq // 2
        pad_t = self.kernel_time // 2
        out = F.conv3d(x5, weight=kernel, bias=self.ft_bias, stride=1, padding=(0, pad_f, pad_t))
        return out.squeeze(1).squeeze(1).permute(0, 2, 1).contiguous()


class TFRToSeqPixelWeightCollapse(nn.Module):
    """Learnable weight per (f, t); sum over channels → [B, T, F]."""

    def __init__(self) -> None:
        super().__init__()
        self.weight: nn.Parameter | None = None
        self._built = False
        self._shape: tuple[int, int] | None = None

    def _build(self, f: int, t: int, device: torch.device, dtype: torch.dtype) -> None:
        self.weight = nn.Parameter(torch.randn(f, t, device=device, dtype=dtype))
        self._shape = (f, t)
        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,F,T], got {tuple(x.shape)}")
        b, c, f, t = x.shape
        if not self._built:
            self._build(f, t, x.device, x.dtype)
        if (f, t) != self._shape:
            raise ValueError(f"(F,T) mismatch: expected {self._shape}, got {(f, t)}")
        x = x.sum(dim=1) * self.weight
        return x.permute(0, 2, 1)


PREPROCESS_BUILDERS = {
    "flatten": TFRToSeqFlatten,
    "channel_conv": TFRToSeqChannelConvCollapse,
    "ft_plane_conv": TFRToSeqFTPlaneConvCollapse,
    "pixel_weight": TFRToSeqPixelWeightCollapse,
}
