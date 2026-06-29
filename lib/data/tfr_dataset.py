"""
PyTorch :class:`torch.utils.data.Dataset` для эпох TFR.

**Элемент:** ``x`` форма ``(C, F, T)``, ``y`` — скаляр класса (long).
Опционально ``time_crop`` — случайное окно по времени при каждом ``__getitem__``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset


class TFRDataset(Dataset):
    """Обертка над ``X`` (N,C,F,T) и ``y`` (N,) в тензоры float32 / long."""

    def __init__(self, X: Any, y: Any, *, time_crop: int | None = None) -> None:
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.time_crop = time_crop

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        if self.time_crop is not None:
            _c, _f, t = x.shape
            tc = self.time_crop
            if tc < t:
                t0 = torch.randint(0, t - tc + 1, (1,)).item()
                x = x[:, :, t0 : t0 + tc]
        return x, self.y[idx]
