"""
Нормализация TFR — перенос из legacy `utils.normalisation`.
"""

from __future__ import annotations

import numpy as np


def normalize_tfr_robust(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Робастная нормализация TFR:
    - вместо среднего — медиана
    - вместо std — IQR (q75 - q25)
    - результат приводится к диапазону [0, 1]

    Parameters
    ----------
    X:
        ``(N, C, F, T)``, dtype float.
    eps:
        Число для устойчивости деления.
    """

    median = np.median(X, axis=(0, 3), keepdims=True)
    q25 = np.percentile(X, 25, axis=(0, 3), keepdims=True)
    q75 = np.percentile(X, 75, axis=(0, 3), keepdims=True)
    iqr = (q75 - q25) + eps
    X_norm = (X - median) / iqr + 0.5
    return X_norm

