"""
Нормализация TFR — перенос из legacy `utils.normalisation`.
"""

from __future__ import annotations

import numpy as np


def fit_tfr_robust_stats(
    X: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit median/IQR statistics on train data only (axes N and T).

    Returns
    -------
    median, iqr
        Arrays with shape ``(1, C, F, 1)`` suitable for :func:`apply_tfr_robust_norm`.
    """
    median = np.median(X, axis=(0, 3), keepdims=True)
    q25 = np.percentile(X, 25, axis=(0, 3), keepdims=True)
    q75 = np.percentile(X, 75, axis=(0, 3), keepdims=True)
    iqr = (q75 - q25) + eps
    return median, iqr


def apply_tfr_robust_norm(
    X: np.ndarray,
    median: np.ndarray,
    iqr: np.ndarray,
) -> np.ndarray:
    """Apply train-fitted robust normalization to ``X``."""
    return (X - median) / iqr + 0.5


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

    median, iqr = fit_tfr_robust_stats(X, eps=eps)
    return apply_tfr_robust_norm(X, median, iqr)

