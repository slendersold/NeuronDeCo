from __future__ import annotations

from typing import List

import numpy as np


def loss_slope(losses: list[float]) -> float:
    """Linear trend on last (up to 10) validation losses; more negative is better."""
    tail = losses[-10:] if len(losses) >= 10 else losses
    y = np.asarray(tail, dtype=np.float64)
    if y.size < 2:
        return 0.0
    e = np.arange(len(y), dtype=np.float64)
    try:
        return float(np.polyfit(e, y, 1)[0])
    except np.linalg.LinAlgError:
        return 0.0


def aggregate(values: List[float], mode: str) -> float:
    arr = np.asarray(values, dtype=float)
    if mode == "median":
        return float(np.median(arr))
    if mode == "mean":
        return float(np.mean(arr))
    raise ValueError(f"Unknown aggregate mode: {mode}")
