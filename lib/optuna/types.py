from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Union

import numpy as np

Params = Dict[str, Dict[str, Any]]
Attrs = Dict[str, Any]
Values = Union[float, Sequence[float]]


@dataclass(frozen=True)
class Split:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray


@dataclass
class FoldResult:
    split: str
    best_f1: float
    loss_metric: float
    curves: Dict[str, List[float]]
