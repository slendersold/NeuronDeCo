"""
Скалярные агрегаты для мультиобъективной Optuna (F1 + динамика loss).
"""

from __future__ import annotations

from typing import List

import numpy as np


def loss_slope(losses: list[float]) -> float:
    """
    Оценка тренда валидационного loss по эпохам одного фолда.

    Берётся линейная регрессия (полином степени 1) по последним **до 10** точкам
    ``val_loss``. Отрицательный наклон = loss в среднем падает (желательно при втором
    objective ``minimize``).

    **Устойчивость:** при ``len(losses) < 2`` или ``LinAlgError`` у ``polyfit`` возвращается
    ``0.0``, чтобы trial не падал на коротком обучении.

    Parameters
    ----------
    losses:
        Список скаляров ``val_loss`` по эпохам (монотонно растущий индекс = эпоха).

    Returns
    -------
    float
        Коэффициент наклона ``a`` в аппроксимации ``loss ≈ a * epoch_local + b``.
    """
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
    """
    Агрегация метрик по фолдам (median / mean).

    Parameters
    ----------
    values:
        Одно значение на фолд (например ``best_f1`` или ``loss_metric``).
    mode:
        ``\"median\"`` или ``\"mean\"``.

    Returns
    -------
    float
        Сводная метрика для передачи в Optuna.
    """
    arr = np.asarray(values, dtype=float)
    if mode == "median":
        return float(np.median(arr))
    if mode == "mean":
        return float(np.mean(arr))
    raise ValueError(f"Unknown aggregate mode: {mode}")
