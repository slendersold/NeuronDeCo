"""
Скалярные агрегаты для мультиобъективной Optuna (F1 + динамика loss).
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np


def cumulative_loss_metric_factory(
    *, up_weight: float = 1.1, down_weight: float = 1.0
) -> Callable[[list[float]], float]:
    """
    Фабрика метрики в стиле ``*_factory``: возвращает callable ``losses -> metric``.

    Параметры ``up_weight`` / ``down_weight`` фиксируются при создании функции и не
    передаются при каждом вызове.
    """

    def _metric(losses: list[float]) -> float:
        return loss_cumulative_delta(
            losses, up_weight=up_weight, down_weight=down_weight
        )

    return _metric


def loss_cumulative_delta(
    losses: list[float], *, up_weight: float = 1.1, down_weight: float = 1.0
) -> float:
    """
    Дискретная кумулятивная метрика по траектории ``val_loss``.

    Инициализация:
    ``acc = losses[0]``.

    Для каждой следующей эпохи считаем ``delta = cur - prev``:
    * если ``delta > 0`` (loss вырос), прибавляем ``delta * up_weight``;
    * иначе прибавляем ``delta * down_weight``.

    При минимизации objective лучше модели с меньшим финальным ``acc``.
    Если финальный ``acc`` ниже стартового ``losses[0]``, траектория в целом
    считается устойчиво улучшающейся.

    Parameters
    ----------
    losses:
        Список скаляров ``val_loss`` по эпохам (монотонно растущий индекс = эпоха).
    up_weight:
        Штраф на рост loss (обычно > 1).
    down_weight:
        Вес на снижение loss (обычно 1.0).

    Returns
    -------
    float
        Финальное значение аккумулятора ``acc``.
    """
    if not losses:
        return 0.0
    acc = float(losses[0])
    prev = float(losses[0])
    for cur_raw in losses[1:]:
        cur = float(cur_raw)
        delta = cur - prev
        weight = up_weight if delta > 0.0 else down_weight
        acc += delta * weight
        prev = cur
    return acc


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
