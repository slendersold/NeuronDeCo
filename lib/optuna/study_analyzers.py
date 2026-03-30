from __future__ import annotations

from typing import Sequence

import optuna
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial


def pareto_front(trials: Sequence[FrozenTrial], directions: Sequence[StudyDirection]) -> list[FrozenTrial]:
    """
    Возвращает список trial-ов на Парето-фронте.

    Параметры
    ----------
    trials:
        Коллекция FrozenTrial.
    directions:
        Список направлений оптимизации для каждой метрики (обычно как ``study.directions``).
    """

    def dominates(a: FrozenTrial, b: FrozenTrial) -> bool:
        # a доминирует b, если a не хуже b по всем целям и строго лучше хотя бы по одной.
        better_or_equal = True
        strictly_better = False
        for av, bv, d in zip(a.values or (), b.values or (), directions):
            if d == StudyDirection.MAXIMIZE:
                if av < bv:
                    better_or_equal = False
                if av > bv:
                    strictly_better = True
            else:  # MINIMIZE
                if av > bv:
                    better_or_equal = False
                if av < bv:
                    strictly_better = True
        return better_or_equal and strictly_better

    front: list[FrozenTrial] = []
    for t in trials:
        if not any(dominates(o, t) for o in trials if o.number != t.number):
            front.append(t)
    return front


def feasible_trials_less_zero(
    study: optuna.Study,
    *,
    slope_idx: int = 1,
    threshold: float = 0.0,
) -> list[FrozenTrial]:
    """
    Фильтр trial-ов по условию на заданную objective.

    В ноутбуках условие обычно: ``t.values[1] <= 0`` (slope <= 0 при minimize).
    """

    out: list[FrozenTrial] = []
    for t in study.trials:
        if t.values is None:
            continue
        if len(t.values) <= slope_idx:
            continue
        if t.values[slope_idx] <= threshold:
            out.append(t)
    return out

