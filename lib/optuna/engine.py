"""
Инвариантный конвейер objective для Optuna: сплиты → фолды → агрегат целей → user attrs.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

import numpy as np

from lib.core.tensors import EpochLabelsArray, TFRFeatureArray
from lib.optuna.types import Attrs, FoldResult, Params, Split, Values


def make_objective_engine(
    *,
    X: TFRFeatureArray | np.ndarray,
    y: EpochLabelsArray | np.ndarray,
    make_splits_fn: Callable[[np.ndarray, np.ndarray], List[Split]],
    run_fold_fn: Callable[[Split, Params], FoldResult],
    aggregate_mode: str = "median",
    params_fn: Callable[[Any], Params],
    objectives_fn: Callable[[List[FoldResult], str], Values],
    attrs_fn: Optional[Callable[[Any, Params, List[FoldResult], Values], Attrs]] = None,
) -> Callable[[Any], float | tuple[float, ...]]:
    """
    Собирает callable ``objective(trial)`` для ``study.optimize``.

    Поток данных
    ------------
    1. ``X``, ``y`` приводятся к ``ndarray``; ожидаемые формы — ``(N,C,F,T)`` и ``(N,)``.
    2. ``make_splits_fn(X, y)`` возвращает список :class:`Split`.
    3. Для каждого trial: ``params = params_fn(trial)``, затем для каждого сплита
       ``run_fold_fn(split, params)`` → :class:`FoldResult`.
    4. ``values = objectives_fn(fold_results, aggregate_mode)`` — скаляр или кортеж
       для multi-objective.
    5. Если задан ``attrs_fn``, результаты кладутся в ``trial.set_user_attr``.

    Parameters
    ----------
    X, y:
        Полный датасет (все эпохи); сплиты режут по оси ``N``.
    make_splits_fn:
        См. :func:`lib.optuna.splits.make_splits_fn_factory`.
    run_fold_fn:
        См. :func:`lib.optuna.fold_runner.run_fold_fn_factory`.
    aggregate_mode:
        Передаётся в ``objectives_fn`` (обычно ``\"median\"`` или ``\"mean\"`` по фолдам).
    params_fn:
        Должен быть **детерминирован** для фиксированного trial (одинаковый порядок ``suggest_*``).
    objectives_fn, attrs_fn:
        См. :mod:`lib.optuna.objectives`.

    Returns
    -------
    callable
        Функция одного аргумента ``trial`` (Optuna), возвращающая float или tuple float.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    splits = make_splits_fn(X_arr, y_arr)

    def objective(trial: Any) -> float | tuple[float, ...]:
        params = params_fn(trial)
        fold_results: List[FoldResult] = []
        for sp in splits:
            fold_results.append(run_fold_fn(sp, params))

        values = objectives_fn(fold_results, aggregate_mode)

        if attrs_fn is not None:
            attrs = attrs_fn(trial, params, fold_results, values)
            for k, v in attrs.items():
                trial.set_user_attr(k, v)

        if isinstance(values, (list, tuple, np.ndarray)):
            return tuple(float(x) for x in values)
        return float(values)

    return objective
