"""
Стандартные функции целей и атрибутов trial для двух objective (F1 ↑, slope ↓).
"""

from __future__ import annotations

from typing import Any, List, Tuple

from lib.optuna.metrics import aggregate
from lib.optuna.types import Attrs, FoldResult, Params, Values


def objectives_fn(folds: List[FoldResult], agg_mode: str) -> Tuple[float, float]:
    """
    Два скаляра для Optuna: агрегированный best F1 и агрегированный ``loss_metric``.

    Parameters
    ----------
    folds:
        Результаты по каждому сплиту (holdout или K фолдов).
    agg_mode:
        ``\"median\"`` или ``\"mean\"`` — см. :func:`lib.optuna.metrics.aggregate`.

    Returns
    -------
    tuple[float, float]
        ``(f1_agg, slope_agg)`` — порядок должен совпадать с ``directions`` в study.
    """
    f1 = aggregate([f.best_f1 for f in folds], agg_mode)
    slope = aggregate([f.loss_metric for f in folds], agg_mode)
    return f1, slope


def attrs_fn(trial: Any, params: Params, folds: List[FoldResult], values: Values) -> Attrs:
    """
    Сохраняет в user attrs подробности для анализа (кривые, параметры, режим CV).

    Parameters
    ----------
    trial:
        Optuna trial (используется только как носитель ``set_user_attr`` снаружи).
    params:
        Тот же словарь, что вернул ``params_fn``.
    folds:
        Все :class:`FoldResult` текущего trial.
    values:
        То, что вернул ``objectives_fn`` (для логирования).
    """
    return {
        "params": params,
        "fold_best_f1s": [f.best_f1 for f in folds],
        "fold_slopes": [f.loss_metric for f in folds],
        "fold_curves": [{"split": f.split, **f.curves} for f in folds],
        "objectives": values,
        "cv_mode": "kfold" if len(folds) > 1 else "holdout",
    }
