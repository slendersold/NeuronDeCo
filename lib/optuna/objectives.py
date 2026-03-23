from __future__ import annotations

from typing import Any, List, Tuple

from lib.optuna.metrics import aggregate
from lib.optuna.types import Attrs, FoldResult, Params, Values


def objectives_fn(folds: List[FoldResult], agg_mode: str) -> Tuple[float, float]:
    f1 = aggregate([f.best_f1 for f in folds], agg_mode)
    slope = aggregate([f.loss_metric for f in folds], agg_mode)
    return f1, slope


def attrs_fn(trial: Any, params: Params, folds: List[FoldResult], values: Values) -> Attrs:
    return {
        "params": params,
        "fold_best_f1s": [f.best_f1 for f in folds],
        "fold_slopes": [f.loss_metric for f in folds],
        "fold_curves": [{"split": f.split, **f.curves} for f in folds],
        "objectives": values,
        "cv_mode": "kfold" if len(folds) > 1 else "holdout",
    }
