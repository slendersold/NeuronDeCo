from __future__ import annotations

from typing import Any, Callable, List, Optional

import numpy as np

from lib.optuna.types import Attrs, FoldResult, Params, Split, Values


def make_objective_engine(
    *,
    X,
    y,
    make_splits_fn: Callable[[np.ndarray, np.ndarray], List[Split]],
    run_fold_fn: Callable[[Split, Params], FoldResult],
    aggregate_mode: str = "median",
    params_fn: Callable[[Any], Params],
    objectives_fn: Callable[[List[FoldResult], str], Values],
    attrs_fn: Optional[Callable[[Any, Params, List[FoldResult], Values], Attrs]] = None,
):
    """
    Invariant pipeline: splits → fold runs → aggregate objectives → optional user attrs.

    ``params_fn`` must be deterministic for a given trial (same suggest_* order).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    splits = make_splits_fn(X, y)

    def objective(trial):
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
