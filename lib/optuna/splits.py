from __future__ import annotations

from typing import List

import numpy as np

from lib.optuna.types import Split


def make_splits_fn_factory(test_size: float, seed: int, cv: bool):
    def _make_splits(X: np.ndarray, y: np.ndarray) -> List[Split]:
        if cv:
            from sklearn.model_selection import StratifiedKFold

            n_splits = max(2, int(round(1 / test_size)))
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            out: List[Split] = []
            for i, (tr, va) in enumerate(skf.split(X, y)):
                out.append(Split(f"fold{i}", X[tr], y[tr], X[va], y[va]))
            return out
        from sklearn.model_selection import train_test_split

        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        return [Split("holdout", X_tr, y_tr, X_va, y_va)]

    return _make_splits
