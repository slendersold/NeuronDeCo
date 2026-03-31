"""
Constraint helpers for Optuna samplers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lib.optuna.metrics import loss_cumulative_delta


def slope_constraint(trial: Any) -> tuple[float]:
    """
    Feasible when median slope of validation loss is <= 0.

    Optuna expects a sequence (tuple/list) from ``constraints_func``.
    """

    try:
        fold_curves = trial.user_attrs.get("fold_curves", None)
        if not fold_curves:
            return (0.0,)

        slopes: list[float] = []
        for fc in fold_curves:
            val_losses = fc.get("val_losses", [])
            if not val_losses:
                continue
            tail = val_losses[-10:] if len(val_losses) >= 10 else val_losses
            e = np.arange(len(tail), dtype=np.float64)
            slope = float(np.polyfit(e, np.array(tail, dtype=np.float64), 1)[0])
            slopes.append(slope)

        if not slopes:
            return (0.0,)
        return (float(np.median(np.asarray(slopes, dtype=np.float64))),)
    except Exception as exc:  # keep constraint function non-failing
        trial.set_user_attr("constraint_error", repr(exc))
        return (0.0,)


def cumulative_loss_constraint(trial: Any) -> tuple[float]:
    """
    Feasible when median cumulative accumulator does not exceed start loss.
    """

    try:
        fold_curves = trial.user_attrs.get("fold_curves", None)
        if not fold_curves:
            return (0.0,)

        deviations: list[float] = []
        for fc in fold_curves:
            val_losses = fc.get("val_losses", [])
            if not val_losses:
                continue
            start_loss = float(val_losses[0])
            cumulative = float(loss_cumulative_delta(val_losses))
            deviations.append(cumulative - start_loss)

        if not deviations:
            return (0.0,)
        return (float(np.median(np.asarray(deviations, dtype=np.float64))),)
    except Exception as exc:  # keep constraint function non-failing
        trial.set_user_attr("constraint_error", repr(exc))
        return (0.0,)

