"""
Optuna objective engine split into focused modules.

Import from this package directly.
"""

from lib.optuna.engine import make_objective_engine
from lib.optuna.fold_runner import run_fold_fn_factory
from lib.optuna.metrics import (
    aggregate,
    cumulative_loss_metric_factory,
    loss_cumulative_delta,
    loss_slope,
)
from lib.optuna.objectives import attrs_fn, objectives_fn
from lib.optuna.params_alexnet import params_fn_factory as params_fn_factory_alexnet
from lib.optuna.params_transformer import params_fn_factory as params_fn_factory_transformer
from lib.optuna.splits import make_splits_fn_factory
from lib.optuna.constraints import cumulative_loss_constraint, slope_constraint
from lib.optuna.types import (
    AlexNetFoldParams,
    Attrs,
    FoldResult,
    FoldTrainingCurves,
    Params,
    Split,
    TransformerFoldParams,
    Values,
)
from lib.optuna.study_analyzers import feasible_trials_less_zero, pareto_front
from lib.optuna.study_io import load_study_sqlite
from lib.optuna.trial_visualization import (
    plot_trial_fold_curves,
    plot_trials_fold_curves,
)

# Backward name: notebooks expect ``params_fn_factory`` for AlexNet-style models.
params_fn_factory = params_fn_factory_alexnet

__all__ = [
    "aggregate",
    "AlexNetFoldParams",
    "Attrs",
    "attrs_fn",
    "FoldResult",
    "FoldTrainingCurves",
    "loss_slope",
    "loss_cumulative_delta",
    "cumulative_loss_metric_factory",
    "make_objective_engine",
    "make_splits_fn_factory",
    "objectives_fn",
    "Params",
    "params_fn_factory",
    "params_fn_factory_alexnet",
    "params_fn_factory_transformer",
    "run_fold_fn_factory",
    "slope_constraint",
    "cumulative_loss_constraint",
    "Split",
    "TransformerFoldParams",
    "Values",
    "feasible_trials_less_zero",
    "pareto_front",
    "load_study_sqlite",
    "plot_trial_fold_curves",
    "plot_trials_fold_curves",
]
