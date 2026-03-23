"""
Optuna objective engine split into focused modules.

Import from here or use the compatibility shim ``lib.optuna_objective_makers``.
"""

from lib.optuna.engine import make_objective_engine
from lib.optuna.fold_runner import run_fold_fn_factory
from lib.optuna.metrics import aggregate, loss_slope
from lib.optuna.objectives import attrs_fn, objectives_fn
from lib.optuna.params_alexnet import params_fn_factory as params_fn_factory_alexnet
from lib.optuna.params_transformer import params_fn_factory as params_fn_factory_transformer
from lib.optuna.splits import make_splits_fn_factory
from lib.optuna.types import FoldResult, Params, Split, Values

# Backward name: notebooks expect ``params_fn_factory`` for AlexNet-style models.
params_fn_factory = params_fn_factory_alexnet

__all__ = [
    "aggregate",
    "attrs_fn",
    "FoldResult",
    "loss_slope",
    "make_objective_engine",
    "make_splits_fn_factory",
    "objectives_fn",
    "Params",
    "params_fn_factory",
    "params_fn_factory_alexnet",
    "params_fn_factory_transformer",
    "run_fold_fn_factory",
    "Split",
    "Values",
]
