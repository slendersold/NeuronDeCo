from __future__ import annotations

from typing import Any, Callable

from beartype import beartype

from lib.optuna.types import AlexNetFoldParams


@beartype
def params_fn_factory(
    *,
    in_channels: int,
    num_classes: int,
) -> Callable[[Any], AlexNetFoldParams]:
    """
    Пространство поиска для :class:`lib.models.alexnet.model.AlexNetTFR`.

    * **model:** ``in_channels``, ``num_classes``, ``dropout``.
    * **optimizer:** ``lr``, ``weight_decay`` (AdamW).
    * **loaders:** один общий ``batch_size`` для train/val.
    """

    def _params_fn(trial) -> AlexNetFoldParams:
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        return {
            "model": {
                "in_channels": in_channels,
                "num_classes": num_classes,
                "dropout": trial.suggest_float("dropout", 0.0, 0.7),
            },
            "optimizer": {
                "lr": trial.suggest_float("lr", 1e-5, 3e-3, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            },
            "tr_dataset": {"time_crop": None},
            "tr_loader": {"batch_size": batch_size, "shuffle": True},
            "vl_dataset": {"time_crop": None},
            "vl_loader": {"batch_size": batch_size, "shuffle": False},
        }

    return _params_fn
