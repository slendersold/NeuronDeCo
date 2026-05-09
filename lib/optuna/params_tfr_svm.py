from __future__ import annotations

from typing import Any, Callable, Sequence

from beartype import beartype

from lib.models.tfr_transformer.preprocess import PREPROCESS_BUILDERS, SeqPool
from lib.optuna.types import TfrSvmFoldParams


@beartype
def params_fn_factory(
    *,
    num_classes: int,
    batch_size_choices: Sequence[int] = (16, 32, 64),
    preprocess_keys: Sequence[str] = ("flatten", "channel_conv", "ft_plane_conv", "pixel_weight"),
    pooling_modes: Sequence[str] = ("mean", "max", "last"),
) -> Callable[[Any], TfrSvmFoldParams]:
    """
    Пространство поиска: тот же препроцесс / пулинг, что у transformer, плюс SVM.

    Режим ``softmax`` для :class:`~lib.models.tfr_transformer.preprocess.SeqPool`
    не предлагается: без обратного распространения веса ``LazyLinear`` не обучаются.
    """

    def _params_fn(trial) -> TfrSvmFoldParams:
        preprocess_name = trial.suggest_categorical("preprocess", list(preprocess_keys))
        preprocess_mod = PREPROCESS_BUILDERS[preprocess_name]()
        batch_size = trial.suggest_categorical("batch_size", list(batch_size_choices))
        kernel = trial.suggest_categorical("kernel", ["rbf", "linear"])
        svm_C = trial.suggest_float("svm_C", 1e-2, 1e3, log=True)
        if kernel == "rbf":
            svm_gamma = trial.suggest_float("svm_gamma", 1e-5, 10.0, log=True)
        else:
            svm_gamma = "scale"

        params_dict: TfrSvmFoldParams = {
            "model": {
                "num_classes": num_classes,
                "preprocess": preprocess_mod,
                "pooling": SeqPool(mode=trial.suggest_categorical("pooling", list(pooling_modes))),
                "kernel": kernel,
                "svm_C": svm_C,
                "svm_gamma": svm_gamma,
            },
            "optimizer": {
                "lr": 1e-3,
                "weight_decay": 0.0,
            },
            "tr_dataset": {"time_crop": None},
            "tr_loader": {"batch_size": batch_size, "shuffle": True},
            "vl_dataset": {"time_crop": None},
            "vl_loader": {"batch_size": batch_size, "shuffle": False},
        }
        return params_dict

    return _params_fn
