from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

from beartype import beartype

from lib.models.tfr_transformer.preprocess import PREPROCESS_BUILDERS, SeqPool
from lib.optuna.types import TransformerFoldParams


@beartype
def params_fn_factory(
    *,
    num_classes: int,
    seq_len: int,
    batch_size_choices: Sequence[int] = (16, 32, 64),
    embed_dim_choices: Sequence[int] = (32, 64, 128, 256),
    nhead_choices: Iterable[int] = (2, 4, 8, 16),
    dim_fc_choices: Sequence[int] = (64, 128, 256, 512),
    preprocess_keys: Sequence[str] = ("flatten", "channel_conv", "ft_plane_conv", "pixel_weight"),
    pooling_modes: Sequence[str] = ("mean", "softmax"),
) -> Callable[[Any], TransformerFoldParams]:
    """
    Search space aligned with ``transformer_17_03_26.ipynb``.

    ``seq_len`` must be ≥ number of time bins after preprocessing (positional buffer size).
    Pass ``X.shape[3]`` (or your cropped T) from the loaded tensor.
    """

    def _params_fn(trial) -> TransformerFoldParams:
        embed_dim = trial.suggest_categorical("embed_dim", list(embed_dim_choices))
        possible_heads = [h for h in nhead_choices if embed_dim % h == 0]
        if not possible_heads:
            possible_heads = [1]
        preprocess_name = trial.suggest_categorical("preprocess", list(preprocess_keys))
        preprocess_mod = PREPROCESS_BUILDERS[preprocess_name]()

        batch_size = trial.suggest_categorical("batch_size", list(batch_size_choices))
        base_dropout = trial.suggest_float("dropout", 0.1, 0.6)
        params_dict: TransformerFoldParams = {
            "model": {
                "num_classes": num_classes,
                "seq_len": seq_len,
                "embed_dim": embed_dim,
                "nhead": trial.suggest_categorical("nhead", possible_heads),
                "dim_fc": trial.suggest_categorical("dim_fc", list(dim_fc_choices)),
                "num_layers": trial.suggest_int("num_layers", 1, 8),
                "dropout": base_dropout,
                "encoder_dropout": trial.suggest_float("encoder_dropout", 0.05, 0.6),
                "mlp_dropout": trial.suggest_float("mlp_dropout", 0.05, 0.6),
                "use_conv": trial.suggest_categorical("use_conv", [False, True]),
                "conv_kernel_size": trial.suggest_categorical("conv_kernel_size", [3, 5, 7]),
                "conv_dropout": trial.suggest_float("conv_dropout", 0.0, 0.6),
                "pooling": SeqPool(mode=trial.suggest_categorical("pooling", list(pooling_modes))),
                "preprocess": preprocess_mod,
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
        return params_dict

    return _params_fn
