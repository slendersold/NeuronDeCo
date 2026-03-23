#!/usr/bin/env python3
"""
Минимальный запуск **Optuna** на синтетических ``X (N,C,F,T)``, ``y (N,)``.

Проверяет цепочку ``make_objective_engine`` → ``study.optimize`` (без файлов TFR).

Запуск из ``NeuronDeCo/`` с окружением, где установлены ``optuna``, ``torch``, …::

    conda activate /path/to/cursor_neuron/conda-env
    # или
    /path/to/cursor_neuron/conda-env/bin/python examples/optuna_synthetic_tfr.py

    # только AlexNet (по умолчанию оба)
    python examples/optuna_synthetic_tfr.py --alexnet-only

    # только Transformer
    python examples/optuna_synthetic_tfr.py --transformer-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import optuna
from optuna.trial import TrialState
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.models.alexnet import AlexNetTFR
from lib.models.tfr_transformer import TFRTransformerWrapper
from lib.optuna import (
    attrs_fn,
    loss_slope,
    make_objective_engine,
    make_splits_fn_factory,
    objectives_fn,
    params_fn_factory,
    params_fn_factory_transformer,
    run_fold_fn_factory,
)
from utils.TFRDataset import TFRDataset
from utils.train_eval_helpers import eval_one_epoch_f1_macro, train_one_epoch


def make_synthetic_xy(
    *,
    n: int = 96,
    c: int = 7,
    f: int = 80,
    t: int = 64,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, c, f, t)).astype(np.float32)
    y = rng.integers(0, 2, size=(n,), dtype=np.int64)
    return X, y


def run_optuna_alexnet(*, n_trials: int, seed: int) -> None:
    X, y = make_synthetic_xy(seed=seed)
    device = torch.device("cpu")
    objective = make_objective_engine(
        X=X,
        y=y,
        make_splits_fn=make_splits_fn_factory(test_size=0.25, seed=seed, cv=False),
        run_fold_fn=run_fold_fn_factory(
            ModelCls=AlexNetTFR,
            device=device,
            max_epochs=2,
            patience=1,
            TFRDataset=TFRDataset,
            DataLoader=DataLoader,
            train_one_epoch=train_one_epoch,
            eval_one_epoch_f1_macro=eval_one_epoch_f1_macro,
            loss_metric=loss_slope,
        ),
        aggregate_mode="median",
        params_fn=params_fn_factory(in_channels=X.shape[1], num_classes=2),
        objectives_fn=objectives_fn,
        attrs_fn=attrs_fn,
    )
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(directions=["maximize", "minimize"], sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    done = [tr for tr in study.trials if tr.state == TrialState.COMPLETE]
    t = study.best_trials[0] if study.best_trials else (done[-1] if done else study.trials[-1])
    print(f"[Optuna AlexNet] n_trials={n_trials} trial={t.number} values={t.values} params={t.params}")


def run_optuna_transformer(*, n_trials: int, seed: int) -> None:
    X, y = make_synthetic_xy(seed=seed)
    t_bins = int(X.shape[3])
    device = torch.device("cpu")
    objective = make_objective_engine(
        X=X,
        y=y,
        make_splits_fn=make_splits_fn_factory(test_size=0.25, seed=seed, cv=False),
        run_fold_fn=run_fold_fn_factory(
            ModelCls=TFRTransformerWrapper,
            device=device,
            max_epochs=2,
            patience=1,
            TFRDataset=TFRDataset,
            DataLoader=DataLoader,
            train_one_epoch=train_one_epoch,
            eval_one_epoch_f1_macro=eval_one_epoch_f1_macro,
            loss_metric=loss_slope,
        ),
        aggregate_mode="median",
        params_fn=params_fn_factory_transformer(
            num_classes=2,
            seq_len=t_bins,
            batch_size_choices=(8, 16),
            embed_dim_choices=(32, 64),
            dim_fc_choices=(64, 128),
        ),
        objectives_fn=objectives_fn,
        attrs_fn=attrs_fn,
    )
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(directions=["maximize", "minimize"], sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    done = [tr for tr in study.trials if tr.state == TrialState.COMPLETE]
    t = study.best_trials[0] if study.best_trials else (done[-1] if done else study.trials[-1])
    print(f"[Optuna Transformer] n_trials={n_trials} trial={t.number} values={t.values} params={t.params}")


def main() -> None:
    p = argparse.ArgumentParser(description="Optuna smoke test on synthetic TFR arrays.")
    p.add_argument("--n-trials", type=int, default=3, help="Number of Optuna trials per model.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alexnet-only", action="store_true")
    p.add_argument("--transformer-only", action="store_true")
    args = p.parse_args()

    if args.transformer_only and not args.alexnet_only:
        run_optuna_transformer(n_trials=args.n_trials, seed=args.seed)
        return
    if args.alexnet_only and not args.transformer_only:
        run_optuna_alexnet(n_trials=args.n_trials, seed=args.seed)
        return
    run_optuna_alexnet(n_trials=args.n_trials, seed=args.seed)
    run_optuna_transformer(n_trials=args.n_trials, seed=args.seed + 1)


if __name__ == "__main__":
    main()
