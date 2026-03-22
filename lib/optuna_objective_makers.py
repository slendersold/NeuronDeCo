from pydantic import BaseModel, Field, create_model
from beartype import beartype

import numpy as np
from utils.TFRDataset import TFRDataset

import torch
from torch.utils.data import DataLoader

from utils.train_eval_helpers import train_one_epoch, eval_one_epoch_f1_macro

from dataclasses import dataclass
from typing import Type, Any, Callable, Dict, List, Sequence, Tuple, Union, Optional


def loss_slope(losses):
    # линейный тренд: loss ~ a*epoch + b, хотим a < 0, и чем меньше (более отрицательно) — тем лучше
    tail = losses[-10:] if len(losses) >= 10 else losses
    e = np.arange(len(tail), dtype=np.float64)
    return float(np.polyfit(e, np.array(tail, dtype=np.float64), 1)[0])

# ---------- типы ----------
Params = Dict[str, Dict[str, Any]]
Attrs  = Dict[str, Any]

@dataclass(frozen=True)
class Split:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray

@dataclass
class FoldResult:
    split: str
    best_f1: float
    loss_metric: float
    curves: Dict[str, List[float]]
    # можно расширять: cm, best_epoch, etc.

Values = Union[float, Sequence[float]]  # 1 objective или N objectives

'''
    modules
'''

def make_splits_fn_factory(test_size: float, seed: int, cv: bool):
    def _make_splits(X, y):
        if cv:
            from sklearn.model_selection import StratifiedKFold
            n_splits = max(2, int(round(1 / test_size)))
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            out = []
            for i, (tr, va) in enumerate(skf.split(X, y)):
                out.append(Split(f"fold{i}", X[tr], y[tr], X[va], y[va]))
            return out
        else:
            from sklearn.model_selection import train_test_split
            X_tr, X_va, y_tr, y_va = train_test_split(
                X, y, test_size=test_size, random_state=seed, stratify=y
            )
            return [Split("holdout", X_tr, y_tr, X_va, y_va)]
    return _make_splits

@beartype
def run_fold_fn_factory(
    *,
    ModelCls: Type[Any],
    device: Any,
    max_epochs: int,
    patience: int,
    TFRDataset: Type[Any],
    DataLoader: Type[Any],
    train_one_epoch: Callable[..., Any],
    eval_one_epoch_f1_macro: Callable[..., Any],
    loss_metric: Callable[..., float],
) -> Callable[[Split, Params], FoldResult]:

    @beartype
    def _run_fold(sp: Split, params: Params) -> FoldResult:
        # достаём параметры
        p_model = params["model"]
        p_optim = params["optimizer"]
        p_tr_ds = params["tr_dataset"]
        p_vl_ds = params["vl_dataset"]
        p_tr_ld = params["tr_loader"]
        p_vl_ld = params["vl_loader"]

        train_ds = TFRDataset(sp.X_train, sp.y_train, **p_tr_ds)
        val_ds   = TFRDataset(sp.X_val,   sp.y_val,   **p_vl_ds)

        train_loader = DataLoader(train_ds,  **p_tr_ld)
        val_loader   = DataLoader(val_ds,    **p_vl_ld)

        model = ModelCls(**p_model).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), **p_optim)

        bad = 0
        best_f1 = -1.0
        train_losses, val_losses, val_f1s = [], [], []

        for _epoch in range(max_epochs):
            tr_loss = train_one_epoch(model, train_loader, optimizer, device)
            va_loss, va_f1 = eval_one_epoch_f1_macro(model, val_loader, device)
            train_losses.append(float(tr_loss))
            val_losses.append(float(va_loss))
            val_f1s.append(float(va_f1))

            if va_f1 > best_f1:
                best_f1 = float(va_f1)
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        metric = float(loss_metric(val_losses))

        return FoldResult(
            split=sp.name,
            best_f1=best_f1,
            loss_metric=metric,
            curves={"train_losses": train_losses, "val_losses": val_losses, "val_f1s": val_f1s},
        )

    return _run_fold

@beartype
def params_fn_factory(
    *,
    in_channels: int,
    num_classes: int,
    ) -> Callable[[Any], Params]:
    def _params_fn(trial) -> Params:
        params_dict = {}
        params_dict["model"] = {
                "in_channels": in_channels, 
                "num_classes": num_classes,
                "dropout": trial.suggest_float("dropout", 0.0, 0.7),
            }
        params_dict["optimizer"] = {
                "lr": trial.suggest_float("lr", 1e-5, 3e-3, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            }
        params_dict["tr_dataset"] = {
                "time_crop": None,
                # в будущем:
                # зависимые:
                # "time_crop": (
                #     trial.suggest_int("time_crop", 200, 278)  # или T из ctx, если добавишь ctx
                #     if trial.suggest_categorical("time_crop_on", [0, 1]) else None
                # ),
            }
        params_dict["tr_loader"] = {
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
                "shuffle": True
                # можно добавить:
                # "num_workers": 4,
                # "pin_memory": True,
            }
        params_dict["vl_dataset"] = dict(params_dict["tr_dataset"])
        params_dict["vl_loader"] = dict(params_dict["tr_loader"])
        params_dict["vl_loader"]["shuffle"] = False
        return params_dict
    return _params_fn

@beartype
def objectives_fn(folds: List[FoldResult], agg_mode: str) -> Tuple[float, float]:
    f1 = aggregate([f.best_f1 for f in folds], agg_mode)
    slope = aggregate([f.loss_metric for f in folds], agg_mode)
    return (f1, slope)  # 2 objectives

def attrs_fn(trial, params, folds, values):
    return {
        "params": params,
        "fold_best_f1s": [f.best_f1 for f in folds],
        "fold_slopes": [f.loss_metric for f in folds],
        "fold_curves": [{"split": f.split, **f.curves} for f in folds],
        "objectives": values,
        "cv_mode": "kfold" if len(folds) > 1 else "holdout",
    }

'''
Инвариантный “движок” objective

Что инвариантно: сплиты → прогоны фолдов → objectives_fn → attrs_fn → return.

Важные ограничения/правила (чтобы “инвариантность” не сломалась)

params_fn(trial) должен быть детерминированным: одна и та же архитектура suggest_* при каждом вызове.

Границы suggest_* лучше делать не зависящими от fold (например, верхняя граница time_crop из X.shape[3], а не из X_train.shape[3] каждого фолда).

Если хочешь “контекст” (например T = X.shape[3]) в params_fn, сделай params_fn_factory(ctx) и замкни T в замыкание.
'''

# ---------- инвариантные утилиты ----------
def aggregate(values: List[float], mode: str) -> float:
    arr = np.asarray(values, dtype=float)
    if mode == "median":
        return float(np.median(arr))
    if mode == "mean":
        return float(np.mean(arr))
    raise ValueError(f"Unknown aggregate mode: {mode}")

# ---------- движок ----------
def make_objective_engine(
    *,
    X, y,
    make_splits_fn: Callable[[np.ndarray, np.ndarray], List[Split]],
    run_fold_fn: Callable[[Split, Params], FoldResult],
    aggregate_mode: str = "median",

    # плагины пользователя:
    params_fn: Callable[[Any], Params],
    objectives_fn: Callable[[List[FoldResult], str], Values],
    attrs_fn: Optional[Callable[[Any, Params, List[FoldResult], Values], Attrs]] = None,
):
    X = np.asarray(X)
    y = np.asarray(y)
    splits = make_splits_fn(X, y)

    def objective(trial):
        params = params_fn(trial)

        fold_results: List[FoldResult] = []
        for sp in splits:
            fr = run_fold_fn(sp, params)
            fold_results.append(fr)

        values = objectives_fn(fold_results, aggregate_mode)

        if attrs_fn is not None:
            attrs = attrs_fn(trial, params, fold_results, values)
            for k, v in attrs.items():
                trial.set_user_attr(k, v)

        # Optuna ждёт float или tuple/list floats
        if isinstance(values, (list, tuple, np.ndarray)):
            return tuple(float(x) for x in values)
        return float(values)

    return objective