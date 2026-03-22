from pydantic import BaseModel, Field
from beartype import beartype

import numpy as np
from utils.TFRDataset import TFRDataset

import torch
from torch.utils.data import DataLoader

from utils.train_eval_helpers import train_one_epoch, eval_one_epoch_f1_macro

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union, Optional

def make_objective(
    X, 
    y, 
    test_size, 
    seed, 
    device, 
    ModelCls, 
    in_channels=7, 
    num_classes=2, 
    max_epochs = 30, 
    patience = 6,     
    cv=False,                 # None/0 -> holdout, int -> kfold, или объект Splitter
    cv_aggregate="mean",     # "mean" | "median"
): 

    X = np.asarray(X)
    y = np.asarray(y)

    # --- заранее готовим сплиты внутри генератора ---
    if cv :
        from sklearn.model_selection import StratifiedKFold

        splitter = StratifiedKFold(n_splits=max(2, int(round(1/test_size))), shuffle=True, random_state=seed)

        splits = []
        for fold_id, (tr_idx, va_idx) in enumerate(splitter.split(X, y)):
            splits.append((f"fold{fold_id}", X[tr_idx], y[tr_idx], X[va_idx], y[va_idx]))
        
    else:
        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=test_size,
            random_state=seed,
            stratify=y
        )
        splits = [("holdout", X_train, y_train, X_val, y_val)]
    
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.7)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        crop_on = trial.suggest_categorical("time_crop_on", [0, 1])
        time_crop = None

        fold_best_f1s = []
        fold_curves = []  # чтобы сохранять кривые по каждому фолду

        # --- прогоняем все сплиты (1 holdout или K фолдов) ---
        for split_name, X_train, y_train, X_val, y_val in splits:
            if crop_on:
                # X_train shape: (N, C, F, T) -> time dim = 3
                time_crop = trial.suggest_int("time_crop", 200, X_train.shape[3])

            train_ds = TFRDataset(X_train, y_train, time_crop=time_crop)
            val_ds   = TFRDataset(X_val,   y_val,   time_crop=None)

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

            model = ModelCls(in_channels=in_channels, num_classes=num_classes, dropout=dropout).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

            bad = 0
            best_f1 = -1.0
            train_losses, val_losses, val_f1s = [], [], []

            for epoch in range(max_epochs):
                tr_loss = train_one_epoch(model, train_loader, optimizer, device)
                va_loss, va_f1 = eval_one_epoch_f1_macro(model, val_loader, device)

                train_losses.append(tr_loss)
                val_losses.append(va_loss)
                val_f1s.append(va_f1)

                if va_f1 > best_f1:
                    best_f1 = va_f1
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        break

            fold_best_f1s.append(best_f1)
            fold_curves.append({
                "split": split_name,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_f1s": val_f1s,
            })

            # report/prune: обычно репортят агрегированный score как один step
            trial.report(score, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # --- агрегируем метрику по фолдам ---
        if cv_aggregate == "median":
            score = float(np.median(fold_best_f1s))
        else:
            score = float(np.mean(fold_best_f1s))

        # сохраним детали
        trial.set_user_attr("fold_best_f1s", fold_best_f1s)
        trial.set_user_attr("fold_curves", fold_curves)
        trial.set_user_attr("cv_mode", "holdout" if (cv is None or cv == 0) else "kfold")

        return score

    return objective

def loss_slope(losses):
    # линейный тренд: loss ~ a*epoch + b, хотим a < 0, и чем меньше (более отрицательно) — тем лучше
    e = np.arange(len(losses), dtype=np.float64)
    return float(np.polyfit(e, np.array(losses, dtype=np.float64), 1)[0])


def make_multi_objective(
    X, y,
    test_size,
    seed,
    device,
    ModelCls,
    in_channels=7,
    num_classes=2,
    max_epochs=30,
    patience=6,
    cv=False,                 # None/0 -> holdout, int -> kfold, или объект Splitter
    cv_aggregate="median",     # "mean" | "median"
):


    X = np.asarray(X)
    y = np.asarray(y)

    # --- заранее готовим сплиты внутри генератора ---
    if cv:
        from sklearn.model_selection import StratifiedKFold

        splitter = StratifiedKFold(n_splits=max(2, int(round(1/test_size))), shuffle=True, random_state=seed)
        splits = []
        for fold_id, (tr_idx, va_idx) in enumerate(splitter.split(X, y)):
            splits.append((f"fold{fold_id}", X[tr_idx], y[tr_idx], X[va_idx], y[va_idx]))
    else:
        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=test_size,
            random_state=seed,
            stratify=y
        )
        splits = [("holdout", X_train, y_train, X_val, y_val)]

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.7)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        # !Заметка Вариативный кроп был все это время, только для трейна
        # crop_on = trial.suggest_categorical("time_crop_on", [0, 1])
        time_crop = None

        fold_best_f1s = []
        fold_slope = []
        fold_curves = []  # чтобы сохранять кривые по каждому фолду

        # --- прогоняем все сплиты (1 holdout или K фолдов) ---
        for fold_id, (split_name, X_train, y_train, X_val, y_val) in enumerate(splits):
            # if crop_on:
            #     # X_train shape: (N, C, F, T) -> time dim = 3
            #     time_crop = trial.suggest_int("time_crop", 200, X_train.shape[3])

            train_ds = TFRDataset(X_train, y_train, time_crop=time_crop)
            val_ds   = TFRDataset(X_val,   y_val,   time_crop=None)

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

            model = ModelCls(in_channels=in_channels, num_classes=num_classes, dropout=dropout).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

            bad = 0
            best_f1 = -1.0
            train_losses, val_losses, val_f1s = [], [], []

            for epoch in range(max_epochs):
                tr_loss = train_one_epoch(model, train_loader, optimizer, device)
                va_loss, va_f1 = eval_one_epoch_f1_macro(model, val_loader, device)

                train_losses.append(tr_loss)
                val_losses.append(va_loss)
                val_f1s.append(va_f1)

                if va_f1 > best_f1:
                    best_f1 = va_f1
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
            
            # ### Если посходит быстрый выброс из кроссвалидации. то прерывание происходит по вине прунинга
            # trial.report(best_f1, step=fold_id)
            # if trial.should_prune():
            #     raise optuna.TrialPruned()
            # Прунинг в мультиобджектив не работает (

            # slope по хвосту (стабильнее)
            tail = val_losses[-10:] if len(val_losses) >= 10 else val_losses
            slope = loss_slope(tail)

            fold_best_f1s.append(best_f1)
            fold_slope.append(slope)
            fold_curves.append({
                "split": split_name,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_f1s": val_f1s,
            })

        # --- агрегируем метрику по фолдам ---
        if cv_aggregate == "median":
            score = (float(np.median(fold_best_f1s)), float(np.median(fold_slope)))
        else:
            score = (float(np.mean(fold_best_f1s)), float(np.mean(fold_slope)))

        # сохраним детали
        trial.set_user_attr("fold_best_f1s", fold_best_f1s)
        trial.set_user_attr("fold_curves", fold_curves)
        trial.set_user_attr("cv_mode", "holdout" if (cv is None or cv == 0) else "kfold")

        return score[0], score[1]

    return objective

# ---------- типы ----------
Params = Dict[str, Any]
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
    slope: float
    curves: Dict[str, List[float]]
    # можно расширять: cm, best_epoch, etc.

Values = Union[float, Sequence[float]]  # 1 objective или N objectives

'''
    modules
'''

def make_splits_fn_factory(test_size: float, seed: int, cv: bool): 
    # -> def _make_splits(X, y) -> List[Tuple(
        # name: str, 
        # X_train: np.ndarray, 
        # y_train: np.ndarray, 
        # X_val: np.ndarray, 
        # y_val: np.ndarray
    # )]:
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

def run_fold_fn_factory(
    *,
    ModelCls, device, in_channels, num_classes,
    max_epochs, patience,
    TFRDataset, DataLoader,
    train_one_epoch, eval_one_epoch_f1_macro,
    loss_slope, torch
): # -> def _run_fold(sp: Split, params: Params) -> FoldResult:
    @beartype
    def _run_fold(sp: Split, params: Params) -> FoldResult:
        # достаём параметры
        lr = params["optimiser"]["lr"]
        wd = params["optimiser"]["weight_decay"]
        model_params = params["model"]
        batch_size = params["loader"]["batch_size"]
        time_crop = params["loader"].get("time_crop", None)


        train_ds = TFRDataset(sp.X_train, sp.y_train, time_crop=time_crop)
        val_ds   = TFRDataset(sp.X_val,   sp.y_val,   time_crop=None)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        model = ModelCls(in_channels=in_channels, num_classes=num_classes, **model_params).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

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

        tail = val_losses[-10:] if len(val_losses) >= 10 else val_losses
        slope = float(loss_slope(tail))

        return FoldResult(
            split=sp.name,
            best_f1=best_f1,
            slope=slope,
            curves={"train_losses": train_losses, "val_losses": val_losses, "val_f1s": val_f1s},
        )
    return _run_fold

def params_fn(trial):
    return {
        "loader":{
            # независимые:
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            # зависимые:
            # "time_crop": (
            #     trial.suggest_int("time_crop", 200, 278)  # или T из ctx
            #     if trial.suggest_categorical("time_crop_on", [0, 1]) else None
            # ),
            "time_crop": None,
        },
        "model":{
            "dropout": trial.suggest_float("dropout", 0.0, 0.7),
        },
        "optimiser":{
            "lr": trial.suggest_float("lr", 1e-5, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),            
        },
    }

@beartype
def objectives_fn(folds: List[FoldResult], agg_mode: str) -> Tuple[float, float]:
    f1 = aggregate([f.best_f1 for f in folds], agg_mode)
    slope = aggregate([f.slope for f in folds], agg_mode)
    return (f1, slope)  # 2 objectives

def attrs_fn(trial, params, folds, values):
    return {
        "params": params,
        "fold_best_f1s": [f.best_f1 for f in folds],
        "fold_slopes": [f.slope for f in folds],
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
): # -> def objective(trial)
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