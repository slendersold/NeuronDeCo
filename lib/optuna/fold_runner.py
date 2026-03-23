"""
Один train/val фолд: датасеты → ``AdamW`` → цикл эпох с early stopping по ``patience``.
"""

from __future__ import annotations

from typing import Any, Callable, Type

import torch
import torch.nn as nn
from beartype import beartype
from torch.utils.data import DataLoader

from lib.optuna.types import FoldResult, Params, Split


@beartype
def run_fold_fn_factory(
    *,
    ModelCls: Type[nn.Module],
    device: torch.device | str,
    max_epochs: int,
    patience: int,
    TFRDataset: Type[Any],
    DataLoader: Type[DataLoader],
    train_one_epoch: Callable[..., float],
    eval_one_epoch_f1_macro: Callable[..., tuple[float, float]],
    loss_metric: Callable[[list[float]], float],
) -> Callable[[Split, Params], FoldResult]:
    """
    Замыкает гиперпараметры обучения и возвращает ``run_fold(split, params)``.

    Parameters
    ----------
    ModelCls:
        Класс модели (например :class:`lib.models.alexnet.AlexNetTFR`); конструируется как
        ``ModelCls(**params[\"model\"]).to(device)``.
    device:
        Устройство для батчей и модели.
    max_epochs, patience:
        Верхняя граница эпох и early stopping, если ``val_f1`` не улучшается.
    TFRDataset, DataLoader:
        Обычно :class:`utils.TFRDataset.TFRDataset` и ``torch.utils.data.DataLoader``.
    train_one_epoch:
        ``(model, loader, optimizer, device) -> train_loss`` (float).
    eval_one_epoch_f1_macro:
        ``(model, loader, device) -> (val_loss, macro_f1)``.
    loss_metric:
        Скаляризация списка ``val_losses`` (например :func:`lib.optuna.metrics.loss_slope`).

    Returns
    -------
    callable
        Принимает :class:`Split` и :data:`Params`; возвращает :class:`FoldResult`.
    """

    @beartype
    def _run_fold(sp: Split, params: Params) -> FoldResult:
        p_model = params["model"]
        p_optim = params["optimizer"]
        p_tr_ds = params["tr_dataset"]
        p_vl_ds = params["vl_dataset"]
        p_tr_ld = params["tr_loader"]
        p_vl_ld = params["vl_loader"]

        train_ds = TFRDataset(sp.X_train, sp.y_train, **p_tr_ds)
        val_ds = TFRDataset(sp.X_val, sp.y_val, **p_vl_ds)

        train_loader = DataLoader(train_ds, **p_tr_ld)
        val_loader = DataLoader(val_ds, **p_vl_ld)

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
