from __future__ import annotations

from typing import Any, Callable, Type

import numpy as np
import torch
from beartype import beartype
from torch.utils.data import DataLoader

from lib.optuna.types import FoldResult, Params, Split


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
