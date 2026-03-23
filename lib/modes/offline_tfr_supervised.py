"""
Оффлайн обучение **torch.nn.Module** классификаторов на батчах TFR ``(N,C,F,T)``.

В отличие от :class:`lib.modes.offline.OfflineEpochMode` (скелет с ``fit`` у sklearn-подобной
модели), здесь используется :class:`utils.TFRDataset.TFRDataset`, ``AdamW`` и те же
``train_one_epoch`` / ``eval_one_epoch_f1_macro``, что и в Optuna-фолдах.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from lib.core.contracts import ModeRunner, RunResult
from utils.TFRDataset import TFRDataset
from utils.train_eval_helpers import eval_one_epoch_f1_macro, train_one_epoch


def _meta_int(context: Mapping[str, Any], key: str, default: int) -> int:
    v = context.get(key, default)
    return int(v)


def _meta_float(context: Mapping[str, Any], key: str, default: float) -> float:
    v = context.get(key, default)
    return float(v)


class OfflineTFRSupervisedMode(ModeRunner):
    """
    Один прогон train/val с early stopping по ``patience``.

    Parameters (через ``RunContext.metadata``)
    ----------------------------------------
    max_epochs:
        Верхняя граница эпох (по умолчанию ``30``).
    patience:
        Early stopping если ``val_f1`` не растёт (по умолчанию ``5``).
    test_size:
        Доля валидации (по умолчанию ``0.2``).
    batch_size:
        По умолчанию ``32``.
    lr:
        AdamW learning rate (по умолчанию ``1e-3``).
    weight_decay:
        По умолчанию ``1e-4``.
    seed:
        ``random_state`` для split (по умолчанию из ``RunContext.seed`` если положить в metadata,
        иначе ``42``).

    **Входные данные**

    * ``X``: ``numpy.ndarray``, форма ``(N, C, F, T)``, float.
    * ``y``: ``numpy.ndarray``, форма ``(N,)``, целочисленные метки ``0..K-1``.

    **Модель**

    Экземпляр ``nn.Module`` с ``forward(x) -> logits`` для ``x`` формы ``(B,C,F,T)``.
    """

    def run(
        self,
        model: Any,
        X: Any,
        y: Any,
        *,
        context: Mapping[str, Any],
    ) -> RunResult:
        if not isinstance(model, nn.Module):
            raise TypeError(
                "OfflineTFRSupervisedMode expects an nn.Module (e.g. AlexNetTFR, TFRTransformerWrapper)."
            )
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if X.ndim != 4:
            raise ValueError(f"X must be (N,C,F,T), got shape {X.shape}")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(f"y must be (N,) matching X, got {y.shape} vs N={X.shape[0]}")

        max_epochs = _meta_int(context, "max_epochs", 30)
        patience = _meta_int(context, "patience", 5)
        test_size = _meta_float(context, "test_size", 0.2)
        batch_size = _meta_int(context, "batch_size", 32)
        lr = _meta_float(context, "lr", 1e-3)
        weight_decay = _meta_float(context, "weight_decay", 1e-4)
        seed = int(context.get("seed", 42))

        device = torch.device(context.get("device", "cpu"))

        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )

        train_ds = TFRDataset(X_tr, y_tr, time_crop=None)
        val_ds = TFRDataset(X_va, y_va, time_crop=None)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_f1 = -1.0
        bad = 0
        final_val_loss = 0.0
        train_losses: list[float] = []
        val_losses: list[float] = []
        val_f1s: list[float] = []

        for _ in range(max_epochs):
            tr = train_one_epoch(model, train_loader, opt, device)
            va_loss, va_f1 = eval_one_epoch_f1_macro(model, val_loader, device)
            train_losses.append(float(tr))
            val_losses.append(float(va_loss))
            val_f1s.append(float(va_f1))
            if va_f1 > best_f1:
                best_f1 = float(va_f1)
                final_val_loss = float(va_loss)
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        return RunResult(
            name="offline_tfr_supervised",
            metrics={
                "best_val_f1_macro": best_f1,
                "final_val_loss": final_val_loss,
                "epochs_ran": float(len(val_f1s)),
            },
            artifacts={
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_f1s": val_f1s,
            },
        )
