"""
Train / eval для :class:`~lib.models.tfr_svm.classifier.TfrParadigmSvmClassifier`:

* первый проход ``train_one_epoch`` вызывает ``fit_from_loader`` (StandardScaler + SVC);
* ``eval`` — negative log-likelihood по ``predict_proba``, macro-F1 по argmax вероятностей.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

_PROBA_EPS = 1e-12


def train_one_epoch_svm(model: Any, loader: Any, optimizer: Any, device: Any) -> float:
    """
    Один вызов обучает SVM ровно один раз (пока ``_pipeline`` не задан); оптимизатор не используется.
    """
    del optimizer
    if getattr(model, "_pipeline", None) is None:
        if not hasattr(model, "fit_from_loader"):
            raise TypeError("Expected TfrParadigmSvmClassifier with fit_from_loader.")
        model.fit_from_loader(loader, device)
    return 0.0


@torch.no_grad()
def eval_one_epoch_svm_f1_macro(model: Any, loader: Any, device: Any) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    n = 0
    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        proba = model(x)
        log_p = torch.log(proba.clamp_min(_PROBA_EPS))
        loss = F.nll_loss(log_p, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        pred = proba.argmax(dim=1)
        all_pred.append(pred.cpu().numpy())
        all_true.append(y.cpu().numpy())
    val_loss = total_loss / max(n, 1)
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    f1m = float(f1_score(y_true, y_pred, average="macro"))
    return val_loss, f1m
