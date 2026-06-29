"""
Один проход train / eval с ``cross_entropy`` и macro-F1 (sklearn).

Ожидается модель ``logits = model(x)`` для ``x`` формы ``(B,C,F,T)``, ``logits`` — ``(B,K)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


def train_one_epoch(model: Any, loader: Any, optimizer: Any, device: Any) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_one_epoch_f1_macro(model: Any, loader: Any, device: Any) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    n = 0
    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        pred = logits.argmax(dim=1)
        all_pred.append(pred.cpu().numpy())
        all_true.append(y.cpu().numpy())
    val_loss = total_loss / max(n, 1)
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    f1m = float(f1_score(y_true, y_pred, average="macro"))
    return val_loss, f1m
