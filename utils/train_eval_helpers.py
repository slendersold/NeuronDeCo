import numpy as np

import torch
import torch.nn.functional as F

from sklearn.metrics import f1_score
# ---- train/eval helpers ----

def train_one_epoch(model, loader, optimizer, device):
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
    return total_loss / n


@torch.no_grad()
def eval_one_epoch_f1_macro(model, loader, device):
    model.eval()
    total_loss = 0.0
    n = 0

    all_pred = []
    all_true = []

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

    val_loss = total_loss / n
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)

    f1m = f1_score(y_true, y_pred, average="macro")
    return val_loss, f1m