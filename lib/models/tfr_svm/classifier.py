"""
TFR: тот же препроцессор и пулинг по времени, что в transformer-пайплайне, но
классификатор — **sklearn SVM** на векторе признаков ``(B, D)`` после пулинга.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from lib.models.tfr_transformer.preprocess import SeqPool


class TfrParadigmSvmClassifier(nn.Module):
    """
    Parameters
    ----------
    num_classes:
        ``K`` бинов для ``predict_proba`` / логитов ``(B, K)``.
    preprocess:
        ``(B, C, F, T) -> (B, T, D)`` — как в :class:`TFRTransformerWrapper`.
    pooling:
        :class:`SeqPool` с режимом не ``none``, выход ``(B, D)``.
    svm_C, kernel, svm_gamma:
        Параметры :class:`sklearn.svm.SVC` (``gamma`` для ``rbf``/``poly``/``sigmoid``;
        для ``linear`` не передаётся).
    """

    def __init__(
        self,
        *,
        num_classes: int,
        preprocess: nn.Module,
        pooling: SeqPool,
        svm_C: float = 1.0,
        kernel: str = "rbf",
        svm_gamma: float | str = "scale",
    ) -> None:
        super().__init__()
        if pooling.mode == "none":
            raise ValueError("SVM head needs pooled features; set SeqPool mode != 'none'.")
        self.num_classes = int(num_classes)
        self.preprocess = preprocess
        self.pooling = pooling
        self.svm_C = float(svm_C)
        self.kernel = str(kernel)
        self.svm_gamma: float | str = svm_gamma
        self._pipeline: Pipeline | None = None
        # ``fold_runner`` всегда строит ``AdamW(model.parameters())``. Часть препроцессоров
        # без параметров (например ``TFRToSeqFlatten``), у других веса появляются только после
        # первого ``forward`` (ленивая сборка). Без хотя бы одного ``nn.Parameter`` PyTorch
        # выдаёт ``ValueError: optimizer got an empty parameter list``. SVM-путь градиенты не
        # использует — placeholder не участвует в ``fit_from_loader`` / ``forward`` логике.
        self._adamw_placeholder = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        seq = self.preprocess(x)
        out = self.pooling(seq)
        if out.dim() != 2:
            raise ValueError(f"Expected pooled (B, D), got {tuple(out.shape)}")
        return out

    def fit_from_loader(self, loader: Any, device: torch.device | str) -> None:
        """Собирает признаки по train loader и обучает ``StandardScaler + SVC``."""
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        self.train()
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                feats = self._features(xb)
                xs.append(feats.cpu().numpy())
                ys.append(yb.detach().cpu().numpy())
        X = np.vstack(xs)
        y = np.concatenate(ys)
        kw: dict[str, Any] = {
            "C": self.svm_C,
            "kernel": self.kernel,
            "probability": True,
            "random_state": 0,
        }
        if self.kernel in ("rbf", "poly", "sigmoid"):
            kw["gamma"] = self.svm_gamma
        svc = SVC(**kw)
        self._pipeline = Pipeline([("scaler", StandardScaler()), ("svc", svc)])
        self._pipeline.fit(X, y)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._pipeline is None:
            raise RuntimeError("Call fit_from_loader before forward.")
        self.eval()
        feats = self._features(x)
        X = feats.cpu().numpy()
        proba = self._pipeline.predict_proba(X)
        return torch.from_numpy(proba).to(dtype=torch.float32, device=x.device)
