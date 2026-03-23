#!/usr/bin/env python3
"""
Оффлайн обучение AlexNet и TFR-Transformer на **синтетических** TFR-подобных массивах.

Не требует EDF/TFR файлов: ``X`` — случайный шум, ``y`` — случайные метки (классы
разделимы только статистикой шума; цель — проверить пайплайн).

Запуск из каталога ``NeuronDeCo/``::

    python examples/offline_synthetic_tfr.py

Нужен режим ``offline_tfr_supervised`` (см. :class:`lib.modes.offline_tfr_supervised.OfflineTFRSupervisedMode`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib import MODEL_REGISTRY, MODE_REGISTRY, PipelineRunner, RunContext


def make_synthetic_xy(
    *,
    n: int = 128,
    c: int = 7,
    f: int = 80,
    t: int = 64,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Формы согласованы с AlexNet (достаточно крупные ``F,T`` после пулов) и с Transformer
    (``seq_len == T`` при preprocess ``flatten``).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, c, f, t)).astype(np.float32)
    y = rng.integers(0, 2, size=(n,), dtype=np.int64)
    return X, y


def run_alexnet() -> None:
    X, y = make_synthetic_xy()
    model = MODEL_REGISTRY["alexnet"](in_channels=7, num_classes=2)
    mode = MODE_REGISTRY["offline_tfr_supervised"]()
    ctx = RunContext(
        run_id="synthetic-alexnet",
        device="cpu",
        seed=42,
        metadata={
            "max_epochs": 8,
            "patience": 3,
            "batch_size": 16,
            "lr": 1e-3,
            "test_size": 0.25,
        },
    )
    out = PipelineRunner(mode, model, ctx).run(X, y)
    print("[AlexNet]", out)


def run_transformer() -> None:
    X, y = make_synthetic_xy()
    _, _, _, t_bins = X.shape
    model = MODEL_REGISTRY["tfr_transformer"](
        num_classes=2,
        seq_len=t_bins,
        embed_dim=64,
        nhead=4,
        dim_fc=128,
        num_layers=2,
        dropout=0.2,
        preprocess="flatten",
        pooling="mean",
    )
    mode = MODE_REGISTRY["offline_tfr_supervised"]()
    ctx = RunContext(
        run_id="synthetic-transformer",
        device="cpu",
        seed=42,
        metadata={
            "max_epochs": 8,
            "patience": 3,
            "batch_size": 16,
            "lr": 3e-4,
            "test_size": 0.25,
        },
    )
    out = PipelineRunner(mode, model, ctx).run(X, y)
    print("[Transformer]", out)


def main() -> None:
    run_alexnet()
    run_transformer()


if __name__ == "__main__":
    main()
