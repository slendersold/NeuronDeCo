"""
Контракты ядра: результаты прогонов, режимы, модели.

Два стиля «модели»:

* **SklearnLikeModel** — ``fit`` / ``predict`` (скелет режимов ``modes/*``).
* **TorchTFRClassifier** — ``forward(x) -> logits`` для батчей TFR ``(B,C,F,T)``;
  соответствует ``nn.Module``, используемому в Optuna / ``train_eval_helpers``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Protocol, TypeAlias, runtime_checkable

import torch
from torch import Tensor


# --- Результаты и контекст выполнения ---


@dataclass
class RunResult:
    """
    Итог одного прогона режима (например offline batch).

    Attributes
    ----------
    name:
        Идентификатор режима (строка).
    metrics:
        Скалярные метрики, ключ — имя, значение — float.
    artifacts:
        Произвольные вложения (JSON-подобные, пути к файлам, отладочные структуры).
    """

    name: str
    metrics: Dict[str, float]
    artifacts: Dict[str, Any]


RunMetadata: TypeAlias = Mapping[str, Any]
"""Произвольные метаданные прогона (логируются / прокидываются в ``ModeRunner``)."""


# --- Модели ---


@runtime_checkable
class SklearnLikeModel(Protocol):
    """
    Минимальный контракт для ``PipelineRunner`` + ``modes/*``.

    **Данные:** ``X``, ``y`` без жёсткой типизации здесь (могут быть ndarray, списки и т.д.
    в зависимости от режима). Для TFR-классификации смотри также ``TorchTFRClassifier``.
    """

    def fit(self, X: Any, y: Any) -> SklearnLikeModel:
        ...

    def predict(self, X: Any) -> Any:
        ...


# Обратная совместимость имён
Model = SklearnLikeModel


@runtime_checkable
class TorchTFRClassifier(Protocol):
    """
    Классификатор одной конфигурации TFR (батч на GPU/CPU).

    **Вход ``x``:** ``FloatTensor``, форма ``(B, C, F, T)``.

        * ``B`` — размер minibatch (переменный).
        * ``C, F, T`` — фиксированы после препроцессинга датасета.

    **Выход:** ``FloatTensor``, форма ``(B, K)`` — логиты классов (без softmax),
    ``K = num_classes``. Ожидается :func:`torch.nn.functional.cross_entropy`.

    **Вариабельность:** между датасетами меняются ``C, F, T`` и ``K``; внутри одного
    study / одной модели они фиксированы. Разные архитектуры (AlexNet vs Transformer)
    реализуют этот контракт по-разному, но сохраняют эти формы на выходе.
    """

    def forward(self, x: Tensor) -> Tensor:
        ...


@runtime_checkable
class ModeRunner(Protocol):
    """Стратегия применения модели к данным при заданных метаданных прогона."""

    def run(
        self,
        model: SklearnLikeModel,
        X: Any,
        y: Any,
        *,
        context: RunMetadata,
    ) -> RunResult:
        ...


@runtime_checkable
class ObjectiveEngine(Protocol):
    """
    Фабрика callable для Optuna.

    Ожидаемые ``X``, ``y`` в типичном сценарии — :class:`numpy.ndarray` в формах
    ``(N,C,F,T)`` и ``(N,)``; сигнатура намеренно остаётся ``Any`` для скриптов,
    которые ещё не привели данные к ndarray.
    """

    def build(self, X: Any, y: Any, **kwargs: Any) -> Any:
        ...

