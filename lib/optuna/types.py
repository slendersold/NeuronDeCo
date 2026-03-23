"""
Типы для движка Optuna: сплиты, результаты фолда, **структурированные** параметры.

Общий контейнер ``Params`` остаётся ``dict`` для совместимости с произвольными
``ModelCls`` и :class:`utils.TFRDataset.TFRDataset`. Для двух основных методик
ниже заданы :class:`typing.TypedDict` — их можно использовать в аннотациях
фабрик и для статической проверки.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, NotRequired, Sequence, TypedDict, Union

import numpy as np
import torch.nn as nn

from lib.core.tensors import EpochLabelsArray, TFRFeatureArray

# --- Совместимость: общий тип бандла для ``run_fold_fn`` ---

Params = Dict[str, Dict[str, Any]]
"""
Ключи верхнего уровня (инвариант движка):

* ``"model"`` — kwargs для ``ModelCls(**...)``
* ``"optimizer"`` — kwargs для ``torch.optim.AdamW(model.parameters(), **...)``
* ``"tr_dataset"``, ``"vl_dataset"`` — kwargs для ``TFRDataset(X, y, **...)``
* ``"tr_loader"``, ``"vl_loader"`` — kwargs для ``DataLoader(dataset, **...)``
"""

Attrs = Dict[str, Any]
Values = Union[float, Sequence[float]]


# --- TypedDict: AlexNet + типичный TFRDataset / DataLoader ---


class TfrDatasetKwargs(TypedDict, total=False):
    """Аргументы :class:`utils.TFRDataset.TFRDataset` помимо ``X``, ``y``."""

    time_crop: int | None
    """
    Если ``int`` — случайный crop по времени длины ``time_crop`` внутри ``T``;
    ``None`` — полная матрица ``(C,F,T)`` на эпоху.
    """


class DataLoaderKwargs(TypedDict):
    batch_size: int
    shuffle: bool


class AdamWKwargs(TypedDict):
    lr: float
    weight_decay: float


class AlexNetModelKwargs(TypedDict):
    """Kwargs для :class:`lib.models.alexnet.model.AlexNetTFR`."""

    in_channels: int
    num_classes: int
    dropout: float


class AlexNetFoldParams(TypedDict):
    """Полный бандл параметров фолда для AlexNet + AdamW + TFRDataset."""

    model: AlexNetModelKwargs
    optimizer: AdamWKwargs
    tr_dataset: TfrDatasetKwargs
    vl_dataset: TfrDatasetKwargs
    tr_loader: DataLoaderKwargs
    vl_loader: DataLoaderKwargs


# --- TypedDict: Transformer wrapper ---


class TransformerModelKwargs(TypedDict):
    """
    Kwargs для :class:`lib.models.tfr_transformer.wrapper.TFRTransformerWrapper`.

    ``pooling`` / ``preprocess`` — экземпляры ``nn.Module``
    (например :class:`~lib.models.tfr_transformer.preprocess.SeqPool`).
    """

    num_classes: int
    seq_len: int
    embed_dim: int
    nhead: int
    dim_fc: int
    num_layers: int
    dropout: float
    pooling: nn.Module
    preprocess: nn.Module
    kernel_freq: NotRequired[int]
    kernel_time: NotRequired[int]


class TransformerFoldParams(TypedDict):
    model: TransformerModelKwargs
    optimizer: AdamWKwargs
    tr_dataset: TfrDatasetKwargs
    vl_dataset: TfrDatasetKwargs
    tr_loader: DataLoaderKwargs
    vl_loader: DataLoaderKwargs


# --- Кривые обучения за один фолд ---


class FoldTrainingCurves(TypedDict):
    """Списки по эпохам внутри одного train/val фолда."""

    train_losses: List[float]
    val_losses: List[float]
    val_f1s: List[float]


@dataclass(frozen=True)
class Split:
    """
    Один train/validation блок данных.

    **Тензоры NumPy**

    * ``X_*``: ожидаемая форма ``(N, C, F, T)``, dtype float — см. :data:`TFRFeatureArray`.
    * ``y_*``: форма ``(N,)``, целочисленные метки ``0..K-1`` — см. :data:`EpochLabelsArray`.

    Имена сплита: ``\"holdout\"`` или ``\"fold{i}\"`` при CV.
    """

    name: str
    X_train: TFRFeatureArray
    y_train: EpochLabelsArray
    X_val: TFRFeatureArray
    y_val: EpochLabelsArray


@dataclass
class FoldResult:
    """
    Итог одного фолда после early stopping по ``patience``.

    Attributes
    ----------
    split:
        Имя сплита (как у :class:`Split`).
    best_f1:
        Лучший macro-F1 на валидации по эпохам фолда.
    loss_metric:
        Скаляр из ``loss_metric(val_losses)`` (например наклон ``val_loss``).
    curves:
        Подробные кривые по эпохам; ключи см. :class:`FoldTrainingCurves`.
    """

    split: str
    best_f1: float
    loss_metric: float
    curves: FoldTrainingCurves
