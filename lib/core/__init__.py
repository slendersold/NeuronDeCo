"""
Ядро библиотеки: контракты режимов/моделей и **именованные** типы для TFR.

* :mod:`lib.core.tensors` — соглашения по осям ``N,C,F,T,K`` и алиасы NumPy.
* :mod:`lib.core.contracts` — ``RunResult``, ``TorchTFRClassifier``, ``ModeRunner``, …
"""

from lib.core.contracts import (
    Model,
    ModeRunner,
    ObjectiveEngine,
    RunMetadata,
    RunResult,
    SklearnLikeModel,
    TorchTFRClassifier,
)
from lib.core.context import RunContext
from lib.core.tensors import (
    EpochLabelsArray,
    Float32Array,
    FloatArray,
    Int64Array,
    TFRFeatureArray,
)

__all__ = [
    "EpochLabelsArray",
    "Float32Array",
    "FloatArray",
    "Int64Array",
    "TFRFeatureArray",
    "Model",
    "ModeRunner",
    "ObjectiveEngine",
    "RunContext",
    "RunMetadata",
    "RunResult",
    "SklearnLikeModel",
    "TorchTFRClassifier",
]
