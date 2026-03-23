from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Protocol, runtime_checkable


@dataclass
class RunResult:
    name: str
    metrics: Dict[str, float]
    artifacts: Dict[str, Any]


@runtime_checkable
class Model(Protocol):
    """Minimal model contract used by modes."""

    def fit(self, X: Any, y: Any) -> "Model":
        ...

    def predict(self, X: Any) -> Any:
        ...


@runtime_checkable
class ModeRunner(Protocol):
    """A mode defines how a model is applied."""

    def run(self, model: Model, X: Any, y: Any, *, context: Mapping[str, Any]) -> RunResult:
        ...


@runtime_checkable
class ObjectiveEngine(Protocol):
    """Hyperparameter objective factory."""

    def build(self, X: Any, y: Any, **kwargs: Any) -> Any:
        ...

