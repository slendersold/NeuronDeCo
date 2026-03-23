from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lib.optuna_objective_makers import make_objective_engine


@dataclass
class OptunaObjectiveEngine:
    """
    Thin wrapper around existing make_objective_engine from lib.
    Keeps current behavior while exposing a stable skeleton entrypoint.
    """

    def build(self, X: Any, y: Any, **kwargs: Any) -> Any:
        return make_objective_engine(X=X, y=y, **kwargs)

