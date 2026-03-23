from __future__ import annotations

from typing import Any, Mapping

from lib.core.contracts import ModeRunner, Model, RunResult


class OfflineEpochMode(ModeRunner):
    """
    Offline batch: один вызов ``model.fit(X, y)`` и возврат заглушки метрик.

    ``X``, ``y`` здесь намеренно ``Any``: скелет не навязывает форму TFR;
    для явных форм см. :mod:`lib.core.tensors`.
    """

    def run(self, model: Model, X: Any, y: Any, *, context: Mapping[str, Any]) -> RunResult:
        model.fit(X, y)
        return RunResult(
            name="offline_epoch",
            metrics={"status": 1.0},
            artifacts={"note": "skeleton mode executed", "context_keys": list(context.keys())},
        )

