from __future__ import annotations

from typing import Any, Mapping

from lib.core.contracts import ModeRunner, Model, RunResult


class OnlineSlidingWindowMode(ModeRunner):
    """Online-like mode placeholder for sliding window decoding."""

    def __init__(self, window_size: int = 128, step: int = 32):
        self.window_size = window_size
        self.step = step

    def run(self, model: Model, X: Any, y: Any, *, context: Mapping[str, Any]) -> RunResult:
        # Skeleton only: real online inference logic comes in next iterations.
        _ = (model, X, y, context)
        return RunResult(
            name="online_sliding_window",
            metrics={"status": 1.0},
            artifacts={"window_size": self.window_size, "step": self.step},
        )

