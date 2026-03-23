from __future__ import annotations

from typing import Any

from .context import RunContext
from .contracts import ModeRunner, Model, RunResult
from .decorators import with_logging, with_timing


class PipelineRunner:
    """Core runner: mode strategy + model implementation."""

    def __init__(self, mode: ModeRunner, model: Model, context: RunContext):
        self.mode = mode
        self.model = model
        self.context = context

    @with_logging
    @with_timing
    def run(self, X: Any, y: Any) -> RunResult:
        return self.mode.run(self.model, X, y, context=self.context.metadata)

