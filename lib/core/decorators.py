"""
Декораторы для :class:`lib.core.pipeline.PipelineRunner.run`.

Добавляют в ``RunResult.artifacts`` время выполнения и печатают короткий лог.
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

from lib.core.contracts import RunResult

F = TypeVar("F", bound=Callable[..., RunResult])


def with_timing(fn: F) -> F:
    """
    После вызова записывает ``elapsed_sec`` в ``result.artifacts``.

    Notes
    -----
    Использует ``time.perf_counter()`` (монотонные часы).
    """

    def wrapper(*args, **kwargs):  # type: ignore[override]
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        result.artifacts["elapsed_sec"] = time.perf_counter() - t0
        return result

    return wrapper  # type: ignore[return-value]


def with_logging(fn: F) -> F:
    """Печатает имя функции до/после и итоговые ``metrics``."""

    def wrapper(*args, **kwargs):  # type: ignore[override]
        print(f"[framework] start: {fn.__name__}")
        result = fn(*args, **kwargs)
        print(f"[framework] done: {fn.__name__} metrics={result.metrics}")
        return result

    return wrapper  # type: ignore[return-value]
