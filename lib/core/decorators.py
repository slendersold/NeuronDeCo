from __future__ import annotations

import time
from typing import Callable, TypeVar

from .contracts import RunResult

F = TypeVar("F", bound=Callable[..., RunResult])


def with_timing(fn: F) -> F:
    """Attach elapsed seconds to result artifacts."""

    def wrapper(*args, **kwargs):  # type: ignore[override]
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        result.artifacts["elapsed_sec"] = time.perf_counter() - t0
        return result

    return wrapper  # type: ignore[return-value]


def with_logging(fn: F) -> F:
    """Tiny wrapper-level logging for mode execution."""

    def wrapper(*args, **kwargs):  # type: ignore[override]
        print(f"[framework] start: {fn.__name__}")
        result = fn(*args, **kwargs)
        print(f"[framework] done: {fn.__name__} metrics={result.metrics}")
        return result

    return wrapper  # type: ignore[return-value]

