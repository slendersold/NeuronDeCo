"""
Контекст одного «прогона» в смысле скелета ``PipelineRunner`` (не путать с Optuna trial).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from lib.core.contracts import RunMetadata


@dataclass
class RunContext:
    """
    Параметры окружения для режимов ``lib.modes``.

    Attributes
    ----------
    run_id:
        Строковый идентификатор эксперимента / прогона.
    device:
        Строка для ``torch.device`` (``\"cpu\"``, ``\"cuda\"``, …).
    seed:
        Базовое значение для RNG, если режим его использует.
    output_dir:
        Каталог артефактов по умолчанию.
    metadata:
        Произвольный словарь, пробрасывается в ``ModeRunner.run(..., context=...)``.
        Тип — :data:`~lib.core.contracts.RunMetadata`.
    """

    run_id: str = "dev-run"
    device: str = "cpu"
    seed: int = 42
    output_dir: Path = Path("./outputs")
    metadata: Dict[str, Any] = field(default_factory=dict)
