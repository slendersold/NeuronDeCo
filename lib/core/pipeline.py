"""
Связка «режим + модель»: единая точка вызова ``run(X, y)``.

**Типичные формы для TFR-классификации** (если режим работает с ndarray):

* ``X``: ``float``, форма ``(N, C, F, T)`` — см. :mod:`lib.core.tensors`.
* ``y``: целые метки, форма ``(N,)``.

Конкретная интерпретация зависит от реализации :class:`~lib.core.contracts.ModeRunner`.
"""

from __future__ import annotations

from typing import Any

from lib.core.context import RunContext
from lib.core.contracts import ModeRunner, Model, RunResult
from lib.core.decorators import with_logging, with_timing


class PipelineRunner:
    """
    Выполняет ``mode.run(model, X, y, context=context.metadata)``.

    Parameters
    ----------
    mode:
        Стратегия (offline / online и т.д.).
    model:
        Зависит от ``mode``: для :class:`lib.modes.offline.OfflineEpochMode` —
        :class:`~lib.core.contracts.SklearnLikeModel`; для
        :class:`lib.modes.offline_tfr_supervised.OfflineTFRSupervisedMode` —
        ``torch.nn.Module`` (AlexNet / Transformer).
    context:
        :class:`RunContext` с ``metadata`` для режима.
    """

    def __init__(self, mode: ModeRunner, model: Any, context: RunContext) -> None:
        self.mode = mode
        self.model = model
        self.context = context

    @with_logging
    @with_timing
    def run(self, X: Any, y: Any) -> RunResult:
        # Поля RunContext всегда доступны режимам (поверх пользовательского metadata).
        ctx = dict(self.context.metadata)
        ctx.setdefault("device", self.context.device)
        ctx.setdefault("seed", self.context.seed)
        return self.mode.run(self.model, X, y, context=ctx)
