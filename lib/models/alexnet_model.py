from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from lib.AlexNet.AlexNet import AlexNetTFR


@dataclass
class AlexNetModelAdapter:
    """
    Minimal adapter that keeps constructor metadata.
    Training/inference details remain in mode/objective layers.
    """

    in_channels: int = 7
    num_classes: int = 2
    dropout: float = 0.5
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    def fit(self, X: Any, y: Any) -> "AlexNetModelAdapter":
        # Placeholder in skeleton: actual training is handled by objective/mode.
        return self

    def predict(self, X: Any) -> Any:
        raise NotImplementedError("Prediction is mode-specific in this skeleton.")

    def build_torch_module(self) -> AlexNetTFR:
        return AlexNetTFR(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            dropout=self.dropout,
            **self.extra_kwargs,
        )

