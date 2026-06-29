"""
Общие протоколы для моделей.

Основной контракт для TFR + PyTorch см. :class:`lib.core.contracts.TorchTFRClassifier`.
"""

from __future__ import annotations

from lib.core.contracts import TorchTFRClassifier

# Историческое имя в репозитории — алиас на ядро.
TFRClassifier = TorchTFRClassifier

__all__ = ["TFRClassifier", "TorchTFRClassifier"]
