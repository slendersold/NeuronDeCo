from __future__ import annotations

from typing import Dict, Type

from lib.modes.offline import OfflineEpochMode
from lib.modes.online import OnlineSlidingWindowMode
from lib.models.alexnet_model import AlexNetModelAdapter

MODE_REGISTRY: Dict[str, Type] = {
    "offline_epoch": OfflineEpochMode,
    "online_sliding_window": OnlineSlidingWindowMode,
}

MODEL_REGISTRY: Dict[str, Type] = {
    "alexnet": AlexNetModelAdapter,
}

