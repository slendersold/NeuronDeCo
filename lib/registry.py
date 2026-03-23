from __future__ import annotations

from typing import Dict, Type

from lib.modes.offline import OfflineEpochMode
from lib.modes.online import OnlineSlidingWindowMode
from lib.models.alexnet import AlexNetTFR
from lib.models.tfr_transformer import TFRTransformerWrapper

MODE_REGISTRY: Dict[str, Type] = {
    "offline_epoch": OfflineEpochMode,
    "online_sliding_window": OnlineSlidingWindowMode,
}

MODEL_REGISTRY: Dict[str, Type] = {
    "alexnet": AlexNetTFR,
    "tfr_transformer": TFRTransformerWrapper,
}
