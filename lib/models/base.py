from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Type


@dataclass
class ModelSpec:
    name: str
    cls: Type[Any]
    default_kwargs: Dict[str, Any]

