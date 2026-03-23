from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class RunContext:
    run_id: str = "dev-run"
    device: str = "cpu"
    seed: int = 42
    output_dir: Path = Path("./outputs")
    metadata: Dict[str, Any] = field(default_factory=dict)

