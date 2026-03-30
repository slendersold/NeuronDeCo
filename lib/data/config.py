"""Explicit runtime initialization for external config.py."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_config_from_path(path: Path) -> ModuleType:
    spec = spec_from_file_location("neurondeco_external_config", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {path}")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_VALUES: dict[str, Any] = {}
_CONFIG_PATH: Path | None = None


def initialize_config(config_path: str | Path) -> None:
    """Initialize config globals from an explicit path to config.py."""
    global _CONFIG_PATH

    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"config.py not found: {path}")
    if path.name != "config.py":
        raise ValueError(f"Expected path to config.py, got: {path}")

    cfg = _load_config_from_path(path)
    _VALUES["ch_to_keep"] = getattr(cfg, "ch_to_keep")
    _VALUES["best_ch_by_power"] = getattr(cfg, "best_ch_by_power")
    _VALUES["epoch_thresh_dict"] = getattr(cfg, "epoch_thresh_dict")
    _CONFIG_PATH = path


def get_config_path() -> Path:
    if _CONFIG_PATH is None:
        raise RuntimeError(
            "lib.data.config is not initialized. "
            "Call initialize_config('/abs/path/to/config.py') first."
        )
    return _CONFIG_PATH


def __getattr__(name: str) -> Any:
    if name in ("ch_to_keep", "best_ch_by_power", "epoch_thresh_dict"):
        if _CONFIG_PATH is None:
            raise RuntimeError(
                "lib.data.config is not initialized. "
                "Call initialize_config('/abs/path/to/config.py') first."
            )
        return _VALUES[name]
    raise AttributeError(name)


__all__ = [
    "initialize_config",
    "get_config_path",
    "ch_to_keep",
    "best_ch_by_power",
    "epoch_thresh_dict",
]
