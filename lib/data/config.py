"""Runtime config loader with lazy auto-discovery and backward compatibility."""
from __future__ import annotations

import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


_REQUIRED_KEYS: tuple[str, ...] = ("ch_to_keep", "best_ch_by_power", "epoch_thresh_dict")
_VALUES: dict[str, Any] = {}
_CONFIG_PATH: Path | None = None


def _load_config_from_path(path: Path) -> ModuleType:
    spec = spec_from_file_location("neurondeco_external_config", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {path}")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve_candidates() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    cwd = Path.cwd().resolve()

    candidates: list[Path] = []
    env_path = os.environ.get("NEURONDECO_CONFIG_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser().resolve())

    candidates.extend(
        [
            repo_root / "PreprocessedData" / "config.py",
            repo_root.parent / "PreprocessedData" / "config.py",
            cwd / "PreprocessedData" / "config.py",
            cwd.parent / "PreprocessedData" / "config.py",
            cwd / "config.py",
        ]
    )
    # Keep order, remove duplicates.
    deduped: list[Path] = []
    seen: set[Path] = set()
    for p in candidates:
        if p not in seen:
            deduped.append(p)
            seen.add(p)
    return deduped


def _extract_values(cfg: ModuleType) -> dict[str, Any]:
    extracted: dict[str, Any] = {}

    # 1) Backward-compatible format: top-level module attributes.
    for key in _REQUIRED_KEYS:
        if hasattr(cfg, key):
            extracted[key] = getattr(cfg, key)

    # 2) New format: config values stored under a mapping key.
    mapping_candidates = ("CONFIG", "config", "VALUES", "settings", "DATA_CONFIG")
    for mapping_name in mapping_candidates:
        mapping_obj = getattr(cfg, mapping_name, None)
        if isinstance(mapping_obj, Mapping):
            for key in _REQUIRED_KEYS:
                if key in mapping_obj and key not in extracted:
                    extracted[key] = mapping_obj[key]

    missing = [k for k in _REQUIRED_KEYS if k not in extracted]
    if missing:
        available_module_keys = sorted(k for k in vars(cfg).keys() if not k.startswith("__"))
        raise KeyError(
            "Missing required config keys: "
            f"{missing}. Expected top-level attrs or one of mapping keys "
            f"{mapping_candidates}. Available module keys: {available_module_keys}"
        )
    return extracted


def initialize_config(config_path: str | Path | None = None) -> None:
    """Initialize config values from an explicit path or auto-discovered candidates."""
    global _CONFIG_PATH

    if config_path is None:
        candidates = _resolve_candidates()
    else:
        candidates = [Path(config_path).expanduser().resolve()]

    last_error: Exception | None = None
    for path in candidates:
        try:
            if not path.exists():
                continue
            if path.name != "config.py":
                continue
            cfg = _load_config_from_path(path)
            values = _extract_values(cfg)
            _VALUES.clear()
            _VALUES.update(values)
            _CONFIG_PATH = path
            return
        except Exception as exc:  # try next candidate
            last_error = exc

    if config_path is None:
        searched = ", ".join(str(p) for p in candidates)
        msg = (
            "Failed to auto-initialize lib.data.config. "
            f"Tried: {searched}. "
            "Call initialize_config('/abs/path/to/config.py') explicitly."
        )
        if last_error is not None:
            msg += f" Last error: {last_error!r}"
        raise RuntimeError(msg)

    explicit = str(candidates[0])
    if last_error is None:
        raise FileNotFoundError(f"config.py not found or invalid path: {explicit}")
    raise RuntimeError(f"Failed to initialize config from {explicit}: {last_error!r}")


def _ensure_initialized() -> None:
    if _CONFIG_PATH is not None and all(k in _VALUES for k in _REQUIRED_KEYS):
        return
    initialize_config(None)


def get_config_path() -> Path:
    _ensure_initialized()
    assert _CONFIG_PATH is not None
    return _CONFIG_PATH


def __getattr__(name: str) -> Any:
    if name in _REQUIRED_KEYS:
        _ensure_initialized()
        return _VALUES[name]
    raise AttributeError(name)


__all__ = ["initialize_config", "get_config_path", *_REQUIRED_KEYS]
