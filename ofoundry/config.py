"""
Global server / model configuration (dataclass-based, no external file required for the prototype).

Persistent user config is stored in ``~/.ofoundry/config.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Persistent user configuration  (~/.ofoundry/config.json)
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path.home() / ".ofoundry"
_CONFIG_FILE = _CONFIG_DIR / "config.json"

# Default model root when the user hasn't run ``ofoundry init``
_DEFAULT_MODELS_DIR = Path("./models")


def _read_config() -> dict:
    """Read the persistent config file; return {} if missing or invalid."""
    if _CONFIG_FILE.exists():
        try:
            return json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _write_config(cfg: dict) -> None:
    """Atomically write *cfg* to the persistent config file."""
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _CONFIG_FILE.write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def get_models_dir() -> Path:
    """Return the configured model root directory (always absolute).

    Resolution order:
    1. ``models_dir`` key in ``~/.ofoundry/config.json``
    2. ``./models`` (relative to cwd)
    """
    cfg = _read_config()
    raw = cfg.get("models_dir")
    if raw:
        return Path(raw).expanduser().resolve()
    return _DEFAULT_MODELS_DIR.resolve()


def set_models_dir(path: str | Path) -> Path:
    """Persist *path* as the model root and return the resolved value."""
    resolved = Path(path).expanduser().resolve()
    cfg = _read_config()
    cfg["models_dir"] = str(resolved)
    _write_config(cfg)
    return resolved


def get_ort_models_dir() -> Path:
    """Return ``<models_dir>/ort`` — the sub-folder for ORT-converted models."""
    return get_models_dir() / "ort"


def get_ort_cache_dir() -> Path:
    """Return ``<models_dir>/cache_dir`` — HF download cache for ORT conversion."""
    return get_models_dir() / "cache_dir"


def get_foundrylocal_models_dir() -> Path:
    """Return ``<models_dir>/foundrylocal`` — the sub-folder for Foundry Local models."""
    return get_models_dir() / "foundrylocal"


@dataclass
class ServerConfig:
    """Top-level configuration for ofoundry serve."""

    model_path: Path = field(default_factory=Path)
    host: str = "127.0.0.1"
    port: int = 18000
    execution_provider: str = "cpu"

    # Generation defaults (overridable per-request)
    default_max_new_tokens: int = 1024
    default_temperature: float = 1.0
    default_top_p: float = 0.95
    default_top_k: int = 50
    default_system_prompt: str = "You are a helpful AI assistant."

    def validate(self) -> None:
        if not self.model_path or not Path(self.model_path).exists():
            raise ValueError(f"model_path does not exist: {self.model_path}")
        if self.execution_provider not in {"cpu", "cuda", "dml"}:
            raise ValueError(f"Unsupported execution_provider: {self.execution_provider}")
