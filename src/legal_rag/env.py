"""Environment loading helpers for Utopia runtime configuration."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_env_file(path: str | Path | None) -> Path | None:
    """Resolve only the explicitly configured environment file."""
    if path is None:
        return None
    env_path = Path(path)
    if env_path.exists():
        return env_path
    return env_path


def load_env_file(path: str | Path | None, *, override: bool = False) -> dict[str, str]:
    """Load simple KEY=VALUE pairs from a .env file without extra dependencies."""
    env_path = resolve_env_file(path)
    if env_path is None:
        return {}
    if not env_path.exists():
        return {}
    loaded: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = value
    return loaded
