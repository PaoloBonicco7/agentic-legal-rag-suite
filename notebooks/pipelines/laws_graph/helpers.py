from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = ["has_current", "load_parquet_or_json_fallback"]


def load_parquet_or_json_fallback(path: Path) -> pd.DataFrame:
    """Read a parquet artifact or fallback JSONL content with the same filename."""
    try:
        return pd.read_parquet(path)
    except Exception:
        rows: list[dict[str, Any]] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return pd.DataFrame(rows)


def has_current(value: object) -> bool:
    if isinstance(value, list):
        return "current" in value
    if isinstance(value, str):
        return "current" in value
    return False
