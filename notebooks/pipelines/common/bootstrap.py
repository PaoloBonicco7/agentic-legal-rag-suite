from __future__ import annotations

from pathlib import Path
import sys

from dotenv import load_dotenv


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if (candidate / "src" / "legal_indexing").exists() and (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(
        "Project root non trovato. Avvia il notebook dentro il repository o imposta manualmente ROOT."
    )


def bootstrap_notebook(
    *,
    start: Path | None = None,
    include_root: bool = True,
    include_src: bool = True,
    load_env: bool = True,
    env_override: bool = False,
) -> tuple[Path, Path]:
    root = find_project_root(start)
    src = root / "src"

    if include_root and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if include_src and str(src) not in sys.path:
        sys.path.insert(0, str(src))

    if load_env:
        load_dotenv(root / ".env", override=env_override)

    return root, src
