from __future__ import annotations

import sys
from pathlib import Path

from notebooks.pipelines.common.bootstrap import bootstrap_notebook, find_project_root


def test_find_project_root_resolves_repo_root() -> None:
    root = find_project_root(Path.cwd())
    assert (root / "pyproject.toml").exists()
    assert (root / "src" / "legal_indexing").exists()


def test_bootstrap_notebook_adds_paths_without_loading_env() -> None:
    root, src = bootstrap_notebook(start=Path.cwd(), load_env=False)
    assert root == find_project_root(Path.cwd())
    assert src == root / "src"
    assert str(root) in sys.path
    assert str(src) in sys.path
