from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd

__all__ = ["save_signal_counts_chart", "save_status_distribution_chart"]

def save_signal_counts_chart(signal_counts: Mapping[str, int], run_dir: Path) -> Path:
    series = pd.Series(dict(signal_counts)).drop(labels=["n_files"], errors="ignore").sort_values(ascending=False)
    fig = plt.figure(figsize=(12, 4))
    series.plot(kind="bar")
    plt.title("Segnali rilevati nel corpus")
    plt.tight_layout()
    fig_path = run_dir / "step01_signal_counts.png"
    plt.savefig(fig_path)
    plt.show()
    return fig_path


def save_status_distribution_chart(status_counts: pd.Series, run_dir: Path) -> Path:
    fig = plt.figure(figsize=(6, 4))
    status_counts.plot(kind="bar")
    plt.title("Distribuzione status leggi")
    plt.tight_layout()
    fig_path = run_dir / "step03_status_distribution.png"
    plt.savefig(fig_path)
    plt.show()
    return fig_path
