"""Deterministic HTML -> JSONL ingestion for a legal RAG dataset."""

from .core import build_corpus_registry, ingest_law
from .data_preparation.laws_graph import PipelineConfig, SCHEMA_VERSION, run_pipeline

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "PipelineConfig",
    "SCHEMA_VERSION",
    "build_corpus_registry",
    "ingest_law",
    "run_pipeline",
]
