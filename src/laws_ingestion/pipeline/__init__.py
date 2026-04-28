"""Backward-compatible pipeline namespace.

Prefer: laws_ingestion.data_preparation.laws_graph
"""

from laws_ingestion.data_preparation.laws_graph import PipelineConfig, SCHEMA_VERSION, run_pipeline

__all__ = ["PipelineConfig", "SCHEMA_VERSION", "run_pipeline"]
