# Changelog - Laws Graph Pipeline

## 2026-02-22
- Added deterministic conservative pipeline modules under `laws_ingestion/pipeline/`:
  - `scan.py`, `status.py`, `relations.py`, `events.py`, `views.py`, `reporting.py`, `workflow.py`.
- Added full notebook `notebooks/pre_processing_data/03_laws_graph_pipeline.ipynb` (Italian) with step 0-8 flow.
- Implemented clean export target `data/laws_dataset_clean/` with additive fields and new `events.jsonl`.
- Added run QA artifacts under `notebooks/data/laws_graph_pipeline/<run_id>/` including reports, figures, and tables.
- Hardened parser/reference logic:
  - improved `stdlib` HTML parser behavior in table contexts,
  - reduced duplicate article starts from TOC/`INDICE`,
  - filtered ingest self-loop edges,
  - expanded reference regex support for typo variants (`L:R`, `L.R..`).
- Updated default CLI HTML path from `data/leggi-html` to `data/laws_html`.
- Updated docs and archive notes for legacy notebooks (`notebooks/old/`).
- Added tests for status, relations normalization, events extraction, and mandatory ingestion edge cases.
