# 03 - Indexing Contract Results

## Implemented Scope

The indexing step is implemented in `legal_rag.indexing`.

It validates `data/laws_dataset_clean/`, embeds `text_for_embedding`, writes deterministic Qdrant point IDs from `chunk_id`, stores the complete payload contract, creates payload indexes for filterable fields, and writes reproducible run artifacts under `data/indexing_runs/<run_id>/`.

## Implementation Choices

- Qdrant is used through the official `qdrant-client` SDK.
- The historical notebook full run targeted Qdrant local file mode at `data/indexes/qdrant`; new full runs should target Qdrant Docker/server mode at `http://127.0.0.1:6333` with storage under `data/indexes/qdrant_server`.
- The collection uses named vectors: `dense`, plus `sparse` when `hybrid_enabled=True`.
- The historical `legal_chunks` collection is preserved for baseline runs.
- The BGE-M3 re-index setup writes to the parallel `legal_chunks_bge_m3` collection with local `BAAI/bge-m3`, dense size `1024`, and native sparse vectors enabled.
- Docker/server indexing supports parallel upload, explicit shard count, and temporary HNSW indexing-threshold deferral for bulk load.
- The pipeline stores `content_hash = sha256(text_for_embedding.strip())` and uses UUIDv5 point IDs derived from `chunk_id` for idempotent reruns.

## Generated Artifacts

Each run writes:

- `index_manifest.json`
- `payload_profile.json`
- `index_quality_report.md`
- `sample_retrieval_report.json`
- `diagnostic_queries.json`
- `failures.jsonl`

The manifest records dataset hashes, embedding backend/model/dimension, Qdrant path or URL, vector names, hybrid flag, indexing counts, requested payload indexes with creation statuses, duplicate checks, filter checks, and quality gates.

## Verification

Focused tests cover dataset validation, stable hashes and point IDs, Qdrant in-memory indexing, idempotent reuse, CLI smoke behavior, Utopia dense adapter behavior, and local BGE-M3 dense/sparse adapter parsing.
Additional tests cover Docker/server tuning fields, Qdrant optimizer threshold wiring, and the `upload_points(..., parallel=...)` branch.

## BGE-M3 Re-index Setup

The monitorable setup for the Phase 1 re-index is available in `notebooks/03_indexing_contract.ipynb`, section 7.

Configuration prepared for the manual full run:

- `collection_name`: `legal_chunks_bge_m3`
- `index_dir`: `data/indexes/qdrant_server`
- `qdrant_url`: `http://127.0.0.1:6333`
- `embedding_backend`: `local`
- `embedding_model`: `BAAI/bge-m3`
- `embedding_dim`: `1024`
- `hybrid_enabled`: `True`
- `force_rebuild`: `True`, scoped only to `legal_chunks_bge_m3`
- `qdrant_upload_parallel`: `4`
- `qdrant_shard_number`: `4`
- `qdrant_bulk_indexing_threshold_kb`: `10000000`

The notebook writes progress events to `data/indexing_runs/<run_id>_progress.jsonl` and, after completion, reports the `legal_chunks` state, verifies that `legal_chunks_bge_m3` has `dense` and `sparse` vectors, checks that the manifest is ready for retrieval, and runs dense/hybrid smoke retrieval.

## BGE-M3 Full Re-index Run

Run artifact:

- Manifest: `data/indexing_runs/20260512_212818/index_manifest.json`
- Progress log: `data/indexing_runs/bge_m3_full_20260512_212742_progress.jsonl`
- Collection: `legal_chunks_bge_m3`

Validation summary:

- `ready_for_retrieval`: `true`
- `selected_count`: `76467`
- `indexed_count`: `76467`
- `collection_points_count`: `76467`
- `failure_count`: `0`
- embedding backend/model: `local` / `BAAI/bge-m3`
- dense vector size: `1024`
- sparse vector: `sparse`
- quality gates: all `true`

Dense and hybrid smoke retrieval both returned non-empty results for `contributi regionali`. Qdrant local mode emits a performance warning for collections above 20,000 points; this is acceptable for the thesis PoC but should be considered when running long sweeps.
