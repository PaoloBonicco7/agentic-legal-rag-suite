# 03 - Indexing Contract Results

## Implemented Scope

The indexing step is implemented in `legal_rag.indexing`.

It validates `data/laws_dataset_clean/`, embeds `text_for_embedding`, writes deterministic Qdrant point IDs from `chunk_id`, stores the complete payload contract, creates payload indexes for filterable fields, and writes reproducible run artifacts under `data/indexing_runs/<run_id>/`.

## Implementation Choices

- Qdrant is used through the official `qdrant-client` SDK.
- The notebook full run targets Qdrant server mode at `http://127.0.0.1:6333`, backed by `docker-compose.qdrant.yml`.
- The collection uses named vectors: `dense`, plus `sparse` when `hybrid_enabled=True`.
- The notebook full run uses Utopia embeddings, resolving `UTOPIA_EMBED_MODEL` or falling back to `SLURM.nomic-embed-text:latest`.
- Local `BAAI/bge-m3` remains available for hybrid dense+sparse runs, but Utopia is used dense-only with `hybrid_enabled=False`.
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
