# 03 - Indexing Contract Specification

## Purpose

Define how the clean legal dataset becomes a retrieval-ready index for RAG.

This step proves that generated chunks can be indexed with stable payload metadata, reproducible hashes, dense and sparse vectors for hybrid retrieval, and filterable fields needed by simple and advanced retrieval. The index must be reproducible from the exact clean dataset hash recorded at indexing time.

The vector store, embedding backend, and retrieval strategy are fixed at the project level in `AGENTS.md`. This spec only details how those choices are wired into a reproducible indexing artifact.

## Inputs

- Clean legal dataset: `data/laws_dataset_clean/`.
- Required files from step 01: `manifest.json`, `chunks.jsonl`, `laws.jsonl`, `articles.jsonl`, `edges.jsonl`.
- Indexing configuration (`IndexingConfig` Pydantic model):
  - `clean_dataset_dir: str` (default `data/laws_dataset_clean`).
  - `index_dir: str` (default `data/indexes/qdrant`).
  - `runs_dir: str` (default `data/indexing_runs`).
  - `collection_name: str` (default `legal_chunks`).
  - `qdrant_url: str | None` (default `None`; the thesis workflow uses local persistent mode).
  - `qdrant_api_key: str` (default empty; retained only for compatibility with explicit remote experiments).
  - `reuse_vectors_index_dir: str | None` (optional local source index for verified vector reuse).
  - `reuse_vectors_collection: str | None` (source collection paired with the reuse index).
  - `reuse_vectors_manifest_path: str | None` (source manifest used to verify embedding and collection identity).
  - `embedding_backend: Literal["local","utopia"]` (default `local`).
  - `embedding_model: str` (default `BAAI/bge-m3`).
  - `embedding_dim: int | None` (optional override; otherwise derived from the model).
  - `hybrid_enabled: bool` (default `True`).
  - `chunk_selection_mode: Literal["full","sample"]` (default `full`; `sample` is for notebook smoke runs).
  - `sample_size: int | None` (used only when `chunk_selection_mode == "sample"`).
  - `force_rebuild: bool` (default `False`).
  - `batch_size: int` (default `64`).
  - `upload_batch_size: int` (default `64`).
  - `qdrant_upload_parallel: int` (default `1`).
  - `env_file: str | None` (default `.env`, used to load `UTOPIA_*` credentials when `embedding_backend == "utopia"`).

## Embedding and Vector Store

- **Vector store**: Qdrant local persistent file mode at `index_dir`. Definitive experiments use isolated local paths and collection names. Remote mode remains an explicit compatibility path and is not used by the validity-filter audit.
- **Parallel collections**: experiments that change embedding model or vector topology must use a new `collection_name` instead of overwriting a historical collection. The BGE-M3 hybrid retrieval index uses `legal_chunks_bge_m3` while the previous `legal_chunks` collection remains available for baseline runs.
- **Distance metric**: `Cosine` for dense vectors.
- **Dense vector**: produced by the configured embedding backend.
  - Backend `local`: model loaded via `sentence-transformers` or `FlagEmbedding`. Default `BAAI/bge-m3`, dim `1024`.
  - Backend `utopia`: HTTP call to `<UTOPIA_BASE_URL>/ollama/api/embeddings` with `model=embedding_model`. Dimensionality read from the first response and recorded in the manifest; if `embedding_dim` is set in config, it must match the response or indexing fails.
- **Sparse vector**: only stored when `hybrid_enabled=True`.
  - For `BAAI/bge-m3` (local): use the model's native sparse output.
  - For other local models without native sparse output: use `qdrant-client[fastembed]` BM25 sparse encoder client-side.
  - For the current `utopia` backend implementation: only dense output is supported, so
    `hybrid_enabled` must be disabled. Hybrid runs use the local BGE-M3 backend.
- **Embedding input text**: every chunk is embedded using its `text_for_embedding` field from step 01 (not the raw `text`), to preserve legal context like article label and structure path.
- **Payload indexes**: at index creation, keyword indexes are created before upload for `law_id`, `law_status`, `article_id`, `article_status`, `passage_status`, `content_availability`, `index_views`, and `relation_types`.

## Idempotency and Rebuild

- Each chunk is stored under a stable point id derived from `chunk_id` (a UUID5 of `chunk_id` keeps Qdrant id constraints satisfied without losing the human-readable identifier in the payload).
- On re-run with `force_rebuild=False`:
  - if `content_hash` and `payload_hash` both match, the point is skipped;
  - if only `payload_hash` differs, payload metadata is updated without recomputing vectors;
  - if `content_hash` differs, the point is upserted with new vectors and payload;
  - if no point with that id exists, it is inserted.
- On re-run with `force_rebuild=True`, the collection is dropped and recreated from scratch.
  This operation is scoped to the configured `collection_name`; other collections in the same Qdrant local path or server storage are not removed.
- The manifest records counts for `inserted`, `vector_updated`, `payload_updated`, `skipped`, and `removed` (the last only when rebuild is requested).

An optional full rebuild may reuse vectors from an isolated local source collection. All three
`reuse_vectors_*` fields must be configured together, the target must be a different local path,
and `force_rebuild=True` is required. Before reading source vectors, the pipeline verifies the
source manifest, collection topology, embedding model, vector size, hybrid mode, chunks hash, and
ready state. A vector is reused only when the source point has the same `chunk_id` and
`content_hash`; otherwise the chunk is embedded normally. The manifest records the source
provenance and the separate `vector_reused_count` and `embedded_count`.

## Outputs

Default generated output locations:

- retrieval index under `data/indexes/qdrant/`;
- indexing artifacts under `data/indexing_runs/<timestamp>/`.

Required artifacts:

- `index_manifest.json`: actual source and chunk hashes, clean-manifest hash, pipeline code identity, embedding identity, collection identity, counts, payload indexes, schema/status-rule versions, and run configuration.
- `payload_profile.json`: coverage and bounded value distributions for all filterable metadata fields and single-valued collection identity fields.
- `index_quality_report.md`: human-readable validation report.
- optional `diagnostic_queries.json`: results for a small fixed set of diagnostic queries (a few question-shaped probes), used to sanity-check retrieval before running step 05.

## Pipeline

1. Validate the clean dataset contract and recompute file hashes.
   Why: indexing must fail before embedding when files differ from the hashes recorded by preprocessing.
2. Resolve embedding backend and load the embedder.
   Why: failing here surfaces credential or model issues before any chunk is processed.
3. Create or open the Qdrant collection with the correct dense and (if enabled) sparse vector schema and payload indexes.
   Why: collection topology must match the chosen backend; mismatches must be caught up front.
4. Select chunks for indexing (`full` or `sample`).
   Why: the PoC supports both full runs and small notebook runs while preserving the same contract.
5. Build embedding input from `text_for_embedding`, reuse only verified unchanged vectors when
   configured, and produce the remaining dense (and optionally sparse) vectors in batches.
   Why: batching keeps memory and HTTP behavior predictable while reuse avoids recomputing
   representations already produced for byte-identical input with the same model configuration.
6. Upsert points with stable ids, applying the idempotency policy.
   Why: repeated runs should not duplicate unchanged chunks.
7. Validate the index.
   Why: later RAG steps need confidence that filters, hybrid retrieval, and payload fields work.
8. Export manifest, payload profile, and quality report.
   Why: every run must be traceable back to its inputs and configuration.

## Contract

The Qdrant collection schema must include:

- one named **dense vector** (e.g., `dense`) with size matching the embedding model and `Cosine` distance;
- one **sparse vector** (e.g., `sparse`) when `hybrid_enabled=True`;
- payload indexes on `law_id`, `law_status`, `article_id`, `article_status`, `passage_status`, `content_availability`, `index_views`, and `relation_types`.

The index payload for every stored chunk must include:

- `chunk_id`
- `passage_id`
- `article_id`
- `law_id`
- `text`
- `law_date`
- `law_number`
- `law_title`
- `law_status`
- `article_status`
- `passage_status`
- `content_availability`
- `status_event_ids`
- `status_rule_ids`
- `article_label_norm`
- `passage_label`
- `structure_path`
- `source_file`
- `index_views`
- `related_law_ids`
- `inbound_law_ids`
- `outbound_law_ids`
- `relation_types`
- `content_hash`
- `payload_hash`
- `dataset_source_hash`
- `dataset_manifest_hash`
- `dataset_chunks_hash`
- `preprocessing_schema_version`
- `status_rules_version`

Filterable fields must include at least `law_id`, `law_status`, `article_id`, `article_status`, `passage_status`, `content_availability`, `index_views`, and `relation_types`.

`INDEXING_SCHEMA_VERSION` is `indexing-contract-v2`.

The index manifest must record actual SHA-256 values for the clean manifest and chunks file, the preprocessing schema/status-rules versions, source and pipeline identities, embedding identity, collection identity, and indexing counts.

## Quality Gates

- Source clean dataset has `ready_for_indexing=True`.
- Actual required-file hashes match the preprocessing manifest.
- Embedding backend and model are recorded in the manifest, together with the resolved dimensionality.
- When vector reuse is configured, source manifest identity and topology are compatible, every
  reused point has a matching content hash, and reused plus embedded counts equal selected count.
- The Qdrant collection topology matches the configured dense (and sparse, when enabled) schema.
- Payload indexes exist for every required filterable field.
- Every indexed point carries the full payload contract.
- Indexed count equals selected count after applying the idempotency policy.
- Collection identity fields have exactly one value and match the index manifest.
- Status and index-view distributions reconcile with the selected chunks.
- Duplicate `chunk_id` values are rejected.
- Filterable fields are queryable: a smoke filter on `law_status` returns the expected count.
- Diagnostic queries return non-empty results for at least one well-known law.
- Index manifest links back to the exact clean dataset and pipeline identities from step 01.

## Notebook Role

`notebooks/03_indexing_contract.ipynb` should:

- validate the clean dataset contract;
- run a small indexing job in `sample` mode to demonstrate the pipeline quickly;
- show the Qdrant collection schema, a representative payload, and the payload profile;
- run a few diagnostic queries (one filter-only, one dense-only, one hybrid) to confirm that retrieval and filters work;
- provide a guarded, monitorable full BGE-M3 re-index section against a dedicated local persistent path, with progress output and a JSONL progress log under `data/indexing_runs/`;
- run the full indexing pipeline on the entire clean dataset and produce the final artifacts when the guarded full-run cell is explicitly enabled.

The notebook should explain the payload fields because they are the bridge between preprocessing (step 01) and RAG behavior (steps 05 and 06), and should explicitly note which retrieval modes the produced collection supports.

## Acceptance Criteria

- The clean dataset can be indexed reproducibly with the local BGE-M3 backend for dense+hybrid
  retrieval and with the current Utopia backend for dense-only retrieval.
- Full BGE-M3 indexing runs in an isolated Qdrant local persistent path and records `qdrant.mode="local"` in the manifest.
- The collection topology and payload contract are sufficient for simple RAG (step 05) and advanced graph RAG (step 06) without further indexing changes.
- The index can be traced back to the exact clean dataset hash and embedding model identity recorded in the manifest.
- Re-running indexing without changes is a no-op; metadata-only changes update payload without recomputing vectors, while content changes update vectors and payload.
- The notebook demonstrates that retrieval and filters work before any answer generation is evaluated downstream.
