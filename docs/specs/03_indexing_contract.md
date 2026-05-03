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
  - `embedding_backend: Literal["local","utopia"]` (default `local`).
  - `embedding_model: str` (default `BAAI/bge-m3`).
  - `embedding_dim: int | None` (optional override; otherwise derived from the model).
  - `hybrid_enabled: bool` (default `True`).
  - `chunk_selection_mode: Literal["full","sample"]` (default `full`; `sample` is for notebook smoke runs).
  - `sample_size: int | None` (used only when `chunk_selection_mode == "sample"`).
  - `force_rebuild: bool` (default `False`).
  - `batch_size: int` (default `64`).
  - `env_file: str | None` (default `.env`, used to load `UTOPIA_*` credentials when `embedding_backend == "utopia"`).

## Embedding and Vector Store

- **Vector store**: Qdrant in local persistent file mode at `index_dir`. The collection is created on first run and reused on subsequent runs unless `force_rebuild=True`.
- **Distance metric**: `Cosine` for dense vectors.
- **Dense vector**: produced by the configured embedding backend.
  - Backend `local`: model loaded via `sentence-transformers` or `FlagEmbedding`. Default `BAAI/bge-m3`, dim `1024`.
  - Backend `utopia`: HTTP call to `<UTOPIA_BASE_URL>/ollama/api/embeddings` with `model=embedding_model`. Dimensionality read from the first response and recorded in the manifest; if `embedding_dim` is set in config, it must match the response or indexing fails.
- **Sparse vector**: only stored when `hybrid_enabled=True`.
  - For `BAAI/bge-m3` (local): use the model's native sparse output.
  - For other local models without native sparse output: use `qdrant-client[fastembed]` BM25 sparse encoder client-side.
  - For `utopia` backend: hybrid is supported only if the configured remote model exposes sparse weights; otherwise indexing fails with a clear error and the user must either disable hybrid or switch backend.
- **Embedding input text**: every chunk is embedded using its `text_for_embedding` field from step 01 (not the raw `text`), to preserve legal context like article label and structure path.
- **Payload indexes**: at index creation, payload indexes are created for `law_id`, `law_status`, `index_views`, `article_id`, and `relation_types` so that filters at query time are efficient.

## Idempotency and Rebuild

- Each chunk is stored under a stable point id derived from `chunk_id` (a UUID5 of `chunk_id` keeps Qdrant id constraints satisfied without losing the human-readable identifier in the payload).
- On re-run with `force_rebuild=False`:
  - if a point with the same id exists and its payload `content_hash` matches the new chunk hash, it is skipped;
  - if `content_hash` differs, the point is upserted with the new vectors and payload;
  - if no point with that id exists, it is inserted.
- On re-run with `force_rebuild=True`, the collection is dropped and recreated from scratch.
- The manifest records counts for `inserted`, `updated`, `skipped`, and `removed` (the last only when rebuild is requested).

## Outputs

Default generated output locations:

- retrieval index under `data/indexes/qdrant/` (Qdrant local persistent files);
- indexing artifacts under `data/indexing_runs/<timestamp>/`.

Required artifacts:

- `index_manifest.json`: source dataset hash, embedding backend, embedding model identity, embedding dim, hybrid flag, distance metric, collection name, indexed/inserted/updated/skipped counts, payload index list, schema version, run configuration.
- `payload_profile.json`: coverage and value distributions for filterable metadata fields (`law_id`, `law_status`, `index_views`, `article_id`, `relation_types`).
- `index_quality_report.md`: human-readable validation report.
- optional `diagnostic_queries.json`: results for a small fixed set of diagnostic queries (a few question-shaped probes), used to sanity-check retrieval before running step 05.

## Pipeline

1. Validate the clean dataset contract.
   Why: indexing must fail before embedding when `manifest.json` is missing `ready_for_indexing` or required files are absent.
2. Resolve embedding backend and load the embedder.
   Why: failing here surfaces credential or model issues before any chunk is processed.
3. Create or open the Qdrant collection with the correct dense and (if enabled) sparse vector schema and payload indexes.
   Why: collection topology must match the chosen backend; mismatches must be caught up front.
4. Select chunks for indexing (`full` or `sample`).
   Why: the PoC supports both full runs and small notebook runs while preserving the same contract.
5. Build embedding input from `text_for_embedding` and produce dense (and optionally sparse) vectors in batches.
   Why: batching keeps memory and HTTP behavior predictable.
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
- payload indexes on `law_id`, `law_status`, `index_views`, `article_id`, `relation_types`.

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

Filterable fields must include at least `law_id`, `law_status`, `index_views`, `article_id`, and `relation_types`.

The index manifest must record the source dataset hash, the embedding backend, the embedding model identity, the embedding dimensionality, the hybrid flag, the schema version, and the indexing counts.

## Quality Gates

- Source clean dataset has `ready_for_indexing=True`.
- Embedding backend and model are recorded in the manifest, together with the resolved dimensionality.
- The Qdrant collection topology matches the configured dense (and sparse, when enabled) schema.
- Payload indexes exist for every required filterable field.
- Every indexed point carries the full payload contract.
- Indexed count equals selected count after applying the idempotency policy (insert + update + skip = selected).
- Duplicate `chunk_id` values are rejected.
- Filterable fields are queryable: a smoke filter on `law_status` returns the expected count.
- Diagnostic queries return non-empty results for at least one well-known law.
- Index manifest links back to the exact clean dataset source hash from step 01.

## Notebook Role

`notebooks/03_indexing_contract.ipynb` should:

- validate the clean dataset contract;
- run a small indexing job in `sample` mode to demonstrate the pipeline quickly;
- show the Qdrant collection schema, a representative payload, and the payload profile;
- run a few diagnostic queries (one filter-only, one dense-only, one hybrid) to confirm that retrieval and filters work;
- run the full indexing pipeline on the entire clean dataset and produce the final artifacts.

The notebook should explain the payload fields because they are the bridge between preprocessing (step 01) and RAG behavior (steps 05 and 06), and should explicitly note which retrieval modes the produced collection supports.

## Acceptance Criteria

- The clean dataset can be indexed reproducibly under both `local` and `utopia` embedding backends, with hybrid enabled when the chosen backend supports sparse vectors.
- The collection topology and payload contract are sufficient for simple RAG (step 05) and advanced graph RAG (step 06) without further indexing changes.
- The index can be traced back to the exact clean dataset hash and embedding model identity recorded in the manifest.
- Re-running indexing without changes is a no-op (all points skipped); re-running after a chunk content change updates only the affected points.
- The notebook demonstrates that retrieval and filters work before any answer generation is evaluated downstream.
