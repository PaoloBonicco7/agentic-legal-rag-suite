# Notebook 04 - Qdrant Indexing Pipeline

## Obiettivo
`notebooks/04_qdrant_indexing_pipeline.ipynb` trasforma `data/laws_dataset_clean` in una collection Qdrant pronta per il runtime RAG:
- refinement chunk deterministico,
- embedding,
- sync incrementale,
- validazioni finali di integrita' e filtri payload.

## Input e prerequisiti
Input principali:
- `data/laws_dataset_clean/manifest.json`
- `data/laws_dataset_clean/chunks.jsonl` (+ file sibling)

Prerequisiti:
- dataset 03 con `ready_to_embedding=true`
- variabili embedding Utopia (`UTOPIA_API_KEY`, `UTOPIA_BASE_URL`, `UTOPIA_EMBED_*`)
- path locali disponibili per Qdrant/artifact

Configurazione notebook (via `IndexingConfig`):
- `dataset_dir`, `qdrant_path`, `artifacts_root`
- `subset_limit`, `force_reembed`, `strict_validation`
- `chunking_profile` (default balanced 20/220/40)

## Flusso step-by-step
1. Setup notebook e import moduli `legal_indexing`.
2. Config run e snapshot configurazione.
3. Debug embedding fail-fast (`discover_utopia_models`, `debug_utopia_embedding_connection`).
4. Validazione dataset (`validate_dataset`).
5. Load bundle + refinement (`load_dataset_bundle`, `refine_chunks_with_diagnostics`).
6. Esecuzione indexing (`run_indexing_pipeline`).
7. Recupero artifact run e check finali su collection/punti/validazioni.

## Output e artifact
### Output principale
- collection Qdrant locale (path `data/indexes/qdrant`)

### Artifact run
`data/qdrant_indexing/<run_id>/`:
- `config.json`
- `dataset_validation.json`
- `chunking_stats.json`
- `chunk_examples.json`
- `collection_info.json`
- `duplicates_validation.json`
- `filter_validation.json`
- `failures.jsonl`
- `indexing_summary.json`

## Contratti con notebook precedente/successivo
- Riceve da notebook 03 un dataset clean coerente e validato.
- Fornisce a notebook 05:
- collection_name risolvibile via artifact,
- payload con campi richiesti dal runtime (`chunk_id`, `law_id`, `article_id`, `text`, `source_*`, `index_views`, `law_status`, ...).

## Note operative essenziali
- Sync incrementale: se `content_hash` invariato, il chunk viene skippato (`skipped_unchanged`).
- `FORCE_REEMBED=true` forza il ricalcolo embedding completo.
- In embedded mode Qdrant alcuni warning su payload index sono attesi e non bloccanti.
- Le validazioni finali (`duplicates_validation`, `filter_validation`) sono parte del contratto di affidabilita'.

## Riferimenti codice
- Notebook: `notebooks/04_qdrant_indexing_pipeline.ipynb`
- Pipeline: `src/legal_indexing/pipeline.py`
- Config: `src/legal_indexing/settings.py`
- IO/validation: `src/legal_indexing/io.py`
- Refinement: `src/legal_indexing/chunk_refinement.py`
- Payload metadata: `src/legal_indexing/metadata.py`
- Store Qdrant: `src/legal_indexing/qdrant_store.py`
- Embeddings: `src/legal_indexing/embeddings.py`
