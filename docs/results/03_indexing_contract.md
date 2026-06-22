# 03 — Indicizzazione

Step implementato in `legal_rag.indexing`: valida il dataset pulito, calcola l'embedding di
`text_for_embedding`, scrive point ID deterministici (UUIDv5 da `chunk_id`), salva il payload
completo, crea i payload index sui campi filtrabili e produce artifact riproducibili in
`data/indexing_runs/<run_id>/`. Metodologia in
[note/03](../notes/03_indexing_contract_methodology.md).

## Indice usato dalla pipeline

Gli step 05 e 06 consumano la collection `legal_chunks_bge_m3`, prodotta dalla full run
`bge_m3_full_20260523_095655` in modalità Qdrant server.

- Embedding: `BAAI/bge-m3` locale, dense 1024 + sparse nativo (hybrid abilitato).
- Qdrant: server `http://127.0.0.1:6333`, storage `data/indexes/qdrant_server`, distanza cosine,
  HNSW `m=16` / `ef_construct=100`.
- Idempotenza: `content_hash = sha256(text_for_embedding)`, point ID UUIDv5 da `chunk_id`.
- Validazione: `ready_for_retrieval=true`, `selected = indexed = collection_points = 76.467`,
  `failure_count = 0`, smoke retrieval dense e hybrid non vuoti.

La collection storica `legal_chunks` (Utopia/Nomic, dense-only 768) resta disponibile come baseline;
il confronto retrieval-only tra i due indici è in [06b](06b_retrieval_diagnostics.md).

## Artifact per run

`index_manifest.json`, `payload_profile.json`, `index_quality_report.md`, `diagnostic_queries.json`,
`failures.jsonl`. Il manifest registra hash sorgente, embedding, topologia dei vettori, conteggi,
stato dei payload index e quality gate.

## Riproduzione

Full run BGE-M3 in modalità server, dalla root con Qdrant attivo:

```bash
PYTHONPATH=src python -m legal_rag.indexing --qdrant-url http://127.0.0.1:6333 \
  --index-dir data/indexes/qdrant_server --collection-name legal_chunks_bge_m3 \
  --embedding-backend local --embedding-model BAAI/bge-m3 --embedding-dim 1024
```

Notebook: `notebooks/03_indexing_contract.ipynb`.
