# 03 — Indicizzazione

`legal_rag.indexing` valida il dataset pulito, calcola l'embedding di `text_for_embedding`, usa
point ID deterministici (UUIDv5 da `chunk_id`), salva payload e vettori dense+sparse e produce gli
artefatti in `data/indexing_runs/<run_id>/`. Metodologia in
[note/03](../notes/03_indexing_contract_methodology.md).

## Contratto v2

`indexing-contract-v2` verifica gli SHA-256 reali del corpus e di ogni output clean, registra
identità del dataset e versioni in ogni payload e richiede distribuzioni coerenti. L'idempotenza
confronta due hash:

- `content_hash = sha256(text_for_embedding)`;
- `payload_hash = sha256(payload canonico)`.

Se il testo è invariato viene aggiornato soltanto il payload tramite `set_payload`; gli
inserimenti, gli aggiornamenti vettoriali, gli aggiornamenti payload-only e gli skip sono
conteggiati separatamente. La full run definitiva usa comunque `force_rebuild=true`.

Gli indici keyword vengono richiesti prima dell'upload per `law_status`, `article_status`,
`passage_status`, `content_availability`, `index_views` e gli altri campi filtrabili. In Qdrant
locale il client avverte correttamente che tali indici non migliorano le prestazioni: il controllo
resta utile per la portabilità del contratto.

## Smoke test status-v2

La run `status_v2_sample` verifica il contratto end-to-end senza fornire evidenza sulle
performance retrieval:

- collection locale `legal_chunks_bge_m3_status_v2_sample`;
- 16 point selezionati, indicizzati e presenti nella collection; zero failure;
- `BAAI/bge-m3`, dense 1024 + sparse nativo;
- `ready_for_retrieval=true`; tutti i gate di hash, identità e distribuzione verdi;
- corpus SHA-256
  `aa46ea3758c1de5596a8902b95e90cdfc6d08fbc2956086495a020680be22459`;
- clean manifest SHA-256
  `65eb3cdca461a2a8d894959be9df3a34f3e03cc975d6df08c40505111da51b23`;
- chunk SHA-256
  `dacbb03fd3debfd272031226ad539100cc228141acca94fba304fa28c663cb95`.

## Indice storico v1

Gli step 05, 06 e il baseline storico 06b consumavano `legal_chunks_bge_m3`, run
`bge_m3_full_20260523_095655`, in modalità Qdrant server:

- 76.467 point, `BAAI/bge-m3` dense 1024 + sparse;
- storage `data/indexes/qdrant_server`, distanza cosine, HNSW
  `m=16` / `ef_construct=100`;
- `ready_for_retrieval=true`, zero failure.

Questo indice resta necessario per riprodurre i risultati precedenti, ma non è l'indice
definitivo dell'audit di vigenza. La collection ancora precedente `legal_chunks`
(Utopia/Nomic, dense-only 768) resta una baseline storica.

## Artefatti per run

`index_manifest.json`, `payload_profile.json`, `index_quality_report.md`,
`sample_retrieval_report.json` e `failures.jsonl`. Il manifest registra hash sorgente, hash degli
output clean, identità Git e della pipeline, embedding, topologia vettoriale, conteggi,
distribuzioni payload e quality gate.

## Riproduzione della full run status-v2

```bash
PYTHONPATH=src python -m legal_rag.indexing \
  --clean-dataset-dir data/laws_dataset_clean_status_v2 \
  --index-dir data/indexes/qdrant_status_v2 \
  --runs-dir data/indexing_runs \
  --collection-name legal_chunks_bge_m3_status_v2 \
  --run-id status_v2_full_20260723 \
  --embedding-backend local \
  --embedding-model BAAI/bge-m3 \
  --embedding-dim 1024 \
  --batch-size 64 \
  --upload-batch-size 64 \
  --chunk-selection-mode full \
  --force-rebuild \
  --require-clean-worktree
```

La sezione verrà completata con conteggi e hash del manifest terminale della full run. Notebook:
`notebooks/03_indexing_contract.ipynb`.
