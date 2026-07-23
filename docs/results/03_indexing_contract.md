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

## Full run status-v2

La run terminale `status_v2_full_20260723` ha costruito la collection isolata
`legal_chunks_bge_m3_status_v2` in `data/indexes/qdrant_status_v2/`:

- modalità Qdrant `local`, quindi `QdrantClient(path=...)` senza Docker o server;
- `selected=indexed=collection_points=76.499`, zero failure;
- BGE-M3 dense 1024 + sparse, batch embedding 256 e upload batch 64;
- `inserted=76.499`, `vector_updated=0`, `payload_updated=0`, `skipped=0`;
- `embedded=92` e `vector_reused=76.407`;
- `ready_for_retrieval=true` e tutti i quality gate verdi.

I 76.407 vettori invariati provengono dalla run locale `20260512_212818`, collection
`data/indexes/qdrant/::legal_chunks_bge_m3`. Prima del riuso sono stati verificati manifest,
chunks hash, modello, dimensione, topologia dense+sparse e `content_hash` di ogni punto. I 92
contenuti nuovi o modificati sono stati ricalcolati. Il target è stato comunque ricreato e tutti i
payload sono v2: non si tratta di una copia bit-a-bit del vecchio indice.

La run registra:

- manifest SHA-256
  `8620733009fea277b1f457dc60268f1fe761b0a8d70d72371df02bf106cc69a7`;
- payload profile SHA-256
  `b444a320a2efa80032578efa4599506b1c2c77631e93602081938929c350cb63`;
- manifest sorgente riuso SHA-256
  `60c35d07065c75d2707308d792d3ea5114017257ea4d5022e106a28663cfee04`;
- Git `4fcb8bb15792c36c29b3af002f63f708be1e6ef6`, worktree pulito.

`removed=384` descrive point presenti in tentativi parziali della stessa collection prima del
`force_rebuild`; non sono chunk eliminati dal corpus. La collection terminale riconcilia
esattamente tutti i 76.499 chunk clean.

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

Il target sotto usa nomi nuovi per non sovrascrivere la run documentata:

```bash
STATUS_V2_RUN_ID=status_v2_full_20260723_rerun01

PYTHONPATH=src .venv/bin/python -m legal_rag.indexing \
  --clean-dataset-dir data/laws_dataset_clean_status_v2 \
  --index-dir data/indexes/qdrant_status_v2_rerun01 \
  --runs-dir data/indexing_runs \
  --collection-name legal_chunks_bge_m3_status_v2_rerun01 \
  --run-id "$STATUS_V2_RUN_ID" \
  --embedding-backend local \
  --embedding-model BAAI/bge-m3 \
  --embedding-dim 1024 \
  --batch-size 256 \
  --upload-batch-size 64 \
  --chunk-selection-mode full \
  --force-rebuild \
  --require-clean-worktree \
  --reuse-vectors-index-dir data/indexes/qdrant \
  --reuse-vectors-collection legal_chunks_bge_m3 \
  --reuse-vectors-manifest-path data/indexing_runs/20260512_212818/index_manifest.json
```

Il riuso è opzionale: omettendo insieme i tre flag `--reuse-vectors-*` la pipeline ricalcola tutti
i vettori. Contratto e configurazione sperimentale restano gli stessi, ma i nuovi artifact numerici
devono essere validati e non si assume che vettori o ranking siano identici. Notebook:
`notebooks/03_indexing_contract.ipynb`.
