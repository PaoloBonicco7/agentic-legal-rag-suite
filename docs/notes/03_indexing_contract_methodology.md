# 03 - Metodologia di Indicizzazione

## Obiettivo

Questo step trasforma il dataset pulito delle leggi in una collection Qdrant pronta per retrieval semplice e graph-aware.

La scelta metodologica principale e mantenere l'indice come artifact riproducibile: ogni punto indicizzato deve essere riconducibile al chunk sorgente, al dataset pulito usato per generarlo, al modello embedding e alla configurazione Qdrant della run.

## Input di Partenza

La pipeline consuma solo `data/laws_dataset_clean/` quando il relativo `manifest.json` espone `ready_for_indexing=true`.

Prima di creare embedding viene verificata la presenza dei file necessari:

- `manifest.json`
- `chunks.jsonl`
- `laws.jsonl`
- `articles.jsonl`
- `edges.jsonl`

Questa validazione anticipata evita di spendere tempo su embedding o scritture Qdrant quando il contratto del dataset pulito non e rispettato.

## Strategia di Indicizzazione

L'unita indicizzata e il chunk prodotto dallo step 01. Il testo passato al modello embedding e `text_for_embedding`, non `text`, perche contiene il contesto minimo utile al recupero: legge, articolo, percorso strutturale e testo del passaggio.

Il payload Qdrant mantiene invece il testo originale in `text`, insieme ai metadati necessari per retrieval e spiegabilita:

- identificatori di chunk, passaggio, articolo e legge;
- stato di legge, articolo e passaggio;
- disponibilità del contenuto, eventi e regole di stato;
- titolo, data e numero della legge;
- file sorgente;
- viste di indicizzazione;
- relazioni entranti, uscenti e tipi relazione.
- hash di contenuto/payload e identità del dataset clean.

Questa separazione permette al modello embedding di vedere il contesto giuridico, ma consente al retrieval di restituire una fonte leggibile e filtrabile.

## Qdrant

La pipeline di tesi usa Qdrant in modalità file locale persistente tramite
`QdrantClient(path=...)`. Non richiede Docker, un demone Qdrant o una porta locale: il processo
Python legge e scrive direttamente il path configurato. Docker o un server sono necessari soltanto
impostando esplicitamente `qdrant_url`.

Ogni esperimento definitivo che cambia contratto usa un path e una collection separati.

La collection contiene:

- vettore named `dense`, con distanza cosine;
- vettore named `sparse`, quando `hybrid_enabled=True`;
- payload on disk;
- indici payload sui campi filtrabili richiesti dalla spec.

Gli indici keyword vengono dichiarati prima dell'upload per stati, viste, disponibilità e identità
legali. Il client embedded può segnalare che non hanno lo stesso effetto prestazionale di un server;
il contratto payload resta comunque verificato e riproducibile.

Comando di riproduzione con target isolato (la run documentata non viene sovrascritta):

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

## Embedding

Il backend locale rimane disponibile:

```text
embedding_backend = local
embedding_model = BAAI/bge-m3
hybrid_enabled = True
```

`BAAI/bge-m3` e stato scelto perche supporta italiano e produce sia rappresentazioni dense sia sparse. Questo permette di costruire nello stesso indice la base per retrieval semantico e hybrid retrieval.

La collection storica `legal_chunks` puo rimanere disponibile come baseline dense-only. Per gli esperimenti di miglioramento retrieval si costruisce invece una collection parallela:

```text
collection_name = legal_chunks_bge_m3
index_dir = data/indexes/qdrant
embedding_backend = local
embedding_model = BAAI/bge-m3
hybrid_enabled = True
```

La separazione per collection evita di sovrascrivere l'indice precedente e rende confrontabili baseline e nuovo retrieval. `force_rebuild=True` ricrea solo la collection configurata, non l'intero path Qdrant.

Utopia resta supportato come backend dense-only: `hybrid_enabled` viene disabilitato perche la pipeline non assume disponibilita remota di sparse weights. Per sbloccare hybrid retrieval, la scelta metodologica e quindi BGE-M3 locale.

## Idempotenza

Ogni punto Qdrant usa un ID stabile:

```text
point_id = uuid5(NAMESPACE_URL, chunk_id)
```

Ogni payload registra due identità distinte:

```text
content_hash = sha256(text_for_embedding.strip())
payload_hash = sha256(canonical_payload_without_payload_hash)
```

Quando entrambi coincidono il punto viene saltato. Se cambia soltanto il payload, la pipeline usa
`set_payload` e conserva i vettori; se cambia il contenuto, rigenera embedding e punto.

Prima dell'embedding vengono inoltre ricalcolati gli hash reali di manifest e chunks. L'indice
registra tali hash, le versioni preprocessing/status e la code identity in manifest e payload:
una collection non può quindi essere accettata soltanto perché nome e conteggio sembrano corretti.

## Riuso verificato dei vettori

Una full rebuild isolata può riusare i vettori di una collection locale precedente senza fidarsi
del solo nome. La pipeline richiede insieme path, collection e manifest sorgente, quindi verifica:

- run e manifest sorgente pronti per retrieval;
- stesso modello, dimensione dense, presenza sparse e topologia;
- hash del vecchio `chunks.jsonl`;
- corrispondenza puntuale di `chunk_id` e `content_hash`.

Soltanto i contenuti invariati riusano dense e sparse; ogni contenuto nuovo o modificato viene
ricalcolato con BGE-M3. Il target resta ricreato con `force_rebuild` e riceve sempre il payload v2.
Il manifest distingue `vector_reused_count` da `embedded_count` e conserva hash e identità della
sorgente, così il risparmio computazionale non indebolisce la tracciabilità.

## Artifact di Run

Ogni esecuzione produce una cartella in `data/indexing_runs/<run_id>/`.

Gli artifact principali sono:

- `index_manifest.json`: configurazione, hash sorgente, dimensione embedding, conteggi e quality gates;
- `payload_profile.json`: copertura dei campi payload;
- `index_quality_report.md`: riepilogo leggibile della validazione;
- `diagnostic_queries.json`: risultati di query diagnostiche;
- `sample_retrieval_report.json`: versione compatta per ispezione notebook;
- `failures.jsonl`: errori puntuali, se presenti.

Il manifest e il riferimento principale per collegare una run di retrieval o valutazione alla specifica versione del dataset indicizzato.

## Modalita Operative

Sono previste due modalita:

- `sample`: indicizza un sottoinsieme limitato, utile per notebook, debug e dimostrazione;
- `full`: indicizza tutto `chunks.jsonl`, producendo l'indice usabile dagli step successivi.

Entrambe usano la stessa pipeline. La modalita sample non e una pipeline separata: cambia solo la selezione dei chunk.

Nel notebook 03 la full run BGE-M3 e protetta da una variabile esplicita (`RUN_BGE_M3_REINDEX`). Il callback di progresso scrive eventi JSONL in `data/indexing_runs/<run_id>_progress.jsonl` e stampa batch, percentuale, rate, ETA, upsert, skip e failure. Questo rende monitorabile una run lunga senza introdurre un orchestratore separato.

## Quality Gates

La run e considerata pronta solo se:

- il dataset pulito e valido;
- almeno un chunk viene selezionato;
- la dimensione embedding e rilevata e coerente con `embedding_dim`, se configurata;
- tutti i punti selezionati sono indicizzati o saltati come invariati;
- gli eventuali vettori riusati provengono da una collection compatibile e hanno content hash
  identico;
- non esistono `chunk_id` duplicati;
- i campi payload obbligatori sono presenti;
- i filtri principali sono queryable;
- le query diagnostiche non falliscono.

Questi controlli servono a intercettare errori prima degli step RAG, dove sarebbero piu difficili da distinguere da problemi di retrieval o generazione.
