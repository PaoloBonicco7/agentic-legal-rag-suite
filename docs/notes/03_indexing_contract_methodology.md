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
- stato di legge e articolo;
- titolo, data e numero della legge;
- file sorgente;
- viste di indicizzazione;
- relazioni entranti, uscenti e tipi relazione.

Questa separazione permette al modello embedding di vedere il contesto giuridico, ma consente al retrieval di restituire una fonte leggibile e filtrabile.

## Qdrant

Qdrant puo essere usato in due modalita:

- file locale persistente tramite `qdrant-client`, con default `data/indexes/qdrant`;
- server locale Docker, passando `qdrant_url="http://127.0.0.1:6333"` e usando storage dedicato `data/indexes/qdrant_server`.

La collection contiene:

- vettore named `dense`, con distanza cosine;
- vettore named `sparse`, quando `hybrid_enabled=True`;
- payload on disk;
- indici payload sui campi filtrabili richiesti dalla spec.

La modalita locale riduce le dipendenze operative dello step. La modalita Docker e preferibile per una full run quando si vogliono payload indexes effettivi, meno limiti del client embedded sopra 20.000 punti e un comportamento piu vicino a un deployment server.

In modalita embedded locale, `qdrant-client` accetta la richiesta di payload index ma segnala che gli indici non hanno effetto prestazionale come in un server Qdrant. La pipeline registra comunque i campi richiesti e mantiene lo stesso contratto, cosi il passaggio a server Qdrant non richiede cambiare payload o retrieval.

Per le full run Docker la pipeline puo usare upload parallelo, un numero di shard esplicito e una soglia temporanea di indexing alta. Questo differisce la costruzione HNSW durante il bulk upload; la soglia normale viene ripristinata a fine ingest.

Comando operativo:

```bash
docker compose -f docker-compose.qdrant.yml up -d qdrant
PYTHONPATH=src .venv/bin/python -m legal_rag.indexing --qdrant-url http://127.0.0.1:6333 --index-dir data/indexes/qdrant_server --collection-name legal_chunks_bge_m3 --force-rebuild --embedding-backend local --embedding-model BAAI/bge-m3 --embedding-dim 1024 --qdrant-upload-parallel 4 --qdrant-shard-number 4 --qdrant-bulk-indexing-threshold-kb 10000000
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
qdrant_url = http://127.0.0.1:6333
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

Ogni payload registra:

```text
content_hash = sha256(text_for_embedding.strip())
```

Quando la collection viene riutilizzata, i chunk invariati vengono saltati se il `content_hash` gia presente coincide. Se il contenuto cambia, il punto viene upsertato con nuovi vettori e nuovo payload.

Questo rende le rerun economiche e impedisce duplicati.

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
- non esistono `chunk_id` duplicati;
- i campi payload obbligatori sono presenti;
- i filtri principali sono queryable;
- le query diagnostiche non falliscono.

Questi controlli servono a intercettare errori prima degli step RAG, dove sarebbero piu difficili da distinguere da problemi di retrieval o generazione.
