# Walkthrough: `notebooks/old/02_naive_rag_qdrant_langgraph.ipynb`

> Archivio storico: il notebook e i riferimenti a `src/legal_rag_pipeline/` non sono piu' eseguibili nella codebase corrente.
> La pipeline attuale di indexing e' in `src/legal_indexing/` con notebook `notebooks/04_qdrant_indexing_pipeline.ipynb`.

Questo documento resta come riferimento storico per la baseline precedente.

## Obiettivo del notebook

Costruire una baseline **Naive RAG** per un benchmark MCQ legale e confrontare due varianti di indicizzazione/retrieval:

- `with_metadata`: indicizza `text_for_embedding` (testo arricchito con header/metadati).
- `plain`: indicizza `text` (solo testo “puro”).

La baseline misura:

1) **Retrieval** (senza LLM): `recall@k`, `hit_rate@k`  
2) **End-to-end MCQ** (con LLM): `accuracy`

## Prerequisiti e input dati

- `data/laws_dataset/manifest.json` contiene `dataset_id` (usato per nominare collezioni Qdrant).
- `data/laws_dataset/chunks.jsonl` contiene i chunk indicizzabili (testo + metadati).
- `data/questions.csv` contiene le domande del benchmark.
- `data/laws_html/` contiene gli HTML delle leggi; serve a risolvere “data+numero legge” in `law_id`.
- `.env` deve contenere almeno `UTOPIA_API_KEY` (vedi `.env.example`).

## Schema end-to-end

```mermaid
flowchart TD
  Q["questions.csv"] -->|load_benchmark_questions| QQ["Question[] (gold targets)"]
  H["laws_html/"] -->|CorpusRegistry| QQ
  D["chunks.jsonl"] -->|load_dataset_chunks| CH["ChunkRecord[]"]

  CH -->|build_or_load_qdrant_index (with_metadata)| IDX1["Qdrant collection A"]
  CH -->|build_or_load_qdrant_index (plain)| IDX2["Qdrant collection B"]

  QQ -->|question_to_query| QUERY["query (stem+options)"]
  QUERY -->|similarity_search_with_score| RET["RetrievedDoc[]"]
  RET -->|eval_retrieval| METR["recall@k / hit_rate@k"]

  QUERY -->|retrieve_topk| RET2["RetrievedDoc[]"]
  RET2 -->|_build_context| CTX["context string"]
  CTX -->|LLM prompt| RAW["LLM raw"]
  RAW -->|_parse_answer_json| AJSON["{predicted_label,citations,reasoning,...}"]
  AJSON -->|finalize| RES["MCQResult"]
  RES -->|evaluate_mcq| ACC["accuracy"]
```

## Mappatura notebook → codice

### 1) Setup (notebook)

Nel notebook, la prima cella:

- trova la root del repo (cerca `pyproject.toml` + `src/`);
- aggiunge `src/` al `sys.path` (per importare `legal_rag_pipeline`);
- fa `os.chdir(ROOT)` per rendere stabili i path relativi (`data/...`);
- carica `.env`;
- imposta env var “guardrail”:
  - `RAG_EMBED_MAX_CHARS`, `RAG_EMBED_BATCH_SIZE` (usate in `src/legal_rag_pipeline/qdrant_index.py`)
  - `RAG_CONTEXT_MAX_CHARS`, `RAG_CONTEXT_CHUNK_MAX_CHARS` (usate in `src/legal_rag_pipeline/langgraph_naive.py`)

### 2) Config

File: `src/legal_rag_pipeline/config.py`

- `RunConfig` contiene path e parametri di run (scope, k, ecc.).
- `make_default_config(**overrides)` costruisce la config e applica override dal notebook.
- `utopia_*` sono letti da env (`.env`).

### 3) Load benchmark (domande + gold)

File: `src/legal_rag_pipeline/dataset.py` + `baselines/benchmark.py`

- `load_benchmark_questions(questions_csv, html_dir=...)`
  - costruisce un `CorpusRegistry` dagli HTML (`laws_ingestion.registry`)
  - legge il CSV e produce tuple di `Question`
  - per ogni riferimento “Legge regionale ... Art. ...” risolve un `law_id`
  - produce anche `gold_targets`: coppie `(law_id, article_label_norm)`
- `build_benchmark_law_filter(questions)` estrae tutte le `law_id` dei gold target per filtrare il dataset quando `scope=benchmark`.
- `dataset_id_from_manifest(dataset_dir)` legge `manifest.json` e ritorna `dataset_id`.

### 4) Load chunks (dataset)

File: `src/legal_rag_pipeline/dataset.py`

- `load_dataset_chunks(dataset_dir, law_filter=...)` legge `chunks.jsonl` e costruisce `ChunkRecord`.
- Ogni record include sia:
  - `text` (solo testo)
  - `text_for_embedding` (testo con header di metadati)
  - `metadata` (law_id, article_label_norm, passage_label, relazioni, ecc.)

Nota: il filtro `law_filter` è applicato su `law_id` e serve a velocizzare la baseline su benchmark.

### 5) Build/Load Qdrant (A/B)

File: `src/legal_rag_pipeline/qdrant_index.py`

Punti chiave:

- `collection_name_from(dataset_id, scope, variant)` genera nomi del tipo `lrag_<ds>_<scope>_<variant>`.
- Qdrant è usato in modalità **embedded** con path locale (`RunConfig.qdrant_path`).
- Gestione lock:
  - `_get_or_create_client(path)` cachea i client (evita riaperture).
  - se il path è “locked” (errore tipico: “already accessed by another instance...”), usa un fallback `..._pid<PID>`.
- Invalidation “semplice”:
  - salva un file meta `qdrant_path/_meta/<collection>.json`
  - se meta esistente != meta attuale (dataset_id/scope/variant/chunk_count), cancella e ricrea la collezione
- Variante testo indicizzato:
  - `_document_text(chunk, variant)` sceglie `text_for_embedding` vs `text`
- Inserimento testi:
  - `_prepare_text_for_embedding` tronca a `RAG_EMBED_MAX_CHARS`
  - `_add_texts_with_fallback` prova bulk upsert; se fallisce per limiti del backend, fa fallback per-elemento con ulteriori tronchetti
- IDs deterministici:
  - `_qdrant_point_id(chunk_id)` produce UUIDv5 → stesso `chunk_id` ⇒ stesso point id

Il valore ritornato è `RetrieverHandle` (`src/legal_rag_pipeline/types.py`) che espone `similarity_search_with_score(...)` delegando al `vector_store` LangChain.

### 6) Retrieval evaluation

File: `src/legal_rag_pipeline/eval_retrieval.py` + `src/legal_rag_pipeline/retrieval.py`

- `question_to_query(question, include_options=True)` (in `dataset.py`): query = stem + opzioni.
- `retrieve_topk(query, retriever, k)`:
  - chiama `retriever.similarity_search_with_score`
  - normalizza output in `RetrievedDoc` (incluso `chunk_id`, `metadata`, `text`)
- Gold:
  - `_gold_keys(question)` produce set di `(law_id, article_label_norm)` dai gold target.
- Calcolo metriche:
  - per ogni k: confronta i top-k `doc.article_key` con i gold key.
  - `recall@k`: `|overlap| / |gold|`
  - `hit_rate@k`: `1` se overlap non vuoto, altrimenti `0`
- `evaluate_retrieval_with_traces` produce anche `traces` (utile per debug offline).

### 7) LangGraph Naive RAG (MCQ end-to-end)

File: `src/legal_rag_pipeline/langgraph_naive.py`

Concetto: pipeline minimale “retrieve → build_context → answer”.

1) Query: `question_to_query(question, include_options=True)`
2) Retrieval: `retrieve_topk(query, retriever, k)`
3) Context building:
   - `_build_context(docs)` concatena blocchi con header:
     - indice [i]
     - chunk_id
     - law_id, article_label_norm, passage_label
   - applica limiti:
     - massimo totale `RAG_CONTEXT_MAX_CHARS`
     - massimo per chunk `RAG_CONTEXT_CHUNK_MAX_CHARS`
4) Prompting:
   - `_answer_prompt(question, context)` chiede **SOLO JSON** con:
     - `predicted_label` (A-F)
     - `citations` (chunk_id citati)
     - `reasoning`
5) Robustezza parsing:
   - `_parse_answer_json(...)` prova a estrarre un oggetto `{...}` dal testo e fare `json.loads`.
   - `_normalize_prediction(...)` forza la label a A-F; se fallisce sceglie la prima label “valida”.
6) Risultato:
   - `_finalize_result(...)` costruisce `MCQResult` e filtra `citations` tenendo solo chunk_id effettivamente recuperati.

LangGraph:

- `run_langgraph_naive(...)` prova a importare `langgraph`.
  - se non disponibile: fallback `_run_linear(...)` (stessa logica, senza grafo).
  - se disponibile: costruisce un `StateGraph` con nodi lineari e lo cachea in `_APP_CACHE` (riusa il grafo tra domande).

### 8) MCQ metrics

File: `src/legal_rag_pipeline/eval_mcq.py`

- `evaluate_mcq(results)` calcola:
  - `accuracy` complessiva
  - breakdown per `level`

### 9) Artifacts (salvataggi)

File: `src/legal_rag_pipeline/artifacts.py`

- `make_run_dir(runs_root)` crea `data/runs/naive_rag/<UTC timestamp>/`
- `save_run_artifacts(...)` salva:
  - `config.json`
  - `retrieval_ab_metrics.json`
  - `mcq_ab_metrics.json`
  - `retrieval_traces_<variant>.jsonl`
  - `mcq_results_<variant>.jsonl`
  - `summary.md`

## Perché “with_metadata” può aiutare

Perché `text_for_embedding` include un header con contesto (es. “[LR 1950-10-10 n.1] ... | Art. ... | ... |”) che rende più discriminante la rappresentazione vettoriale quando i chunk testuali sono simili.

In pratica: la query contiene spesso riferimenti impliciti a “articolo X” e l’header può aumentare la probabilità che il retriever recuperi chunk dello stesso articolo/legge.

## Idee per rendere il notebook meno black-box (senza cambiare la pipeline)

- Stampare/visualizzare 1 esempio completo:
  - query generata (`question_to_query`)
  - top-k chunk_id recuperati + metadati (law_id, articolo)
  - contesto finale (magari tronco)
  - output LLM raw + JSON parsato
- Aggiungere link/percorsi ai file di output salvati (run_dir).
- Esplicitare che l’eval MCQ fa **N chiamate LLM** (costo/tempo).
- Documentare i failure mode più comuni:
  - `UTOPIA_API_KEY` mancante
  - lock Qdrant sul path
  - errori di context length (e i guardrail env var)
