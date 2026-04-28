# Overview Progetto

## Contesto e obiettivo
Il progetto implementa una suite **Legal RAG** notebook-centrica per question answering su normativa regionale in HTML.
L'obiettivo operativo e' costruire una pipeline end-to-end, tracciabile e verificabile, che trasformi il corpus normativo in:
- dataset strutturato per retrieval,
- indice vettoriale Qdrant con metadati ricchi,
- baseline Naive RAG valutabile su benchmark MCQ e no-hint.

## Approccio end-to-end
### 1) HTML -> ingestion strutturata
Pipeline in `src/laws_ingestion/data_preparation/laws_graph/pipeline.py`:
- inventory corpus,
- parsing non distruttivo (laws/articles/notes),
- classificazione status,
- normalizzazione relazioni,
- estrazione eventi,
- arricchimento chunk con `index_views` (`historical`, `current`),
- quality gates e `ready_to_embedding`.

Output pubblico:
- `data/laws_dataset_clean/laws.jsonl`
- `data/laws_dataset_clean/articles.jsonl`
- `data/laws_dataset_clean/notes.jsonl`
- `data/laws_dataset_clean/edges.jsonl`
- `data/laws_dataset_clean/events.jsonl`
- `data/laws_dataset_clean/chunks.jsonl`
- `data/laws_dataset_clean/manifest.json`

### 2) Dataset clean -> chunk refinement + indexing Qdrant
Pipeline in `src/legal_indexing/pipeline.py`:
- validazione dataset (`validate_dataset`),
- ricostruzione passages (`reconstruct_passages`),
- refinement deterministico (`refine_chunks_with_diagnostics`),
- payload metadata completo (`refined_chunk_payload`),
- sync incrementale su Qdrant (`sync_points_incremental`),
- validazioni post-indexing (duplicati, filtri payload).

### 3) Qdrant -> runtime Naive RAG
Runtime in `src/legal_indexing/rag_runtime/langgraph_app.py` con 4 nodi reali:
1. `normalize_query`
2. `retrieve_top_k`
3. `build_context`
4. `generate_answer_structured`

Il runtime include:
- risoluzione contratto indice (`index_contract.py`),
- introspezione payload e check campi richiesti,
- verifica fail-fast dimensione embedding/query vs collection,
- provenance e citazioni filtrate sui chunk realmente recuperati.

## Perche' Qdrant
Scelta implementativa coerente col codice corrente (`src/legal_indexing/qdrant_store.py`):
- **Vector search + payload filtering** nello stesso backend (`law_id`, `index_views`, `law_status`, ecc.).
- **Indicizzazione incrementale**: skip automatico dei chunk invariati via `content_hash`.
- **Determinismo**: `point_id` stabile da `chunk_id` + hash payload/content.
- **Auditabilita'**: artifact JSON/JSONL per ogni run (`data/qdrant_indexing/<run_id>/`).
- **Modalita' embedded** utile per PoC locale, con passaggio semplice a deployment server quando serve.

## Evaluation: notebook 01 e 02
### Notebook 01 - build `questions_no_hint.csv`
`notebooks/evaluation/01_build_questions_no_hint.ipynb` prepara il dataset no-hint a partire da `questions.csv`:
- parsing opzioni `A-F` dal campo `Domanda`,
- estrazione dello stem,
- mapping della label corretta in risposta testuale,
- normalizzazione riferimenti normativi,
- validazioni hard (100 righe, livelli bilanciati, no NaN).

### Notebook 02 - baseline no-RAG
`notebooks/evaluation/02_evalutation_no_rag.ipynb` misura baseline senza retrieval:
- flusso MCQ structured output (`McqAnswer`),
- flusso no-hint + judge (`NoHintAnswer` + `JudgeResult` 0/1),
- aggregazioni metriche (`processed`, `judged`, `score_sum`, `accuracy`, `errors`, `by_level`).

Le utility condivise sono in `src/legal_indexing/rag_runtime/benchmarking.py` e sono riusate anche nel notebook 05.

## Snapshot locale al 2 marzo 2026
### Ingestion clean
Fonte: `data/laws_dataset_clean/manifest.json` + artifact run.
- `run_id`: `20260301_222640`
- `created_at`: `2026-03-01T22:26:43Z`
- `counts`: laws=5, articles=104, notes=193, edges=434, events=299, chunks=693
- `ready_to_embedding`: `true`
- `qa_gates`: tutti `true`

Nota run: nel `run_config` il dataset clean e' stato prodotto con `sample_size=5` su corpus scoperto di 3145 file HTML.

### Indexing Qdrant
Fonte: `data/qdrant_indexing/20260301_163556/indexing_summary.json`.
- `run_id`: `20260301_163556`
- `total_passages`: 74436
- `total_refined_chunks`: 50978
- `collection_points_count`: 50978
- `total_embedded`: 0
- `skipped_unchanged`: 50978
- `failures`: 0
- validazioni: `duplicate_chunk_ids_ok=true`, `filter_validation_ok=true`

### Mini benchmark Naive RAG
Fonte: `notebooks/rag_pipeline/artifacts/rag_naive_mini_benchmark_20260301_231227.json`.
- `mcq_summary`: processed=20, judged=0, errors=20
- `no_hint_summary`: processed=20, judged=0, errors=20

Artifact associati:
- `notebooks/rag_pipeline/artifacts/rag_naive_mini_benchmark_20260301_231227_mcq.csv`
- `notebooks/rag_pipeline/artifacts/rag_naive_mini_benchmark_20260301_231227_no_hint.csv`

## Documentazione operativa
- indice docs: `docs/index.md`
- notebook 01: `docs/notebook_01_build_questions_no_hint.md`
- notebook 02: `docs/notebook_02_evaluation_no_rag.md`
- notebook 03: `docs/notebook_03_laws_graph_pipeline.md`
- notebook 04: `docs/notebook_04_qdrant_indexing_pipeline.md`
- notebook 05: `docs/notebook_05_langgraph_rag_pipeline.md`

`docs/archive/` resta storico e non e' riferimento operativo corrente.
