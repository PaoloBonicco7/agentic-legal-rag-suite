# Notebook 05 - Naive RAG Pipeline

## Obiettivo
`notebooks/05_langgraph_rag_pipeline.ipynb` implementa una baseline Naive RAG chiara e confrontabile con il notebook 02.

La baseline separa esplicitamente:
1. retrieval e costruzione contesto,
2. inferenza task (MCQ o no-hint),
3. judge (solo per no-hint).

Scopo: avere metriche leggibili, diagnosi rapida dei fallimenti, e una base estendibile per notebook advanced.

## Input e prerequisiti
Input principali:
- collection Qdrant prodotta dal notebook 04,
- `data/evaluation/questions.csv`,
- `data/evaluation/questions_no_hint.csv`.

Prerequisiti runtime:
- contratto indice risolvibile da artifact notebook 04,
- embedding compatibile con la collection (vector size),
- variabili scoring: `UTOPIA_API_KEY`, `UTOPIA_BASE_URL` (o `UTOPIA_OLLAMA_CHAT_URL`).

Configurazione benchmark rilevante:
- `BENCHMARK_START_POS`, `BENCHMARK_LIMIT`,
- `NO_HINT_MAX_RETRIES` (default `1`),
- `SKIP_JUDGE_IF_EMPTY` (default `True`),
- `MIN_ANSWER_CHARS` (default `8`),
- `FAIL_FAST_SCORING_SMOKE` (default `True`).

## Flusso operativo del notebook
1. Setup ambiente e librerie.
2. Configurazione runtime naive (`RagRuntimeConfig`).
3. `prepare_runtime(config)` con validazioni dataset/indice/payload.
4. Build grafo LangGraph naive e visualizzazione Mermaid.
5. Debug domanda singola e batch qualitativo breve.
6. Caricamento dataset evaluation allineati al notebook 02.
7. Setup scoring structured e smoke test fail-fast endpoint.
8. Benchmark MCQ:
- query retrieval = `question_no_hint` (stem),
- `run_rag_retrieval_context(...)` (solo 3 nodi),
- una sola chiamata LLM structured per classificazione `McqAnswer`.
9. Benchmark no-hint + judge:
- stessa query retrieval dello scenario MCQ,
- chiamata LLM structured `NoHintAnswer`,
- retry su risposta vuota/non valida,
- judge `JudgeResult` solo se la risposta e' valida (policy default).
10. Aggregazioni metriche legacy + estese.
11. Sezioni debug:
- error heads e categorie errore,
- debug retrieval per domanda `1..N`,
- debug scoring per domanda,
- timing diagnostics per stage.
12. Persistenza artifact JSON/CSV.
13. Cleanup risorse.

## Contratto metriche
Ogni dataset (`mcq`, `no_hint`) mantiene retrocompatibilita con notebook 02:
- `processed`,
- `judged`,
- `score_sum`,
- `accuracy`,
- `errors`,
- `by_level`.

Metriche estese aggiunte:
- `coverage = judged / processed`,
- `strict_accuracy = score_sum / processed`,
- `empty_answer_count` (utile per no-hint),
- `error_categories`,
- `timing_summary` (`mean`, `p50`, `p90`, `min`, `max`) per:
  - `t_retrieval_context_s`,
  - `t_task_llm_s`,
  - `t_judge_s`,
  - `t_total_s`.

Interpretazione:
- `accuracy` misura la qualita sui soli record giudicati,
- `coverage` misura quanta parte e' stata effettivamente valutata,
- `strict_accuracy` misura l'end-to-end sul totale processato.

## Debug e troubleshooting
### Caso tipico: `processed > 0` ma `judged = 0`
Prima verifica:
- sezione smoke test scoring (`9.1`),
- `transport_diagnostics` nell'artifact JSON,
- `error_heads_*` e `error_categories`.

Cause comuni:
- endpoint chat non valido (`HTTP 405`, path errato),
- errore schema/validazione structured,
- risposte no-hint vuote persistenti.

Diagnosi retrieval:
- usare `debug_retrieval_for_question(qnum_1_based, mode=...)` per ispezionare chunk, score, metadata e contesto.

Diagnosi scoring:
- usare `debug_scoring_for_question(qnum_1_based, mode='mcq' | 'no_hint_judge')` per vedere prompt e output structured.

## Artifact prodotti
Il notebook salva:
- `notebooks/rag_pipeline/artifacts/rag_naive_mini_benchmark_<timestamp>.json`,
- `notebooks/rag_pipeline/artifacts/rag_naive_mini_benchmark_<timestamp>_mcq.csv`,
- `notebooks/rag_pipeline/artifacts/rag_naive_mini_benchmark_<timestamp>_no_hint.csv`.

Nel JSON sono presenti:
- summary legacy,
- summary estese,
- breakdown errori,
- diagnostica transport/smoke,
- timing diagnostics,
- risultati riga per riga.

## Riferimenti codice
- Notebook: `notebooks/05_langgraph_rag_pipeline.ipynb`
- Runtime app/grafo: `src/legal_indexing/rag_runtime/langgraph_app.py`
- Benchmark utility: `src/legal_indexing/rag_runtime/benchmarking.py`
- Prompt e schema: `src/legal_indexing/rag_runtime/prompts.py`, `src/legal_indexing/rag_runtime/schemas.py`
- Retrieval Qdrant: `src/legal_indexing/rag_runtime/qdrant_retrieval.py`
- Test baseline: `tests/test_rag_langgraph_baseline.py`, `tests/test_rag_benchmarking.py`
