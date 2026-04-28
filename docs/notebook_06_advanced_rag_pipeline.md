# Notebook 06 - Advanced RAG Pipeline (Qdrant + Graph-aware retrieval)

## Obiettivo
Il notebook 06 implementa una pipeline RAG avanzata, comparabile con la baseline naive, con focus sul KPI no-hint + judge.

Contract metriche mantenuto compatibile con notebook 02/05:
- `processed`
- `judged`
- `accuracy`
- `by_level`
- `errors`

## Architettura aggiornata
Pipeline advanced (`pipeline_mode='advanced'`):
1. rewrite/decomposition opzionale;
2. metadata filtering ibrido (esplicito + heuristics);
3. retrieval multi-query;
4. **hybrid search** dense + sparse (BM25-like) con fusione RRF/weighted;
5. graph expansion con gating su query specifiche;
6. reranking deterministico (retrieval + metadata + lexical + sparse score);
7. context building con provenance e diversity control;
8. generation strutturata con answer guard (`retry + fallback non-vuoto`).

## Hybrid Search (Qdrant sparse + dense)
### Indexing
Nel run di indicizzazione vengono prodotti:
- collection con `sparse_vectors_config` (default `bm25`);
- upsert point con vettore dense + sparse;
- artifact encoder sparse (`sparse_encoder.json`) in `data/qdrant_indexing/<run_id>/`.

### Runtime
Il retriever advanced usa:
- `query_dense(...)`
- `query_sparse(...)`
- `query_hybrid(...)` con fusione configurabile:
  - `fusion_method='rrf'` (default)
  - `fusion_method='weighted_sum'`

Output diagnostico row-level:
- `retrieval_mode` (`dense_only|hybrid|fallback_dense`)
- `dense_retrieved_count`
- `sparse_retrieved_count`
- `fusion_overlap_count`
- `context_included_count`
- `reference_law_hit`
- `failure_category`

## Failure taxonomy
Per il no-hint ogni row può essere classificata in:
- `retrieval_miss`: riferimento atteso non coperto o retrieval nullo/errore;
- `context_noise`: retrieval presente ma evidenza non centrata;
- `abstention`: risposta astensiva/fallback/`needs_more_context`;
- `contradiction`: risposta non coerente con valutazione judge.

## Struttura notebook 06
Ogni code cell è preceduta da markdown esplicativa.

Sezioni principali:
1. setup/import;
2. config advanced + toggle mini/full;
3. prepare runtime + contract checks;
4. debug single query con trace;
5. benchmark advanced;
6. benchmark naive di controllo (stesse posizioni);
7. summary compatibile + confronto naive vs advanced;
8. persistenza artifact;
9. dashboard comparative + diagnostica hybrid/failure;
10. cleanup.

## Dashboard e lettura risultati
Grafici principali:
1. accuracy globale Naive vs Advanced (MCQ e no-hint);
2. accuracy no-hint per livello Naive vs Advanced;
3. failure decomposition per livello (stacked);
4. contributo dense/sparse e overlap per livello;
5. scatter `context_included_count` vs score;
6. scatter `sparse_contribution` vs score;
7. empty-answer rate e fallback-used per livello;
8. `eval references covered vs missing` da `index_contract`.

Tabella "Top casi scarsi":
- `qid`, `level`, `predicted_answer`, `rag_answer_source`,
- `rag_was_empty_before_guard`, `retrieved_count`,
- `final_binary_score`, `failure_category`, `judge_result.justification`.

### Coverage contract (prima del benchmark)
Il run di indexing produce `index_contract` in `indexing_summary.json` con:
- `eval_reference_coverage`
- `missing_references_sample`
- `payload_field_coverage`

Nel notebook 06 il dashboard espone questi indicatori per validare rapidamente
se l'indice e' allineato alle domande di evaluation prima di interpretare le metriche RAG.

## Artifact prodotti
Directory: `notebooks/rag_pipeline/artifacts/`

- `rag_advanced_<mini|full>_benchmark_<ts>.json/csv`
- `rag_naive_<mini|full>_benchmark_<ts>.json/csv`
- `rag_comparison_naive_vs_advanced_<ts>.json`

Il payload comparison include:
- delta globali MCQ/no-hint;
- delta by-level no-hint.

## Perché comparivano risposte vuote
Causa: output structured LLM poteva risultare con `answer=""` senza essere trattato come anomalia fatale.

Correzione runtime:
- detect answer vuota;
- retry con guardrail esplicito;
- fallback testuale non-vuoto se persiste;
- tracciamento `answer_source`, `was_empty_before_guard`, `pipeline_errors`.

## Troubleshooting rapido
1. `sparse_enabled=False` o `fallback_dense`: verificare artifact sparse del run indice e `index_contract`.
2. no-hint bassa con retrieval alto: ispezionare `failure_category` e `reference_law_hit`.
3. regressione MCQ: ridurre bonus graph o tuning rerank su query specifiche.
4. benchmark lento: usare mini mode (`RUN_FULL_BENCHMARK=False`) e subset limit.
