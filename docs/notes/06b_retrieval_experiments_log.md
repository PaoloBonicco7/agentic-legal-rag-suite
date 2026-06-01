# 06b - Retrieval Experiments Log

Registro delle run retrieval-only usate per decidere quali configurazioni promuovere in Advanced Graph RAG.

## Run `default__20260511T180530Z`

- **Data run**: 2026-05-11
- **Origine metriche**: `data/retrieval_eval_runs/default__20260511T180530Z/scenarios.csv`
- **Dataset**: `mcq`, `no_hint`
- **Stato**: baseline retroattiva registrata durante la Fase 0

### Metriche principali

| scenario | dataset | article_hit_pct | law_hit_pct | article_mrr | n_questions | status | note |
|---|---|---:|---:|---:|---:|---|---|
| Baseline dense@10 (no filter) | mcq | 45.0 | 72.0 | 0.287734126984127 | 100 | run | baseline |
| Best dense top-k (no filter) | mcq | 66.0 | 87.0 | 0.2945612756503889 | 100 | run | recall ceiling observed at top_k=100 |
| Best direct budget top20 | mcq | 48.0 | 77.0 | 0.2900033577533577 | 100 | run | moderate budget check |
| + Filter law_status=current | mcq | 45.0 | 72.0 | 0.2878730158730159 | 100 | run | 4 filtered questions excluded |
| + Best filter from sweep | mcq | 45.0 | 72.0 | 0.287734126984127 | 100 | run | no improvement over baseline |
| + Graph default seed=3 | mcq | 45.0 | 74.0 | 0.287734126984127 | 100 | run | law hit improves, article hit unchanged |
| + Graph REFERENCES only | mcq | 45.0 | 74.0 | 0.287734126984127 | 100 | run | article hit unchanged |
| + Best graph from sweep | mcq | 66.0 | 88.0 | 0.2945612756503889 | 100 | run | driven by top_k=100 and high expansion noise |
| + Best graph low-noise skipped | mcq | n/a | n/a | n/a | 0 | skipped | no graph configuration with expansion_noise_ratio <= 0.95 |
| Hybrid retrieval skipped | mcq | n/a | n/a | n/a | 0 | skipped | dense-only index, sparse vector unavailable |
| Baseline dense@10 (no filter) | no_hint | 45.0 | 72.0 | 0.287734126984127 | 100 | run | baseline |
| Best dense top-k (no filter) | no_hint | 66.0 | 87.0 | 0.2945612756503889 | 100 | run | recall ceiling observed at top_k=100 |
| Best direct budget top20 | no_hint | 48.0 | 77.0 | 0.2900033577533577 | 100 | run | moderate budget check |
| + Filter law_status=current | no_hint | 45.0 | 72.0 | 0.2878730158730159 | 100 | run | 4 filtered questions excluded |
| + Best filter from sweep | no_hint | 45.0 | 72.0 | 0.287734126984127 | 100 | run | no improvement over baseline |
| + Graph default seed=3 | no_hint | 45.0 | 74.0 | 0.287734126984127 | 100 | run | law hit improves, article hit unchanged |
| + Graph REFERENCES only | no_hint | 45.0 | 74.0 | 0.287734126984127 | 100 | run | article hit unchanged |
| + Best graph from sweep | no_hint | 66.0 | 88.0 | 0.2945612756503889 | 100 | run | driven by top_k=100 and high expansion noise |
| + Best graph low-noise skipped | no_hint | n/a | n/a | n/a | 0 | skipped | no graph configuration with expansion_noise_ratio <= 0.95 |
| Hybrid retrieval skipped | no_hint | n/a | n/a | n/a | 0 | skipped | dense-only index, sparse vector unavailable |

### Sintesi

La baseline dense@10 recupera l'articolo corretto nel 45.0% delle domande e almeno la legge corretta nel 72.0%. Aumentare il budget dense fino a top_k=100 porta `article_hit_pct` al 66.0% e `law_hit_pct` all'87.0%, ma `article_mrr` resta vicino a 0.295: il problema non è solo recall, ma anche ranking.

I filtri metadata non migliorano la baseline. La graph expansion aumenta lievemente `law_hit_pct`, ma non migliora `article_hit_pct` nelle configurazioni non degeneri. La miglior configurazione graph replica il top_k=100 e ha `expansion_noise_ratio` alto, quindi non è una leva da promuovere senza ulteriore controllo.

Hybrid retrieval è stato saltato perché l'indice usato dalla run è dense-only. La prossima fase deve costruire una collection parallela con dense e sparse vectors.

### Prossimo passo

Fase 1: re-index BGE-M3 in `legal_chunks_bge_m3`, verificando sparse vector e smoke test dense/hybrid.

## bge_m3_reindex_setup__20260512 — setup notebook monitorabile

- **Data**: 2026-05-12
- **Run dir**: pending manual notebook run
- **Cambio rispetto al precedente**: preparata la full indexing BGE-M3 in collection parallela senza sovrascrivere `legal_chunks`.
- **Configurazione chiave**: embedding=BAAI/bge-m3, retrieval_mode=hybrid-ready, top_k=n/a, rerank=off, query_rewriting=none
- **Metriche** (MCQ / no_hint):
  - article_hit@n/a: n/a / n/a
  - law_hit@n/a: n/a / n/a
  - MRR: n/a / n/a
- **Costo LLM**: 0 chiamate, 0% cache hit
- **Conclusione**: la fase runtime non e ancora eseguita; il notebook 03 ora contiene preflight, run protetta, progress log JSONL e verifiche post-run per `legal_chunks_bge_m3`.
- **Decisione successiva**: lanciare la cella full run nel notebook, poi registrare manifest, count, schema dense/sparse e smoke retrieval.

## bge_m3_full_20260512_212818 — indice BGE-M3 disponibile

- **Data**: 2026-05-13
- **Run dir**: data/indexing_runs/20260512_212818
- **Cambio rispetto al precedente**: costruita collection Qdrant locale parallela con dense BGE-M3 e sparse vectors native.
- **Configurazione chiave**: embedding=BAAI/bge-m3, retrieval_mode=hybrid-ready, top_k=n/a, rerank=off, query_rewriting=none
- **Metriche** (MCQ / no_hint):
  - article_hit@n/a: n/a / n/a
  - article_hit@5 (post-rerank, se applicabile): n/a / n/a
  - law_hit@n/a: n/a / n/a
  - MRR: n/a / n/a
- **Costo LLM**: 0 chiamate, 0% cache_hit_rate
- **Conclusione**: indice pronto per retrieval diagnostics; manifest `ready_for_retrieval=true`, `selected_count=indexed_count=collection_points_count=76467`, `failure_count=0`, dense vector size 1024 e sparse vector `sparse` presenti.
- **Decisione successiva**: eseguire Fase 2 puntando `notebooks/06b_retrieval_diagnostics.ipynb` a `data/indexing_runs/20260512_212818/index_manifest.json`.

## rerank_integration__20260513 — integrazione Esperimento G

- **Data**: 2026-05-13
- **Run dir**: non prodotto; integrazione codice/notebook senza chiamate Utopia
- **Cambio rispetto al precedente**: aggiunto rerank LLM cache-aware e versionato allo step 06b, eseguibile con `RUN_RERANK=True`.
- **Configurazione chiave**: embedding=BAAI/bge-m3, retrieval_mode=hybrid, top_k={20,50,100}, rerank=on via flag, query_rewriting=none
- **Metriche** (MCQ / no_hint):
  - article_hit@k: n/a / n/a
  - article_hit@5 (post-rerank, se applicabile): n/a / n/a
  - law_hit@k: n/a / n/a
  - MRR: n/a / n/a
- **Costo LLM**: 0 chiamate, n/a cache_hit_rate
- **Conclusione**: la Fase 3 e integrata ma non eseguita; il notebook ora produce `sweep_rerank.csv`, aggiorna la waterfall con `+ LLM rerank`, usa `RERANK_PROMPT_VERSION` nella cache e registra cache/failure nel manifest.
- **Decisione successiva**: completare prima lo sweep hybrid F se mancano metriche reali, poi attivare `RUN_RERANK=True` per il pilot G e rilanciare una seconda volta per verificare cache hit circa 100%.

## query_rewriting_integration__20260513 — integrazione Esperimento H

- **Data**: 2026-05-13
- **Run dir**: non prodotto; integrazione codice/notebook senza chiamate Utopia
- **Cambio rispetto al precedente**: aggiunto supporto cache-aware e versionato per query rewriting, HyDE e multi-query nello step 06b.
- **Configurazione chiave**: embedding=BAAI/bge-m3, retrieval_mode=hybrid, top_k=best hybrid, rerank=off via default, query_rewriting={none,rewrite,hyde,multi_query}
- **Metriche** (MCQ / no_hint):
  - article_hit@k: n/a / n/a
  - article_hit@5 (post-rerank, se applicabile): n/a / n/a
  - law_hit@k: n/a / n/a
  - MRR: n/a / n/a
- **Costo LLM**: 0 chiamate, n/a cache_hit_rate
- **Conclusione**: la Fase 4 e integrata ma non eseguita; il notebook ora puo produrre `sweep_query_rewriting.csv`, mostrare esempi qualitativi e aggiungere `+ Query rewriting` alla waterfall quando `RUN_QUERY_REWRITING=True`.
- **Decisione successiva**: eseguire il pilot H dopo aver fissato il best hybrid e, se utile, dopo il pilot G per confrontare il winner con rerank attivo.
