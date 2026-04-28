# Roadmap
Status: Roadmap (target)  
Scope: milestones and done criteria  
Source of truth: `docs/requirements.md`

Questa roadmap descrive il sistema target in milestone incrementali. Ogni step deve essere misurabile e confrontabile con la baseline.

## Milestone 0 - Fondamenta
Done criteria:
- Dataset MCQ leggibile e parsing robusto (`evaluation/questions.csv`).
- Ingestion deterministica HTML->dataset con QA/manifest (vedi `laws_ingestion/README.md`).
- Tracciabilita' base: ogni run ha `trace_id` e artefatti minimi.

Implementato oggi (indicativo):
- `laws_ingestion/` include pipeline deterministica e output JSONL.

## Milestone 1 - Baseline RAG naive (Qdrant)
Done criteria:
- Costruzione index Qdrant per `chunks`.
- Retrieval top-k vettoriale e answer MCQ con output strutturato.
- Report metriche baseline: accuracy + recall@k/hit-rate@k.

Implementato oggi (indicativo):
- Storico: baseline naive con LangGraph in `src/legal_rag_pipeline/` (rimossa nel refactoring corrente).
- Corrente: indexing Qdrant in `src/legal_indexing/` + `notebooks/04_qdrant_indexing_pipeline.ipynb`.

## Milestone 2 - Chunking/metadata-aware
Done criteria:
- Varianti `plain` vs `with_metadata` confrontabili.
- Miglioramento misurabile su retrieval o citazioni (anche solo interpretabilita').

## Milestone 3 - Hybrid retrieval (BM25 + vector)
Done criteria:
- BM25 index costruibile sul dataset e combinazione con Qdrant.
- Merge deterministico, dedup e policy documentata.
- Miglioramento su recall@k (atteso in query con riferimenti espliciti).

## Milestone 4 - Reranking + noise reduction
Done criteria:
- Recupero largo + reranker + top-n finale.
- Riduzione rumore (chunk ridondanti) e miglioramento precisione contesto (proxy: hit-rate@k a k piccolo, o metriche qualitative).

## Milestone 5 - Query rewrite controllato
Done criteria:
- Rewrite in 1 passaggio con logging della trasformazione.
- Aumento recall@k su subset di domande “difficili” (lessicalmente distanti).

## Milestone 6 - Agentic retrieval loop (self-eval -> action -> retry)
Done criteria:
- State machine (LangGraph) con max iterazioni, stop conditions e azioni (rewrite/increase_k/rerank).
- Log per step e confronto A/B col baseline.
- Riduzione errori da “retrieval miss” nelle domande hard.

## Milestone 7 (opzionale) - Graph expansion
Done criteria:
- Uso controllato di edges (REFERS_TO/AMENDED_BY/ABROGATED_BY) per espansione contesto quando self-eval segnala missing evidence.
- Evidenza empirica su casi multi-hop.
