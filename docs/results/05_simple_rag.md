# 05 — Simple RAG

Baseline RAG minimale: dense retrieval (BGE-M3, named vector `dense`), nessun hybrid, graph o
rerank. Serve a verificare che la catena retrieval → contesto → risposta → citazione → giudizio
funzioni end-to-end, e a misurare il primo guadagno del contesto sul no-RAG.

Run sorgente: `data/rag_runs/simple/`. Config effettiva: collection `legal_chunks_bge_m3`,
`top_k=3`, `max_context_chunks=3`, `max_context_chars=8000`, filtro statico `law_status=current`,
modello answer/judge `SLURM.gpt-oss:120b`.

## Metriche

| dataset | metrica | globale | L1 | L2 | L3 | L4 |
|---|---|---:|---:|---:|---:|---:|
| MCQ | accuracy | 0.79 | 0.84 | 0.80 | 0.80 | 0.72 |
| No-hint | mean_score (0–2) | 1.18 | 1.32 | 1.36 | 1.28 | 0.76 |
| No-hint | accuracy | 0.59 | 0.66 | 0.68 | 0.64 | 0.38 |

Errori: 1 citazione non valida sulle MCQ, 0 sul no-hint.

Rispetto al no-RAG: **MCQ −2pp** (0.81 → 0.79), **no-hint mean +0.21** (0.97 → 1.18).

## Lettura

Il dense top-3 conferma il pattern che guiderà lo step 06: aiuta molto le domande aperte (+0.21 di
`mean_score`) ma penalizza leggermente le MCQ, dove un contesto parziale può distrarre il modello da
una risposta che conosceva. Il punto debole è L4 sul no-hint (0.76): sono domande applicative, in
cui top-3 chunk dense spesso non bastano a coprire l'articolo giusto.

I fallimenti no-hint a score 0 sono in larga parte retrieval miss espliciti — il modello dichiara
"il contesto non contiene la risposta" (es. eval-0007 sui contributi veicoli, eval-0021 sul GAP) —
oppure casi in cui il chunk recuperato è l'intro di una lista senza le voci (es. eval-0001 sugli
organi USL). È la motivazione retrieval-only per passare a hybrid + multi-query, misurata in
[06b](06b_retrieval_diagnostics.md). Le tracce complete per riga sono in
`data/rag_runs/simple/mcq_results.jsonl` e `no_hint_results.jsonl`.
