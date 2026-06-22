# Risultati — quadro d'insieme

Confronto dei tre approcci centrali della tesi sullo stesso benchmark: **no-RAG** (modello senza
contesto), **simple RAG** (dense retrieval, top-3) e **advanced RAG** (hybrid + multi-query). Come
riferimento superiore è incluso l'**oracle context** (step 02b), dove il modello riceve gli articoli
di legge corretti: misura quanto aiuta il contesto giusto, a prescindere dal retrieval.

Benchmark: 100 domande MCQ + 100 domande no-hint (aperte), bilanciate su quattro livelli di
difficoltà L1–L4 (25 ciascuno). Le MCQ sono valutate per match dell'opzione corretta; le no-hint da
un judge LLM con rubric 0–2 (`mean_score` = media sul punteggio 0–2; accuracy = punteggio totale
sul massimo possibile). Tutte le run usano lo stesso modello `SLURM.gpt-oss:120b` su Utopia e
l'indice `legal_chunks_bge_m3` (BGE-M3, dense + sparse).

## Tabella headline

| Metodo | MCQ accuracy | No-hint mean (0–2) | No-hint accuracy |
|---|---:|---:|---:|
| No-RAG (floor) | 0.81 | 0.97 | 0.485 |
| Simple RAG (dense, top-3) | 0.79 | 1.18 | 0.59 |
| Advanced RAG (hybrid + multi-query) | 0.84 | 1.26 | 0.63 |
| Oracle context (ceiling) | 0.99 | 1.78 | 0.89 |

Fonti: `data/baseline_runs/no_rag/`, `data/rag_runs/simple/`,
`data/rag_runs/advanced/…__a4_combined_best_v2/`, `data/evaluation_runs/oracle_context/`.

## Lettura

- **No-hint: il retrieval aiuta, e un retrieval migliore aiuta di più.** Il `mean_score` cresce in
  modo monotono (0.97 → 1.18 → 1.26). Sulle domande aperte il contesto recuperato è la leva
  principale, perché il modello da solo non conosce il dettaglio della normativa regionale.
- **MCQ: il simple RAG peggiora, l'advanced recupera.** Sulle multiple-choice il dense top-3 scende
  sotto il no-RAG (0.81 → 0.79): un contesto rumoroso distrae dalla conoscenza parametrica del
  modello. Hybrid + multi-query inverte il segno (0.84, +3pp sul no-RAG) perché porta l'articolo
  corretto in contesto più spesso.
- **Il collo di bottiglia è il retrieval, non il modello.** Con il contesto corretto (oracle) il
  modello arriva a 0.99 sulle MCQ e 1.78 / 0.89 sul no-hint. Il divario tra advanced e oracle
  (1.26 vs 1.78 sul no-hint) misura quanto si perde ancora per articoli non recuperati — coerente
  con la diagnosi di [06b](06b_retrieval_diagnostics.md) e con l'audit dei casi `unknown`
  ([note/06c](../notes/06c_advanced_unknown_audit.md)), dominati dal pattern "legge giusta,
  articolo sbagliato".

## Verdetto retrieval

I diagnostics retrieval-only ([06b](06b_retrieval_diagnostics.md)) promuovono **due** leve: hybrid
(dense + sparse via RRF) e **multi-query** rewriting. Graph expansion, LLM reranking e filtri
metadata sono stati testati e **scartati** (rispettivamente: rumore quasi totale nei candidati,
perdita di recall, esclusione di domande valide). La pipeline advanced effettivamente eseguita
riflette questa scelta: hybrid `top_k=100` + multi-query `n=3`, con graph / rerank / filtri
disattivati.

## Dettaglio per step

- [01 — preprocessing](01_laws_preprocessing.md) · [02 — evaluation dataset](02_evaluation_dataset.md) · [03 — indicizzazione](03_indexing_contract.md)
- [04 — no-RAG](04_no_rag.md) · [05 — simple RAG](05_simple_rag.md) · [06 — advanced RAG](06_advanced_rag.md) · [06b — retrieval diagnostics](06b_retrieval_diagnostics.md)

> Lo step 07 (reporting cross-method automatico) è definito nella spec ma non ancora prodotto come
> artifact: questa pagina è la sintesi manuale equivalente, derivata dai summary delle run.
