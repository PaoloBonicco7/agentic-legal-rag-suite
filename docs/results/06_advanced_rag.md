# 06 — Advanced RAG

Run end-to-end della configurazione promossa dai diagnostics [06b](06b_retrieval_diagnostics.md):
**hybrid** (dense + sparse via RRF, `top_k=100`, `rrf_k=60`) + **multi-query** rewriting (`n=3`).
Graph expansion, LLM reranking e filtri metadata restano disponibili nel codice ma **disattivati**,
perché 06b non li ha promossi.

Run sorgente: `data/rag_runs/advanced/full_100__…__a4_combined_best_v2/`. Modello answer / judge /
rewriting: `SLURM.gpt-oss:120b`. Metodologia in [note/06](../notes/06_advanced_graph_rag_methodology.md).

## Metriche

| dataset | metrica | globale | L1 | L2 | L3 | L4 |
|---|---|---:|---:|---:|---:|---:|
| MCQ | accuracy | 0.84 | 0.88 | 0.88 | 0.76 | 0.84 |
| No-hint | mean_score (0–2) | 1.26 | 1.36 | 1.44 | 0.96 | 1.28 |
| No-hint | accuracy | 0.63 | 0.68 | 0.72 | 0.48 | 0.64 |

Confronto con gli altri metodi:

| | vs no-RAG | vs simple RAG |
|---|---:|---:|
| MCQ accuracy | +3pp (0.81 → 0.84) | +5pp (0.79 → 0.84) |
| No-hint mean | +0.29 (0.97 → 1.26) | +0.08 (1.18 → 1.26) |

L'advanced è l'unico metodo che batte il no-RAG anche sulle MCQ: hybrid + multi-query recuperano
l'articolo giusto abbastanza spesso da compensare il rumore che aveva penalizzato il simple RAG.

## Diagnostica della run

- **Failure category** (200 righe): `none` 147, `unknown` 43, `context_noise` 5,
  `generation_error` 4, `abstention` 1.
- **Context sufficiency** (no-hint, dichiarata dal modello): `yes` 86, `no` 14.
- **Copertura riferimenti**: la legge attesa è in contesto in 190/200 righe; l'articolo atteso in
  144/200 (186/200 se si guardano i candidati recuperati prima del taglio di contesto).
- **Multi-query**: 200 cache hit, 0 failure (le riformulazioni erano già in cache da 06b).
- Errori: 3 citazioni non valide sulle MCQ, 1 sul no-hint.

La categoria `unknown` resta la più grande dopo i successi: l'audit in
[note/06c](../notes/06c_advanced_unknown_audit.md) mostra che non sono fallimenti puri di
generazione, ma per lo più miss interni alla legge giusta (articolo/comma non recuperato). È la
stessa diagnosi che spiega il divario residuo verso l'oracle in [00_overview](00_overview.md).
