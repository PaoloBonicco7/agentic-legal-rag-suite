# 04 — No-RAG baseline

Baseline modello-solo: stesso benchmark e stesso judge degli altri step, ma senza alcun contesto
recuperato. Fissa il **floor** rispetto a cui si misura ogni guadagno del RAG.

Run sorgente: `data/baseline_runs/no_rag/`. Modello answer e judge: `SLURM.gpt-oss:120b` (Utopia),
`temperature=0`. Metodologia in [note/04](../notes/04_no_rag_baseline_methodology.md).

## Metriche

| dataset | metrica | globale | L1 | L2 | L3 | L4 |
|---|---|---:|---:|---:|---:|---:|
| MCQ | accuracy | 0.81 | 0.80 | 0.84 | 0.76 | 0.84 |
| No-hint | mean_score (0–2) | 0.97 | 0.68 | 1.08 | 0.92 | 1.20 |
| No-hint | accuracy | 0.485 | 0.34 | 0.54 | 0.46 | 0.60 |

Nessun errore di generazione o judge (coverage 1.0 su entrambi i dataset).

## Lettura

Il modello risponde bene alle MCQ (0.81): con le opzioni davanti, la conoscenza parametrica basta
spesso a riconoscere quella giusta. Sulle domande aperte invece crolla (mean 0.97 su 2): senza il
testo della legge non ricostruisce il dettaglio normativo regionale. È esattamente il divario che il
retrieval deve colmare — confronto in [00_overview](00_overview.md).
