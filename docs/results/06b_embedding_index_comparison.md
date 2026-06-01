# 06b - Confronto indici embedding

Questo report riassume i dati gia presenti nel repository per giustificare il passaggio dal vecchio indice dense-only basato su Utopia/Nomic al nuovo indice BGE-M3 dense+sparse, che abilita hybrid search.

## Stato implementazione

- Il confronto retrieval-only e implementato nello step `06b_retrieval_diagnostics`.
- Il notebook `05_simple_rag.ipynb` usa gia il nuovo indice `legal_chunks_bge_m3`, ma non confronta due indici: lo step 05 resta un baseline RAG dense-only, senza hybrid e senza sweep tra embedder.
- Il confronto tra indici va quindi citato da `06b`; lo step 05 puo essere usato solo come evidenza end-to-end che il RAG con il nuovo indice migliora le risposte open-ended rispetto al no-RAG.

## Indici confrontati

| indice | collection | embedder | vettori | chunk | manifest |
|---|---|---|---|---:|---|
| vecchio | `legal_chunks` | `SLURM.nomic-embed-text:latest` via Utopia | dense 768, no sparse | 76.467 | `data/indexing_runs/20260511_172154/index_manifest.json` |
| nuovo | `legal_chunks_bge_m3` | `BAAI/bge-m3` locale | dense 1024 + sparse nativo | 76.467 | `data/indexing_runs/bge_m3_full_20260523_095655/index_manifest.json` |

## Retrieval-only: vecchio indice vs nuovo indice

Fonti:

- Vecchio indice: `data/retrieval_eval_runs/default__20260511T180530Z/sweep_direct.csv`
- Nuovo indice: `data/retrieval_eval_runs/diagnostic_full_utopia_throttled__20260525T091921Z/sweep_direct.csv`

I valori sotto sono calcolati sul dataset `mcq` da 100 domande. Nei run diagnostici le metriche dense/hybrid coincidono anche su `no_hint`, perche il retrieval usa lo stesso insieme di domande/riferimenti attesi.

| scenario | top_k | rrf_k | article_hit | law_hit | article_mrr |
|---|---:|---:|---:|---:|---:|
| Utopia/Nomic dense | 10 | - | 45.0% | 72.0% | 0.288 |
| Utopia/Nomic dense | 100 | - | 66.0% | 87.0% | 0.295 |
| BGE-M3 dense | 10 | - | 73.0% | 97.0% | 0.519 |
| BGE-M3 dense | 100 | - | 88.0% | 99.0% | 0.524 |
| BGE-M3 hybrid | 10 | 30 | 76.0% | 96.0% | 0.560 |
| BGE-M3 hybrid | 100 | 30 | 89.0% | 99.0% | 0.551 |

Delta principali:

- A parita di dense@10, BGE-M3 migliora `article_hit` di +28.0pp (45.0% -> 73.0%), `law_hit` di +25.0pp (72.0% -> 97.0%) e `article_mrr` di +0.231.
- A parita di dense@100, BGE-M3 migliora `article_hit` di +22.0pp (66.0% -> 88.0%), `law_hit` di +12.0pp (87.0% -> 99.0%) e `article_mrr` di +0.229.
- Hybrid aggiunge segnale sopra BGE-M3 dense: a `top_k=10`, `article_hit` sale da 73.0% a 76.0% e `article_mrr` da 0.519 a 0.560; il best operativo `hybrid@100 rrf_k=30` arriva a 89.0% `article_hit`.
- Rispetto al baseline BGE-M3 dense@10 usato nel report 06b, il best hybrid guadagna +16.0pp di `article_hit` (73.0% -> 89.0%) e +0.032 di `article_mrr` (0.519 -> 0.551).

## Evidenza end-to-end da Simple RAG

Fonti:

- No-RAG: `data/baseline_runs/no_rag/no_rag_summary.json`
- Simple RAG: `data/rag_runs/simple/simple_rag_summary.json`
- Manifest Simple RAG: `data/rag_runs/simple/simple_rag_manifest.json`

Il run Simple RAG usa `legal_chunks_bge_m3`, `top_k=3`, `max_context_chunks=3`, filtro statico `law_status=current`, chat model `SLURM.gpt-oss:120b` e judge model `SLURM.gpt-oss:120b`.

| dataset | no-RAG | Simple RAG BGE-M3 | delta |
|---|---:|---:|---:|
| `mcq` accuracy | 81.0% | 79.0% | -2.0pp |
| `no_hint` accuracy | 48.5% | 59.0% | +10.5pp |
| `no_hint` score | 97/200 | 118/200 | +21 punti |

L'evidenza end-to-end e quindi utile soprattutto per le risposte open-ended: il contesto recuperato dal nuovo indice aumenta lo score no-hint da 97/200 a 118/200. Non e pero un confronto tra embedder, perche manca una run Simple RAG equivalente sul vecchio indice `legal_chunks`.

## Conclusione

La giustificazione piu solida per il nuovo embedder e retrieval-only:

1. BGE-M3 dense batte nettamente il vecchio Utopia/Nomic dense-only sullo stesso corpus e sullo stesso set di domande.
2. BGE-M3 produce anche il vettore sparse, quindi rende possibile hybrid search senza cambiare corpus.
3. Hybrid search migliora ulteriormente il ranking e il recall rispetto al baseline BGE-M3 dense@10.
4. Simple RAG conferma che, con il nuovo indice, il contesto recuperato migliora l'open-answer end-to-end rispetto al no-RAG; non dimostra da solo la superiorita dell'embedder.

Se serve una prova end-to-end specifica "embedder vecchio vs embedder nuovo", bisogna eseguire Simple RAG due volte con la stessa configurazione LLM e due soli cambiamenti: `collection_name` e `index_manifest_path`.
