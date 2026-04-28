# Glossario
Status: Roadmap (target)  
Scope: terminology  
Source of truth: `docs/requirements.md`

- `law_id`: ID canonico della legge (deterministico, stabile).
- `article_label_norm`: label normalizzata dell’articolo (es. `Art. 12`).
- `passage`: segmento fine dentro un articolo (intro/comma/lettera).
- `passage_label`: etichetta del passage (es. `c1`, `c1.lit_a`).
- `chunk`: unita' indicizzabile (embedding + retrieval).
- `chunk_id`: identificatore del chunk, usato anche come “citation” nel runtime.
- `evidence set`: lista compatta di chunk/snippet usati come contesto dall’LLM.
- `hybrid retrieval`: combinazione di retrieval lessicale (BM25) e semantico (vector).
- `rerank`: riordinamento dei candidati via modello di reranking su top-N.
- `query rewrite`: riscrittura controllata della query per aumentare recall.
- `self-eval`: valutazione automatica di sufficienza/faithfulness del contesto e output, con azioni suggerite.
- `trace_id`: identificatore di una run o di una domanda, per audit e riproducibilita'.

