# 06b — Retrieval diagnostics (risultati)

Esperimenti retrieval-only usati per scegliere la configurazione da promuovere in Advanced RAG. Lo
step isola la qualità del retrieval dalla generazione: misura se i riferimenti legali attesi
compaiono tra i candidati, in quale posizione, e quali leve meritano una run end-to-end. Una leva è
**promossa** solo se migliora `article_hit_pct` di almeno +1pp su 100 domande, con configurazione
riproducibile dal manifest.

Run di riferimento: `data/retrieval_eval_runs/diagnostic_full_utopia_throttled__20260525T091921Z/`
(profilo `full`, schema `retrieval-evaluation-v4`). Indice `legal_chunks_bge_m3` (76.467 chunk,
BGE-M3 dense 1024 + sparse nativo). Metriche: `article_hit_pct` (primaria), `law_hit_pct`,
`article_mrr`. Definizioni nella [spec 06b](../specs/06b_retrieval_diagnostics.md); metodo in
[note/06b](../notes/06b_retrieval_diagnostics_methodology.md).

I due dataset (`mcq`, `no_hint`) condividono lo stesso stem nelle chiamate retrieval, quindi le
metriche coincidono row-per-row: le tabelle sotto valgono per entrambi salvo dove indicato.

## Re-index: vecchio embedder vs BGE-M3

Prima motivazione della pipeline: il passaggio dal vecchio indice dense-only (`legal_chunks`,
Utopia/Nomic 768-dim) al nuovo `legal_chunks_bge_m3` (BGE-M3, dense 1024 + sparse). Stesso corpus,
stesse domande.

| indice | top_k | rrf_k | article_hit | law_hit | article_mrr |
|---|---:|---:|---:|---:|---:|
| Utopia/Nomic dense | 10 | – | 45.0% | 72.0% | 0.288 |
| Utopia/Nomic dense | 100 | – | 66.0% | 87.0% | 0.295 |
| BGE-M3 dense | 10 | – | 73.0% | 97.0% | 0.519 |
| BGE-M3 dense | 100 | – | 88.0% | 99.0% | 0.524 |
| BGE-M3 hybrid | 100 | 30 | 89.0% | 99.0% | 0.551 |

A parità di dense@10, BGE-M3 guadagna **+28pp** di `article_hit` (45 → 73) e **+0.231** di MRR. Il
re-index, da solo, è il salto più grande della fase. Fonti: `sweep_direct.csv` dei due run
(`default__20260511T180530Z` per il vecchio indice).

## Esperimenti

### A — Dense baseline (curva top-k)

| top_k | article_hit | law_hit | MRR |
|---:|---:|---:|---:|
| 10 (baseline) | 73.0% | 97.0% | 0.519 |
| 20 | 77.0% | 97.0% | 0.521 |
| 50 | 84.0% | 97.0% | 0.523 |
| 100 | 88.0% | 99.0% | 0.524 |

Alzare `top_k` migliora il recall in modo monotono (+15pp da 10 a 100) ma l'MRR resta piatto: il
problema non è solo recall, è **ranking**. `dense@10` è la baseline dei delta, `dense@100` il recall
ceiling senza riformulazioni.

### B — Filtri metadata · scartato

`law_status=current` esclude 4 domande i cui riferimenti puntano a leggi abrogate e non alza il hit
(−2pp `article_hit`). Il corpus Valle d'Aosta è già dominato da leggi correnti: il filtro toglie
segnale invece di aggiungerlo (`promote_filter=false`).

### C — Graph expansion · scartato

Seguendo gli edge del grafo legale dai seed dense il guadagno è marginale (+1pp `article_hit`, MRR
invariato) e si paga con `expansion_noise_ratio = 99.7%`: oltre il 99% dei chunk aggiunti è
irrilevante. Nessuna configurazione del sweep rispetta un noise cap ragionevole
(`graph_expansion_enabled=false`); la leva resta nel codice, riattivabile con un cap più severo.

### F — Hybrid retrieval (dense + sparse via RRF) · promosso

| top_k | dense article_hit | hybrid best | δ |
|---:|---:|---:|---:|
| 10 | 73.0% | 76.0% | +3pp |
| 20 | 77.0% | 78.0% | +1pp |
| 50 | 84.0% | 82.0% | −2pp |
| 100 | 88.0% | 89.0% | +1pp |

Punto operativo scelto: `top_k=100`, `rrf_k=30` → **89.0%** `article_hit`, MRR 0.551. Rispetto a
`dense@10` sono **+16pp** di `article_hit` e +0.032 di MRR. `rrf_k` basso premia i primi rank,
coerente col dominio legale dove i primi candidati pesano di più (a `top_k=100` il hit è 89% per
`rrf_k` 30/60/90; cambia solo l'MRR di terzo decimale).

### G — LLM reranking · scartato

Pilot di 30 domande/dataset, 18 configurazioni (`input_k × output_k`). Risultato netto: su tutte le
configurazioni `recovered=0` — il reranker non recupera mai un articolo che l'hybrid aveva escluso,
può solo riordinare ciò che è già presente — e in molti casi **demota** articoli corretti. Impatto
vs base hybrid: **−6.4pp** su `mcq`, −1.5pp su `no_hint`. L'MRR locale migliora (0.71 vs 0.55), ma è
precision-at-top che non compensa la perdita di recall (`rerank_enabled=false`). Il `failure_rate`
LLM del pilot è 30.6%.

### H — Query rewriting · promosso (`multi_query`)

Pilot di 30 domande/dataset, `failure_rate=0%`. Strategie confrontate sopra l'hybrid base (`mcq`):

| strategia | article_hit | law_hit | MRR | δ vs none |
|---|---:|---:|---:|---:|
| none | 83.3% | 96.7% | 0.493 | — |
| rewrite | 90.0% | 96.7% | 0.525 | +6.7pp |
| hyde | 86.7% | 96.7% | 0.471 | +3.3pp |
| **multi_query** | **93.3%** | **100.0%** | 0.467 | **+10.0pp** |

`multi_query` (n=3) è la migliore su entrambi i dataset: +4.3pp sopra l'hybrid full, `law_hit` al
100% sul pilot. La diversità delle riformulazioni copre meglio il vocabolario normativo della stessa
intent. `rewrite` resta un fallback più economico (una sola query).

## Waterfall

| stage | scenario | article_hit | δ vs dense@10 | esito |
|---|---|---:|---:|---|
| baseline | dense@10 | 73.0% | — | reference |
| ceiling | dense@100 | 88.0% | +15pp | informativo |
| filter | law_status=current | 71.0% | −2pp | scartato |
| graph | best graph | 74.0% | +1pp | scartato (noise 99.7%) |
| **hybrid** | **top_k=100, rrf_k=30** | **89.0%** | **+16pp** | **promosso** |
| rerank | input 20 / output 10 (pilot) | 82.6% | +9.6pp | scartato (−6.4pp vs hybrid) |
| **rewriting** | **multi_query, n=3 (pilot)** | **93.3%** | **+20.3pp** | **promosso (+4.3pp vs hybrid)** |

La fase promuove **due** leve: hybrid (F) come retrieval di base e multi-query (H) come
trasformazione della richiesta.

## Configurazione raccomandata

```json
{
  "hybrid_enabled": true,
  "top_k": 100,
  "rrf_k": 30,
  "metadata_filters_enabled": false,
  "graph_expansion_enabled": false,
  "rerank_enabled": false,
  "query_rewriting_enabled": true,
  "query_rewriting_strategy": "multi_query",
  "query_rewriting_n": 3
}
```

`AdvancedRagConfig` espone tutti questi campi e la run end-to-end li applica: vedi
[06 — advanced RAG](06_advanced_rag.md). L'unica differenza è `rrf_k=60` nella run advanced invece
di 30; è ininfluente sul recall (a `top_k=100` l'`article_hit` è 89% per entrambi). I parametri di
graph e rerank restano nel config come default disattivati, riattivabili con un solo flag.

## Perché questa configurazione

- **Il dominio legale favorisce l'hybrid.** Le domande contengono marker lessicali (numeri di
  articolo, sigle, riferimenti puntuali) che il dense BGE-M3 cattura male da solo; il vettore sparse
  li indicizza esplicitamente. Il +16pp di F non è cosmetico.
- **La fusione migliora anche il ranking, non solo il recall.** L'MRR sale da 0.519 (dense@10) a
  0.551 (hybrid@100), oltre il dense@100 (0.524): la RRF porta in cima i chunk che entrambe le viste
  ritengono rilevanti.
- **Multi-query aggiunge un guadagno indipendente** (+4.3pp sopra l'hybrid), al costo di una
  chiamata LLM per domanda con cache versionata.
- **Le leve scartate hanno costi senza beneficio retrieval-only**: rerank perde recall, graph
  aggiunge ~99.7% di rumore, i filtri droppano domande valide. Promuovere solo le due leve con
  guadagno netto è coerente col principio _simplicity first_ del progetto.

## Riferimenti

- Run dir: `data/retrieval_eval_runs/diagnostic_full_utopia_throttled__20260525T091921Z/`
  (`scenarios.csv`, `sweep_direct.csv`, `sweep_rerank.csv`, `sweep_query_rewriting.csv`,
  `manifest.json`, `recommended_advanced_config.json`).
- Spec: [docs/specs/06b_retrieval_diagnostics.md](../specs/06b_retrieval_diagnostics.md) · Metodo:
  [docs/notes/06b_retrieval_diagnostics_methodology.md](../notes/06b_retrieval_diagnostics_methodology.md).
- Notebook: `notebooks/06b_retrieval_diagnostics.ipynb`.
