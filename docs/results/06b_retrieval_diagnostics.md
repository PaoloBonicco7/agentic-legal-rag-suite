# 06b — Retrieval diagnostics e audit dei filtri (risultati)

Il documento separa due lineage:

- il baseline storico v1, usato per scegliere hybrid e multi-query nella run Advanced RAG;
- l'audit di vigenza v2, che rivaluta esclusivamente i filtri senza promuoverli automaticamente.

Definizioni nella [spec 06b](../specs/06b_retrieval_diagnostics.md); metodo in
[note/06b](../notes/06b_retrieval_diagnostics_methodology.md).

## Baseline storico v1

Run: `data/retrieval_eval_runs/diagnostic_full_utopia_throttled__20260525T091921Z/`
(`retrieval-evaluation-v4`). Indice `legal_chunks_bge_m3`, 76.467 chunk, BGE-M3 dense 1024 +
sparse. Le tabelle di questa sezione sono evidenza storica e non descrivono il nuovo artefatto
status-v2.

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

### B — Filtri metadata v1 · non promossi

Il confronto Dense@10 storico su 100 domande era:

| filtro | Article Success |
|---|---:|
| none | 73 |
| `law_status=current` | 71 |
| `article_status=current` | 69 |
| vista corrente legge+articolo | 67 |

Il peggioramento era reale, ma la precedente spiegazione causale era errata. Il corpus v2 non è
dominato da leggi correnti: soltanto 836 su 3.145 sono `current`. Le esclusioni mescolavano quattro
fenomeni distinti:

- sovraclassificazione `past` di articoli con soli commi o lettere cessati;
- riferimenti realmente storici (`eval-0013`–`eval-0016`);
- due qrel non allineati al passaggio che supporta la risposta (`eval-0013`, `eval-0076`);
- perdita del retriever e, potenzialmente, approssimazione ANN.

Per questo `metadata_filters_enabled=false` resta la scelta storica conservativa, ma il risultato v1
non dimostra che ogni filtro v2 sia inutilizzabile.

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

## Waterfall storico

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

## Configurazione storicamente raccomandata

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
- **Le leve scartate hanno costi senza beneficio retrieval-only nel run v1**: rerank perde recall,
  graph aggiunge ~99.7% di rumore e i filtri v1 riducono l'Article Success. La natura delle
  esclusioni dei filtri è riesaminata separatamente nell'audit v2.

## Audit di vigenza v2

L'audit usa `retrieval-evaluation-v5`, `filter-audit-v1` e prompt
`none-v1`. Non esegue LLM, reranking, graph expansion o query rewriting. L'indice isolato è
`legal_chunks_bge_m3_status_v2`, costruito in Qdrant locale da 76.499 chunk con
`laws-preprocessing-v2` e `legal-status-rules-v1`.

La provenienza clean verificata è:

- corpus SHA-256:
  `aa46ea3758c1de5596a8902b95e90cdfc6d08fbc2956086495a020680be22459`;
- clean manifest SHA-256:
  `65eb3cdca461a2a8d894959be9df3a34f3e03cc975d6df08c40505111da51b23`;
- chunk SHA-256:
  `dacbb03fd3debfd272031226ad539100cc228141acca94fba304fa28c663cb95`;
- eventi SHA-256:
  `ee740ae7b403a2bf6f4b666e59e6b243f493963ea047436bcead322fd4973917`.

Il preflight e il postflight ricalcolano gli hash reali, richiedono una collection full da 76.499
point e verificano che l'identità del dataset sia uniforme in tutti i payload. Una collection
sample viene rifiutata.

### Disegno dell'esperimento

- Dataset: MCQ e no-hint.
- Retrieval: Dense e Hybrid RRF (`rrf_k=30`).
- Cutoff: 5, 10, 20, 50, 100.
- Filtri: `none`, i tre legacy, status `current|partial` separati a livello legge/articolo/passaggio,
  vista `current` e vista `not_explicitly_past`.
- Metriche: Article Success, Law Success e MRR.
- Delta: sempre appaiato a `none` sullo stesso indice e sugli stessi QID.
- Inferenza primaria: no-hint Hybrid@10 sulle due viste, bootstrap appaiato con 10.000 repliche,
  seed 42 e CI Bonferroni 97,5% per Article Success.
- Controllo: Dense ANN-vs-exact a k=10/50 per `none` e per le due viste.

I verdetti sono separati in `benchmark_full_coverage`, `active_slice_safety`,
`retrieval_effect` e `bootstrap_supported`. Nessuno di essi modifica automaticamente le
configurazioni RAG.

### Dieci riferimenti revisionati

| QID | riferimento atteso v2 | relazione risposta | supporto corpus-only | stato supporto | viste supporto |
|---|---|---|---|---|---|
| eval-0002 | LR 5/2000 art. 2 `partial` | expected | art. 2 c. 2 | `current` | historical, not_explicitly_past |
| eval-0003 | LR 5/2000 art. 16 `unknown` | expected | art. 16 c. 2 | `current` | historical, not_explicitly_past |
| eval-0013 | LR 44/1991 art. 2 `past` | elsewhere | art. 3 c. 3 | `past` | historical |
| eval-0014 | LR 44/1991 art. 2 `past` | expected | art. 2 | `past` | historical |
| eval-0015 | LR 44/1991 art. 3 `past` | expected | art. 3 c. 2 | `past` | historical |
| eval-0016 | LR 44/1991 art. 3 `past` | expected | art. 3 c. 1 lett. d | `past` | historical |
| eval-0073 | LR 56/1983 art. 1 `partial` | expected | art. 1 c. 1 | `current` | historical, not_explicitly_past |
| eval-0075 | LR 56/1983 art. 1 `partial` | expected | art. 1 c. 3 | `current` | historical, not_explicitly_past |
| eval-0076 | LR 56/1983 art. 5 `past` | elsewhere | art. 3 c. 5 | `current` | historical, not_explicitly_past |
| eval-0100 | LR 5/2020 art. 5 `partial` | expected | art. 5 c. 1 | `current` | historical, current, not_explicitly_past |

`eval-0003` non è mai interamente `past`: il comma 2 è attivo, mentre una parentesi quadra nel
comma 3 resta evidenza editoriale ambigua. Per `eval-0100`, l'applicabilità temporale nel 2020 non
è risolvibile attraverso la sola vigenza del corpus.

### Risultati retrieval

Questa sezione viene popolata esclusivamente dal run full terminale; nessuna metrica è derivata
dalla collection sample.

## Riferimenti

- Run dir: `data/retrieval_eval_runs/diagnostic_full_utopia_throttled__20260525T091921Z/`
  (`scenarios.csv`, `sweep_direct.csv`, `sweep_rerank.csv`, `sweep_query_rewriting.csv`,
  `manifest.json`, `recommended_advanced_config.json`).
- Spec: [docs/specs/06b_retrieval_diagnostics.md](../specs/06b_retrieval_diagnostics.md) · Metodo:
  [docs/notes/06b_retrieval_diagnostics_methodology.md](../notes/06b_retrieval_diagnostics_methodology.md).
- Notebook: `notebooks/06b_retrieval_diagnostics.ipynb`.
