# 06b - Metodologia Retrieval Diagnostics

## Obiettivo

Questa nota raccoglie il metodo usato per misurare il retrieval prima della generazione finale. La fase 06b serve a capire se la pipeline recupera gli articoli corretti, quanto in alto li ordina e quali varianti meritano una run Advanced Graph RAG end-to-end.

## Metodo iniziale

La diagnosi parte da run retrieval-only salvate in `data/retrieval_eval_runs/`. Ogni scenario viene letto da `scenarios.csv` e confrontato su tre metriche principali:

- `article_hit_pct`: percentuale di domande in cui l'articolo atteso compare nei candidati;
- `law_hit_pct`: percentuale di domande in cui almeno la legge attesa compare nei candidati;
- `article_mrr`: ranking medio reciproco del primo articolo corretto.

La baseline retroattiva è `default__20260511T180530Z`. Le fasi successive useranno questa nota insieme al log esperimenti per motivare re-index, hybrid retrieval, reranking e query rewriting.

## Regole operative

- Non sovrascrivere run precedenti.
- Registrare nel log ogni run utile, anche se uno scenario viene saltato.
- Distinguere sempre miglioramenti reali da aumenti degenerati dovuti solo a budget `top_k` molto grandi.
- Promuovere in step 06 solo configurazioni con metriche e limiti documentati.

Gli esiti delle run, le leve promosse e quelle scartate sono in [results/06b](../results/06b_retrieval_diagnostics.md).

## Hybrid retrieval

La Fase 2 usa il nuovo indice BGE-M3 `legal_chunks_bge_m3`, che espone un vettore dense e uno sparse. Il notebook 06b è configurato per eseguire lo sweep F con:

- dense BGE-M3 su `top_k={10,20,50,100}`;
- hybrid BGE-M3 con RRF su `top_k={10,20,50,100}` e `rrf_k={30,60,90}`;
- filtro metadata disattivato (`filter=none`) per isolare l'effetto del retrieval.

Il confronto con il vecchio indice non riesegue il dense-only storico: legge `data/retrieval_eval_runs/default__20260511T180530Z/scenarios.csv` e lo affianca ai risultati nuovi. La lettura corretta è:

- vecchio dense vs nuovo dense: effetto del re-index BGE-M3;
- nuovo dense vs nuovo hybrid: effetto specifico della fusione dense+sparse;
- vecchio dense vs nuovo hybrid: effetto complessivo della fase.

## LLM reranking

La Fase 3 misura il reranking LLM sopra il miglior assetto hybrid emerso dallo sweep F. Il reranker non sostituisce il retrieval: riceve i candidati gia recuperati, assegna a ogni chunk uno score discreto `0`, `1` o `2`, e l'evaluator applica solo dopo l'ordinamento score-descendente con tie-break sul ranking originale.

Il notebook 06b mantiene `RUN_RERANK=False` di default per evitare chiamate Utopia accidentali. Quando viene attivato:

- usa `RERANK_PROMPT_VERSION` nella chiave cache;
- scrive la cache in `data/cache/rerank/<sanitized_model>.jsonl`;
- usa un pilot deterministico di 30 domande per dataset (`RERANK_SAMPLE_SEED=42`);
- valuta `rerank_input_k={20,50,100}` e `rerank_output_k={3,5,10}`;
- registra hit rate cache, miss cache e failure strutturati nel manifest;
- scrive le righe in `sweep_rerank.csv` e aggiunge `+ LLM rerank` alla waterfall quando ci sono risultati.

La cache memorizza score grezzi per `chunk_id`, non l'ordinamento finale. In questo modo cambiare `rerank_output_k` riusa gli stessi score e non richiede nuove chiamate LLM.

## Query rewriting / HyDE

La Fase 4 prepara tre trasformazioni della domanda prima del retrieval:

- `rewrite`: riscrive la domanda naturale come query legale concisa;
- `hyde`: genera un breve passaggio normativo ipotetico, senza inventare citazioni;
- `multi_query`: produce esattamente tre formulazioni alternative della stessa esigenza informativa.

Il notebook 06b mantiene `RUN_QUERY_REWRITING=False` di default. Quando viene attivato, il pilot usa il miglior assetto hybrid disponibile, campiona 30 domande per dataset con seed fisso e confronta le strategie contro `none`. Le trasformazioni sono prodotte con output strutturato Pydantic, cache versionata e prompt `QUERY_REWRITING_PROMPT_VERSION`.

La cache vive in `data/cache/query_rewriting/` e separa strategia, modello e versione prompt nel nome file. La chiave include domanda, strategia, modello e versione prompt; quindi cambiare prompt invalida la cache senza cancellare artifact precedenti.
