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

## Audit dei filtri di vigenza

Il profilo `filter_audit` separa due domande che il vecchio esperimento aggregava:

1. il filtro conserva i target e i passaggi di supporto compatibili con il diritto corrente?
2. sui medesimi QID e sul medesimo indice migliora effettivamente il retrieval rispetto a `none`?

La copertura statica usa il target unico `(qid, article_id)` e distingue target completamente,
parzialmente o per nulla eleggibili. MCQ e no-hint condividono questa analisi; le due forme vengono
separate soltanto nelle metriche retrieval perché il testo della query può differire.

La matrice confronta filtri legacy, stati `current|partial` ai tre livelli e le viste `current` e
`not_explicitly_past`. Dense e Hybrid usano `k={5,10,20,50,100}` e `rrf_k=30`; lo stesso filtro è
passato ai prefetch dense e sparse. Il profilo non usa LLM, reranking, grafo o query rewriting.

Le metriche restano Article Success, Law Success e MRR. Ogni delta è appaiato alla riga `none` della
stessa domanda e collection. Il confronto primario è no-hint Hybrid@10 per le due viste nuove:
10.000 bootstrap appaiati, seed 42 e intervallo 97,5% per Article Success; MRR e confronti secondari
usano intervalli descrittivi 95%.

La conclusione non usa una singola etichetta: riporta separatamente copertura completa del
benchmark, sicurezza sulla slice attiva, effetto retrieval e supporto bootstrap. I riferimenti
storici restano nel benchmark completo, mentre i due qrel mismatch confermati sono annotati nel
companion review senza correggere lo scoring.

Un controllo Dense con `SearchParams(exact=True)` misura l'overlap ANN/exact. In modalità Qdrant
locale è un controllo di sanità e non dimostra il comportamento HNSW di un deployment server.

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
