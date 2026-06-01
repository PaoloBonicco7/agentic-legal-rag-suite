# Implementation Summary and References

## Scope

Questo documento riassume in forma sintetica le scelte implementative adottate nel progetto di tesi e le ancora alla letteratura. Per ogni step della pipeline (preprocessing, dataset, indexing, baseline no-RAG, Simple RAG, retrieval diagnostics, Advanced Graph RAG) viene fornito: il problema affrontato, la strategia adottata con la motivazione, i risultati principali quando disponibili, e una raccolta di reference focalizzate sull'argomento. I paper possono ripetersi tra sezioni quando uno stesso lavoro motiva scelte diverse; questo è voluto e segnala la centralità di alcuni riferimenti (LegalBench-RAG, BGE-M3, RRF, Lost in the Middle) nell'intera pipeline.

L'obiettivo della tesi è verificare empiricamente se un retrieval più ricco (hybrid dense+sparse + query rewriting) migliora le risposte legali su un corpus di leggi regionali italiane rispetto a un Simple RAG dense-only e a un baseline no-RAG. Il confronto è strutturato come waterfall di componenti isolabili (`A → B → C → F → G → H`) e ogni leva viene promossa solo se i diagnostics retrieval-only la giustificano.

## 01 - Laws Preprocessing

### Problema

Il corpus sorgente contiene leggi regionali in HTML: utile per la lettura umana, ma non direttamente adatto a retrieval, citazioni controllate o RAG graph-aware. Il problema è trasformare documenti semi-strutturati in unità stabili, ispezionabili e recuperabili, senza perdere identità normativa, struttura interna, provenienza e relazioni esplicite tra leggi.

### Strategia adottata

Il preprocessing trasforma gli HTML in record JSONL strutturati: leggi, articoli, passaggi, note, relazioni esplicite e chunk. La pipeline usa segnali deterministici del documento — filename, heading, anchor, link e citazioni testuali — per ricostruire la struttura giuridica prima di applicare il chunking. Ogni chunk mantiene testo pulito, contesto per embedding (`text_for_embedding`), metadati di provenienza, stato normativo e relazioni esplicite. Il dataset corrente contiene 3.145 leggi, 17.774 articoli, 76.390 passaggi, 76.467 chunk e 35.159 edge, con quality gate e hash nel manifest.

Questa è una scelta deliberatamente *structure-aware*: prima si ricostruisce la struttura giuridica del corpus, poi si applica il chunking. Questo evita di trattare le leggi come testo piatto e permette ai passaggi recuperati di portare con sé contesto, citabilità e segnali utili per retrieval filtrato o graph-aware. La separazione `text` (visualizzato) vs `text_for_embedding` (indicizzato) è essenziale: il modello embedding vede il contesto giuridico, ma il retrieval restituisce una fonte leggibile e filtrabile.

### Reference

- [LegalHTML: Semantic mark-up of legal acts using web technologies, 2023](https://www.sciencedirect.com/science/article/pii/S0267364923000985) — motiva la rappresentazione strutturata di atti normativi in HTML/semantic web.
- [Modelling Legislative Systems into Property Graphs to Enable Advanced Pattern Detection, 2024](https://arxiv.org/abs/2406.14935) — giustifica la modellazione di leggi, articoli, citazioni, modifiche e abrogazioni come grafo property-based.
- [An Ontology-Driven Graph RAG for Legal Norms, 2026](https://journals.sagepub.com/doi/10.3233/FAIA251598) — riferimento recente per RAG legale sensibile a struttura, temporalità e provenienza.
- [Legal Chunking: Evaluating Methods for Effective Legal Text Retrieval, 2024](https://journals.sagepub.com/doi/10.3233/FAIA241255) — motiva l'importanza del chunking nel dominio legale.
- [Automatic semantic edge labeling over legal citation graphs, 2018](https://link.springer.com/article/10.1007/s10506-018-9217-1) — motiva l'estrazione e la classificazione di relazioni esplicite tra norme.
- [Retrieval-Augmented Generation for Large Language Models: A Survey, 2023/2024](https://arxiv.org/abs/2312.10997) — colloca il preprocessing nella pipeline RAG.

## 02 - Evaluation Dataset

### Problema

Le domande sorgenti devono essere rese confrontabili tra baseline no-RAG, Simple RAG e Advanced RAG. Senza normalizzazione e controlli, la valutazione rischierebbe di misurare rumore nel dataset invece della qualità del metodo. Il problema è costruire un benchmark piccolo ma stabile, tracciabile e coerente tra formato a scelta multipla e formato aperto.

### Strategia adottata

La pipeline normalizza i CSV sorgenti in due dataset JSONL allineati: `mcq` (multiple-choice) e `no_hint` (open-ended). Ogni coppia condivide `qid`, livello (L1–L4), intento della domanda, risposta corretta e riferimenti attesi (`expected_law_ids`, `expected_article_ids`). Le righe vuote vengono scartate, i campi obbligatori validati e la distribuzione dei livelli resa esplicita. Il dataset corrente contiene 100 domande MCQ e 100 no-hint, bilanciate sui quattro livelli.

Questa è una scelta *evaluation-first*: prima di confrontare sistemi RAG diversi, il benchmark viene reso stabile e riproducibile. L'allineamento MCQ/no-hint permette di confrontare risposte chiuse e aperte sullo stesso intento informativo; i riferimenti attesi servono come ground truth per i diagnostics di retrieval (`article_hit`, `law_hit`, `MRR`).

### Reference

- [BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models, 2021](https://arxiv.org/abs/2104.08663) — motiva benchmark IR standardizzati e comparabili.
- [LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models, 2023](https://arxiv.org/abs/2308.11462) — giustifica task legali costruiti con attenzione al ragionamento giuridico.
- [LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain, 2024](https://arxiv.org/abs/2408.10343) — riferimento diretto per benchmark RAG legali con ground truth retrieval-aware.
- [A Reasoning-Focused Legal Retrieval Benchmark, 2025](https://arxiv.org/abs/2505.03970) — benchmark legal RAG realistici orientati a retrieval+QA.
- [LexGLUE: A Benchmark Dataset for Legal Language Understanding in English, 2021/2022](https://arxiv.org/abs/2110.00976) — tradizione benchmark legal NLP.
- [Datasheets for Datasets, 2021 version](https://arxiv.org/abs/1803.09010) — motiva manifest, hash, quality gate e tracciabilità dei dati generati.
- [Data Statements for Natural Language Processing, 2018](https://aclanthology.org/Q18-1041/) — documentazione esplicita delle caratteristiche del dataset.

## 03 - Indexing Contract

### Problema

Dopo il preprocessing, i chunk giuridici sono strutturati ma non interrogabili in modo efficiente da una pipeline RAG. Il problema è trasformarli in un indice riproducibile che supporti ricerca semantica, matching lessicale, filtri su metadati normativi e tracciabilità delle fonti, senza perdere informazioni come legge, articolo, stato vigente/storico e relazioni esplicite.

### Strategia adottata

La pipeline indicizza ogni chunk come point Qdrant con id stabile (`uuid5(NAMESPACE_URL, chunk_id)`), payload completo e `content_hash = sha256(text_for_embedding.strip())` per rerun idempotenti. Il testo indicizzato è `text_for_embedding`, cioè il testo del chunk arricchito con contesto normativo minimo.

La scelta dell'embedder è stata revisionata: dal vecchio backend `Utopia/Nomic` (dense-only, 768-dim) si è passati a **BGE-M3 locale** (dense 1024 + sparse nativo). Il motivo è che BGE-M3 produce nello stesso modello rappresentazioni multilingue dense e sparse lessicali, abilitando hybrid retrieval senza dipendenze esterne e con un miglioramento netto del retrieval misurato in 06b (vedi sotto). I payload index su `law_id`, `law_status`, `index_views`, `article_id` e `relation_types` permettono di filtrare i risultati per stato normativo, vista corrente/storica, articolo o relazioni — un aspetto centrale nel dominio giuridico, dove la rilevanza non dipende solo dalla similarità testuale ma anche da provenienza, validità e struttura della norma.

Qdrant viene usato come indice di retrieval controllabile, non come semplice vector store: named vectors per dense+sparse, payload JSON per provenance, payload indexes per vincoli giuridici, query ibride con RRF tramite l'API nativa (`prefetch` + `RrfQuery`), e modalità locale/Docker per mantenere il workflow riproducibile.

### Risultati principali

Confronto retrieval-only Nomic vs BGE-M3 sullo stesso corpus e stesse 100 domande (`docs/results/06b_embedding_index_comparison.md`):

| indice | top_k | article_hit | law_hit | MRR |
|---|---:|---:|---:|---:|
| Utopia/Nomic dense | 10 | 45.0% | 72.0% | 0.288 |
| Utopia/Nomic dense | 100 | 66.0% | 87.0% | 0.295 |
| **BGE-M3 dense** | **10** | **73.0%** | **97.0%** | **0.519** |
| BGE-M3 dense | 100 | 88.0% | 99.0% | 0.524 |
| **BGE-M3 hybrid (rrf_k=30)** | **100** | **89.0%** | **99.0%** | **0.551** |

Il re-index BGE-M3 porta **+28pp di article_hit@10** e **+0.231 di MRR** rispetto a Nomic dense@10, prima ancora di attivare hybrid. Questo giustifica il cambio di embedder come scelta architetturale, non cosmetica.

### Reference

- [Qdrant documentation: Indexing](https://qdrant.tech/documentation/manage-data/indexing/) — fonte tecnica per payload indexes, vector index, sparse index e filtered search.
- [Qdrant documentation: Filtering](https://qdrant.tech/documentation/search/filtering/) — uso di filtri su metadati quando l'embedding non può rappresentare vincoli giuridici espliciti.
- [Qdrant documentation: Hybrid Queries](https://qdrant.tech/documentation/search/hybrid-queries/) — dense+sparse retrieval con `prefetch` e RRF.
- [M3-Embedding / BGE-M3, 2024](https://arxiv.org/abs/2402.03216) — giustifica BGE-M3 come modello multilingual, multi-granularity, multi-functionality con dense+sparse nello stesso encoder.
- [Reciprocal Rank Fusion, 2009](https://doi.org/10.1145/1571941.1572114) — riferimento fondativo per la fusione RRF dei ranking dense e sparse, usata in `search_hybrid()`.
- [Survey of Vector Database Management Systems, 2024](https://arxiv.org/abs/2310.14021) — contesto accademico su vector database e query ibride vettori+attributi.
- [Dense Passage Retrieval for Open-Domain Question Answering, 2020](https://arxiv.org/abs/2004.04906) — riferimento generale per il passaggio da documenti a unità recuperabili tramite rappresentazioni dense.
- [LegalBench-RAG, 2024](https://arxiv.org/abs/2408.10343) — motiva retrieval preciso di segmenti legali minimi e citabili invece di documenti troppo ampi.
- [Hybrid Legal Norm Retrieval: Leveraging Knowledge Graphs and Textual Representations, 2024](https://doi.org/10.3233/FAIA241245) — retrieval giuridico ibrido che combina testo, BM25/transformer e conoscenza strutturale.
- [Finding the Law, 2023](https://arxiv.org/abs/2301.12847) — nello statutory retrieval la struttura della legge conta quanto il testo.

## 04 - No-RAG Baseline

### Problema

Prima di attribuire miglioramenti al retrieval, bisogna misurare cosa il modello riesce a rispondere senza accesso al corpus normativo. Senza un baseline no-RAG, il confronto con Simple RAG e Advanced RAG sarebbe ambiguo: un aumento di accuratezza potrebbe dipendere dalla capacità interna del modello, dal prompt o dal dataset, non dalla pipeline di recupero e grounding.

### Strategia adottata

Il notebook valuta i dataset puliti dello step 02 in modalità *model-only*: nessun Qdrant, nessun chunk recuperato, nessun contesto legale aggiunto al prompt. La pipeline produce due segnali complementari: MCQ valutato in modo deterministico sulla label corretta, e no-hint dove il modello genera una risposta aperta poi giudicata da un LLM con rubrica semantica `0-2`. Le chiamate usano output JSON strutturato e temperatura `0`; manifest, hash, versioni di prompt/modelli e metriche (`accuracy`, `coverage`, `strict_accuracy`, `by_level`) restano allineati agli step RAG successivi.

Questa è il controllo sperimentale della tesi: definisce la soglia minima model-only su domande giuridiche italiane e rende misurabile il valore aggiunto del retrieval. Senza questo baseline, attribuire i miglioramenti del Simple/Advanced RAG al retrieval invece che al modello sarebbe scientificamente debole.

### Reference

- [LawBench: Benchmarking Legal Knowledge of Large Language Models, 2023](https://arxiv.org/abs/2309.16289) — benchmark legal LLM su conoscenza, comprensione e applicazione del diritto.
- [GPT-4 Passes the Bar Exam, 2023/2024](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4389233) — esempio di valutazione zero-shot/model-only su compiti legali con componenti MCQ e risposta aperta.
- [Re-evaluating GPT-4's Bar Exam Performance, 2023/2024](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4441311) — cautela, trasparenza metodologica e metriche riproducibili nelle valutazioni legali.
- [LegalBench-RAG, 2024](https://arxiv.org/abs/2408.10343) — colloca il baseline no-RAG nel confronto con pipeline legal RAG.
- [G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment, 2023](https://arxiv.org/abs/2303.16634) — giustifica un giudice LLM con rubrica numerica per risposte aperte, tenendo presenti i possibili bias.
- [ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems, 2023/2024](https://arxiv.org/abs/2311.09476) — valutazioni automatiche coerenti tra componenti generative e sistemi RAG.

## 05 - Simple RAG

### Problema

La baseline no-RAG misura quanto il modello risponde senza accesso al corpus, ma non garantisce grounding sulle fonti. In ambito legale questo è un limite importante: le risposte devono essere ancorate a norme aggiornate, verificabili e citabili. Lo step 05 introduce quindi una baseline RAG minima per misurare il valore del retrieval prima di aggiungere componenti più avanzate come hybrid, reranking o graph RAG.

### Strategia adottata

Il notebook esegue un Simple RAG end-to-end sul benchmark MCQ e no-hint. Per ogni domanda: calcola l'embedding della query, interroga Qdrant in dense retrieval sulla collection `legal_chunks_bge_m3`, applica il filtro statico `law_status=current`, recupera i primi 3 chunk e costruisce un contesto limitato a 3 chunk e 8.000 caratteri. Il modello genera una risposta strutturata usando solo il contesto recuperato, e produce citazioni vincolate ai `chunk_id` effettivamente inclusi. Le metriche restano confrontabili direttamente con no-RAG: MCQ deterministico, no-hint con rubrica `0-2` LLM-as-a-judge.

Questa è una scelta *baseline-first*: prima si verifica che retrieval, costruzione del contesto, generazione, citazioni e metriche funzionino con il minimo numero di componenti. RAG si adatta bene al dominio legale perché il diritto è knowledge-intensive, dipende da fonti testuali esterne e richiede provenienza controllabile; allo stesso tempo, studi empirici mostrano che RAG riduce ma non elimina le allucinazioni, quindi citazioni vincolate e trace row-level sono essenziali.

### Risultati principali

Confronto no-RAG vs Simple RAG (stesso modello, stesso dataset):

| dataset | no-RAG | Simple RAG BGE-M3 | δ |
|---|---:|---:|---:|
| MCQ accuracy | 81.0% | 79.0% | −2.0pp |
| no_hint accuracy | 48.5% | 59.0% | **+10.5pp** |
| no_hint score | 97/200 | 118/200 | **+21 punti** |

Il valore aggiunto del retrieval emerge soprattutto sulle risposte open-ended: il contesto recuperato dal nuovo indice porta lo score no-hint da 97/200 a 118/200. Su MCQ la differenza è entro il rumore e in linea con quanto atteso per domande dove il modello può ragionare correttamente anche senza testo normativo esplicito.

### Reference

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, 2020](https://arxiv.org/abs/2005.11401) — riferimento fondativo per il paradigma RAG.
- [Dense Passage Retrieval for Open-Domain Question Answering, 2020](https://arxiv.org/abs/2004.04906) — motivare una baseline con dense vector retrieval prima di introdurre componenti ibride o reranking.
- [BGE M3-Embedding, 2024](https://arxiv.org/abs/2402.03216) — embedding multilingue e multi-granulare adatto a italiano e passaggi normativi.
- [CBR-RAG: Case-Based Reasoning for Retrieval Augmented Generation in LLMs for Legal Question Answering, 2024](https://arxiv.org/abs/2404.04302) — uso di evidenza recuperata per validare output generati in task legali.
- [Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools, 2024](https://arxiv.org/abs/2405.20362) — retrieval e citazioni aiutano ma richiedono verifica esplicita.
- [LegalBench-RAG, 2024](https://arxiv.org/abs/2408.10343) — retrieval legale con snippet precisi e limiti di contesto.
- [LRAGE: Legal Retrieval Augmented Generation Evaluation Tool, 2025](https://arxiv.org/abs/2504.01840) — valutazioni RAG legali che isolano corpus, retriever, modello generativo e metriche.
- [Enhancing legal document building with Retrieval-Augmented Generation, 2025](https://www.sciencedirect.com/science/article/pii/S2212473X25001014) — riferimento recente per RAG legale con vector database e supervisione umana.

## 06b - Retrieval Diagnostics

### Problema

La baseline Simple RAG usa pochi chunk recuperati con ricerca densa, quindi può fallire quando la domanda usa termini diversi dalla norma, quando il riferimento corretto compare più in basso nel ranking, o quando servono segnali strutturali (relazioni tra leggi, articoli modificati, stato normativo). In ambito legale il problema non è solo generare una risposta plausibile, ma portare nel contesto il passaggio normativo minimo, citabile e aggiornato.

Il notebook `06b_retrieval_diagnostics.ipynb` separa il problema di retrieval da quello di generazione: prima misura se la norma attesa entra nei candidati (`article_hit`, `law_hit`, `MRR`); poi il notebook 06 promuove solo le componenti che migliorano il retrieval senza introdurre rumore non controllato. Una leva viene promossa solo se: gain ≥ +1pp di `article_hit` sopra la baseline corretta, sample non degenere (100 domande), configurazione completamente documentata.

### Strategia adottata — il waterfall A/B/C/F/G/H

Sei esperimenti, eseguiti come waterfall comparabile sullo stesso indice `legal_chunks_bge_m3` (76.467 chunk, BGE-M3 dense 1024 + sparse lessicale nativo):

- **A — Dense baseline e curva top-k**: stabilisce la baseline `dense@10` e il recall ceiling `dense@100`.
- **B — Filtri metadata statici**: testa se `law_status=current` e affini migliorano la baseline.
- **C — Graph expansion**: espande la candidate list seguendo gli archi del grafo legale (`REFERENCES`, `AMENDS`, `INSERTS`, `MODIFIED_BY`, `INSERTED_BY`), a hop 1, con `min_edge_confidence=0.45` e cap `max_chunks_per_law`.
- **F — Hybrid retrieval (dense + sparse via RRF)**: fonde il ranking BGE-M3 dense con quello sparse lessicale tramite Reciprocal Rank Fusion, usando l'API nativa Qdrant.
- **G — LLM reranking (pilot)**: riordina gli `input_k` candidati hybrid con un giudice LLM che assegna score `0/1/2`, poi taglia agli `output_k` migliori.
- **H — Query rewriting / HyDE / multi-query (pilot)**: trasforma la domanda prima del retrieval con `rewrite` (riformulazione legale concisa), `hyde` (passaggio normativo ipotetico) o `multi_query` (n=3 riformulazioni alternative fuse client-side).

Ogni esperimento produce row-level diagnostics, cache LLM versionate (`RERANK_PROMPT_VERSION`, `QUERY_REWRITING_PROMPT_VERSION`) e una riga in `scenarios.csv`. La decisione finale è codificata in `recommended_advanced_config.json` consumato dallo step 06.

### Risultati principali — waterfall

Run di riferimento: `data/retrieval_eval_runs/diagnostic_full_utopia_throttled__20260525T091921Z/`, profilo `full`, schema `retrieval-evaluation-v4`. LLM per G e H: `SLURM.gpt-oss:120b` via Utopia.

| stage | scenario | n | article_hit | law_hit | MRR | δ vs baseline | esito |
|---|---|---:|---:|---:|---:|---:|---|
| baseline | Dense@10 (no filter) | 100 | 73.0% | 97.0% | 0.519 | — | reference |
| direct ceiling | Dense top_k=100 | 100 | 88.0% | 99.0% | 0.524 | +15.0pp | informativo |
| B — filter | + Filter `law_status=current` | 100 | 71.0% | 93.0% | 0.507 | **−2.0pp** | scartato |
| C — graph | + Best graph from sweep | 100 | 74.0% | 97.0% | 0.520 | +1.0pp | scartato (noise=99.7%) |
| **F — hybrid** | **Hybrid top_k=100, rrf_k=30** | 100 | **89.0%** | **99.0%** | **0.551** | **+16.0pp** | **promosso** |
| G — rerank | + LLM rerank (top_k=20, output_k=10) | 23 (pilot) | 82.6% | 100.0% | 0.711 | +9.6pp | scartato (−6.4pp vs hybrid) |
| **H — rewriting** | **+ Multi-query (n=3)** | 30 (pilot) | **93.3%** | **100.0%** | **0.467** | **+20.3pp** | **promosso (+4.3pp vs hybrid)** |

**Letture chiave**:

1. **Hybrid è la leva singola più forte**: +16pp di `article_hit` sopra la baseline dense@10, con MRR che cresce da 0.519 a 0.551. Il guadagno non è solo recall: anche dense@100 raggiunge 88% di hit ma resta con MRR 0.524. RRF promuove in cima i chunk che entrambe le viste (semantica dense, lessicale sparse) considerano rilevanti — esattamente il comportamento desiderato quando i primi candidati pesano molto. Il punto operativo `top_k=100, rrf_k=30` è scelto per massimizzare recall lasciando al downstream Advanced RAG la possibilità di applicare un rerank o un cap di contesto.

2. **Multi-query (H) aggiunge un guadagno indipendente**: sul pilot di 30 domande sale a 93.3% di `article_hit` con `law_hit=100%`, +4.3pp sopra hybrid puro. Le tre riformulazioni alternative coprono meglio la varietà del vocabolario normativo della stessa intent informativa (numeri di articolo, sigle, locuzioni latine, riferimenti puntuali a leggi regionali). Costo: una chiamata LLM per query + 3 retrieval Qdrant per domanda, ammortizzato da cache versionata.

3. **Rerank LLM (G) scartato per evidenza**: su 18 configurazioni testate `recovered=0` ovunque — l'LLM non promuove mai in `output_k` un articolo che hybrid aveva escluso dal proprio `input_k`. In più, in molte configurazioni demota articoli corretti che hybrid aveva piazzato in cima (impatto netto −6.4pp su MCQ vs hybrid puro). Migliora la precision-at-top (MRR 0.71 vs 0.55) ma a costo della recall, e con un `failure_rate=30.6%` su output strutturato. Resta nel codice (`rerank_enabled=false`) come leva riattivabile in futuro con un prompt più stabile.

4. **Filtri metadata (B) e graph expansion (C) scartati senza ambiguità**: il filtro `law_status=current` esclude 4 domande dell'evaluation set i cui riferimenti puntano a leggi abrogate, abbassando il hit di 2pp. La graph expansion ha `expansion_noise_ratio=0.997` — oltre il 99% dei chunk aggiunti non è rilevante. Promuoverle gonfierebbe il candidate set senza beneficio retrieval misurabile.

### Configurazione promossa

```json
{
  "hybrid_enabled": true,
  "top_k": 100,
  "rrf_k": 30,
  "metadata_filters_enabled": false,
  "graph_expansion_enabled": false,
  "rerank_enabled": false,
  "query_rewriting_recommendation": {
    "enabled": true,
    "strategy": "multi_query",
    "n": 3
  }
}
```

### Reference

- [Reciprocal Rank Fusion, 2009](https://doi.org/10.1145/1571941.1572114) — fonde ranking eterogenei (dense, sparse) senza assumere score direttamente confrontabili. Base teorica di Esperimento F.
- [BGE M3-Embedding, 2024](https://arxiv.org/abs/2402.03216) — embedding multilingue con dense+sparse nello stesso modello, adatto all'italiano normativo. Base di tutti gli esperimenti.
- [Qdrant documentation: Hybrid Queries](https://qdrant.tech/documentation/search/hybrid-queries/) — implementazione `prefetch + RrfQuery` usata in `search_hybrid()`.
- [Query Rewriting for Retrieval-Augmented Large Language Models, 2023](https://arxiv.org/abs/2305.14283) — pattern rewrite-retrieve-read quando la query utente non è allineata al linguaggio del corpus. Motiva Esperimento H.
- [Query2doc: Query Expansion with Large Language Models, 2023](https://arxiv.org/abs/2303.07678) — query expansion / pseudo-documenti per migliorare retrieval dense e sparse.
- [Precise Zero-Shot Dense Retrieval without Relevance Labels / HyDE, 2023](https://aclanthology.org/2023.acl-long.99/) — risposte/passaggi ipotetici come ponte verso documenti reali. Strategia `hyde` in Esperimento H.
- [DMQR-RAG: Diverse Multi-Query Rewriting for RAG, 2024](https://arxiv.org/abs/2411.13154) — multi-query rewriting e diversificazione delle formulazioni di ricerca. Motiva `multi_query` (n=3) come strategia preferita.
- [RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs, 2024](https://arxiv.org/abs/2407.02485) — reranking LLM come componente da testare prima di ridurre il contesto. Motiva Esperimento G.
- [Hybrid Legal Norm Retrieval, 2024](https://doi.org/10.3233/FAIA241245) — combinare rappresentazioni testuali, segnali lessicali/semantici e conoscenza strutturata sulle norme.
- [LegalBench-RAG, 2024](https://arxiv.org/abs/2408.10343) — retrieval di snippet legali minimi e citabili; criterio di promozione retrieval-only.
- [LeReRAG: Measuring Legal Relevance in Retrieval Augmented Generation Applications, 2024](https://journals.sagepub.com/doi/abs/10.3233/FAIA241284) — metriche e rubriche di rilevanza specifiche del dominio legale.
- [An Ontology-Driven Graph RAG for Legal Norms, 2026](https://journals.sagepub.com/doi/10.3233/FAIA251598) — Graph RAG legale structure-aware; motiva perché la graph expansion va controllata e non usata in modo indiscriminato (Esperimento C).

## 06 - Advanced Graph RAG (end-to-end)

### Problema

Il notebook `06_advanced_graph_rag.ipynb` deve verificare se il retrieval migliorato emerso da 06b si traduce in risposte legali migliori sul benchmark completo. Non è una somma cieca di tecniche: ogni componente attivata deve essere giustificata dai diagnostics, e la pipeline finale deve restare spiegabile (un reader deve poter ricostruire una risposta dai chunk recuperati al contesto effettivamente passato al modello).

### Strategia adottata

La pipeline applica la configurazione promossa da 06b: hybrid retrieval BGE-M3 dense+sparse con `top_k=100`, `rrf_k=30`, e query rewriting `multi_query` con `n=3` riformulazioni alternative fuse client-side via dedup. Le componenti scartate da 06b — filtri metadata statici, graph expansion, LLM reranking — restano nel codice come flag disattivabili, ma sono **off di default** in linea con l'evidenza diagnostica.

Per ogni domanda il flusso è:

1. (opzionale, attivo di default) trasformare la query con `apply_query_rewriting(strategy="multi_query", n=3)`; cache versionata su `(question, strategy, model, QUERY_REWRITING_PROMPT_VERSION)`.
2. interrogare Qdrant in hybrid (`search_hybrid`) una volta per variante; fondere i candidati client-side con dedup deterministico (preservando ordine di prima apparizione).
3. troncare a `top_k=100` candidati uniti.
4. costruire un contesto limitato (`max_context_chunks=15`, `max_context_chars=16000`) per non saturare la finestra né far cadere informazione rilevante nella zona mediana (Lost in the Middle).
5. generare risposta + citazioni con output strutturato, vincolando le citazioni ai `chunk_id` effettivamente nel contesto.
6. registrare diagnostics row-level: chunk recuperati, edge usati (graph off), score rerank (rerank off), chunk nel contesto, `reference_law_hit`, `reference_article_hit`, failure category.

L'integrazione di `multi_query` riusa direttamente il modulo `retrieval_evaluation/query_rewriting.py` e la cache `data/cache/query_rewriting/`: zero duplicazione di prompt e zero chiamate LLM aggiuntive se modello e prompt restano invariati tra 06b e 06.

Le ablation run (`all_off`, `+ hybrid only`, `+ hybrid + multi_query`, futuri `+ rerank`, `+ graph`) usano lo stesso codice cambiando solo i flag in `AdvancedRagConfig` e producono directory affiancate in `data/rag_runs/advanced/<run_name>/`.

### Cosa misura e quale evidenza produce

- Confronto end-to-end vs Simple RAG sullo stesso evaluation manifest (hash check obbligatorio): `mcq_accuracy`, `no_hint_accuracy`, `no_hint_score`, `reference_article_hit`, `reference_law_hit`, `by_level`.
- Target dichiarati nella roadmap operativa: `reference_article_hit` ≥ Simple RAG + 20pp, `reference_law_hit` ≥ +15pp, MCQ accuracy ≥ +10pp, `no_hint` judge score ≥ +5pp.
- Se i target non vengono raggiunti, il risultato viene documentato in `docs/results/06_advanced_graph_rag.md` invece di forzare ulteriori complessità (graph, rerank). Questo mantiene la linearità della tesi: l'evidenza guida la promozione, non il contrario.

### Reference

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, 2020](https://arxiv.org/abs/2005.11401) — paradigma RAG end-to-end.
- [Lost in the Middle: How Language Models Use Long Contexts, 2023](https://arxiv.org/abs/2307.03172) — motiva il limite esplicito su `max_context_chunks` e `max_context_chars` e la necessità di ranking prima della generazione.
- [DMQR-RAG: Diverse Multi-Query Rewriting for RAG, 2024](https://arxiv.org/abs/2411.13154) — motivazione diretta per `multi_query` come strategia di riformulazione promossa.
- [Query Rewriting for Retrieval-Augmented Large Language Models, 2023](https://arxiv.org/abs/2305.14283) — rewrite-retrieve-read come pattern di riferimento.
- [Reciprocal Rank Fusion, 2009](https://doi.org/10.1145/1571941.1572114) — fusione dei ranking dense+sparse usata in produzione, anche per fondere i risultati delle varianti multi-query.
- [BGE M3-Embedding, 2024](https://arxiv.org/abs/2402.03216) — embedder che alimenta dense e sparse della pipeline.
- [LegalBench-RAG, 2024](https://arxiv.org/abs/2408.10343) — retrieval di snippet legali minimi e citabili; vincola citazioni ai chunk del contesto.
- [LRAGE: Legal Retrieval Augmented Generation Evaluation Tool, 2025](https://arxiv.org/abs/2504.01840) — valutazione legale che isola corpus, retriever, reranker, modello generativo e metriche.
- [Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools, 2024](https://arxiv.org/abs/2405.20362) — anche con RAG le allucinazioni restano un rischio: motiva citazioni vincolate e trace row-level.
- [Incorporating Legal Structure in Retrieval-Augmented Generation: A Case Study on Copyright Fair Use, 2025](https://arxiv.org/abs/2505.02164) — esempio di RAG legale che combina semantic search, knowledge graph e segnali di citazione; utile per discutere perché in questo progetto la struttura grafo resta "presente ma off".
- [An Ontology-Driven Graph RAG for Legal Norms, 2026](https://journals.sagepub.com/doi/10.3233/FAIA251598) — Graph RAG legale structure-aware, temporale e tracciabile; framework di riferimento per Future Work.
- [CLERC: A Dataset for Legal Case Retrieval and Retrieval-Augmented Analysis Generation, 2024](https://arxiv.org/abs/2406.17186) — collega retrieval legale, citazioni e generazione supportata da fonti.

## Future Sections

- **07 — Evaluation Reporting**: aggregazione finale dei run no-RAG, Simple RAG, Advanced RAG su tabelle confrontabili e diagnosi qualitativa dei fallimenti residui (`right_law_wrong_article`, `no_reference_in_topk`, `context_overflow`).
- **Future Work** già istruito nella roadmap: diversity/MMR cap per legge, failure taxonomy qualitativa, cross-encoder locale (`BAAI/bge-reranker-v2-m3`) come confronto al rerank LLM, graph expansion ricalibrata con noise cap più severo a partire dai seed hybrid, revisione chunking strutturale per ridurre `right_law_wrong_article`.

## Lettura trasversale — paper "load-bearing"

Alcuni paper sostengono più scelte della pipeline. Vale la pena tenerli a mente come pilastri della tesi:

- **[BGE-M3, 2024](https://arxiv.org/abs/2402.03216)** — base di tutto il retrieval (step 03, 05, 06b, 06).
- **[Reciprocal Rank Fusion, 2009](https://doi.org/10.1145/1571941.1572114)** — fusione hybrid e fusione multi-query (step 03, 06b F, 06).
- **[LegalBench-RAG, 2024](https://arxiv.org/abs/2408.10343)** — benchmark di riferimento per legal RAG; motiva snippet minimi e citazioni (step 02, 03, 04, 05, 06b, 06).
- **[Query Rewriting for RAG, 2023](https://arxiv.org/abs/2305.14283)** + **[DMQR-RAG, 2024](https://arxiv.org/abs/2411.13154)** — pattern rewrite-retrieve-read e multi-query (step 06b H, 06).
- **[Lost in the Middle, 2023](https://arxiv.org/abs/2307.03172)** — limiti espliciti su `max_context_chunks` (step 05, 06).
- **[Hallucination-Free?, 2024](https://arxiv.org/abs/2405.20362)** — RAG riduce ma non elimina le allucinazioni; motiva trace row-level e citazioni vincolate (step 05, 06).
- **[Ontology-Driven Graph RAG for Legal Norms, 2026](https://journals.sagepub.com/doi/10.3233/FAIA251598)** — framework di riferimento per Graph RAG legale; usato per discutere perché in questa tesi il graph resta off ma rimane esposto (step 01, 06b C, 06).
