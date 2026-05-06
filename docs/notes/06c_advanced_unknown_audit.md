# Audit dei 22 casi `unknown` — no-hint advanced full_100

Run analizzata: `data/rag_runs/advanced/full_100_dense_graph_rerank/no_hint_results.jsonl` (100 record). Distribuzione `failure_category`: 57 successi, 22 `unknown`, 20 `context_noise`, 1 `generation_error`.

I casi `unknown` per definizione del runner sono righe in cui non si è verificato un retrieval miss esplicito né un errore strutturato, ma il judge ha dato score 0. L'ipotesi iniziale era che fossero "problemi di generazione" (modello che fraintende un buon contesto). **L'audit ribalta questa ipotesi**: solo una minoranza dei 22 sono fallimenti puri di generazione; la maggioranza sono problemi a monte (chunking o retrieval) mascherati come unknown perché la legge giusta è in contesto ma l'articolo/comma giusto no.

## Tassonomia derivata dai dati

| Categoria | Definizione |
|---|---|
| `retrieval_miss_in_law` | La legge giusta è in contesto (`reference_law_hit=True`) ma l'articolo/comma con la risposta non è stato recuperato. |
| `chunk_split_orphan` | Il retrieval ha pescato un chunk "intro" che annuncia una lista (es. "Sono organi:") ma le voci della lista — chunk fratelli — non sono in contesto. |
| `chunk_intro_only` | Tutti i chunk in contesto sono titoli parentetici di articolo (`p:intro`), nessun corpo articolato. Il modello vede solo etichette. |
| `wrong_chunk_reranked` | Il reranker ha dato score 2 a un chunk che NON contiene la risposta (tipicamente perché matcha le keyword). |
| `nuance_misread` | Il chunk con la risposta è in contesto e il modello lo legge male: confonde concetti vicini ("iniziare" vs "stipulare") o non collega un'implicazione. |
| `improper_hedging` | Il chunk con la risposta è visibile ma il modello dichiara "il contesto non contiene…". |
| `invention` | Il modello fabbrica un riferimento legale per riempire un retrieval miss. |
| `judge_calibration_issue` | La risposta del modello è plausibile e coerente con i chunk, ma il judge la punisce (per troppo dettaglio, o perché si aspetta conoscenza generale). |
| `out_of_scope` | La risposta è in normativa nazionale o conoscenza extra-corpus. Né retrieval né modello potevano rispondere dal corpus regionale. |

## Categorizzazione per caso

Ogni riga riporta categoria primaria e, se rilevante, secondaria.

| qid | level | top rerank | primaria | secondaria | nota chiave |
|---|---|---|---|---|---|
| eval-0001 | L1 | 2,1,1,1,1 | `chunk_split_orphan` | — | Chunk top: "1. Sono organi dell'azienda USL:" — lista degli organi è nei chunk fratelli `art:12#p:c1.lit_*` non recuperati. Modello cita art.12 ma non elenca. |
| eval-0011 | L3 | 2,2,1,1,1 | `nuance_misread` | — | Chunk 1 dice privati possono *concorrere*; chunk 2 dice qualsiasi soggetto può *iniziare* ma promozione è di Presidente/Sindaco. Modello collassa in "Sì possono stipulare". |
| eval-0024 | L4 | 2,1,1,1,1 | `wrong_chunk_reranked` | `retrieval_miss_in_law` | Top chunk score 2 parla di autorizzazione del sindaco, NON di pubblicità. Articolo sul divieto pubblicità non recuperato. Il modello hedga correttamente. |
| eval-0025 | L1 | 1,0,0,0,0 | `out_of_scope` | `retrieval_miss_in_law` | Risposta "10 anni" è in D.Lgs. 152/2006 nazionale (lo dice il judge stesso). Nel corpus regionale potrebbe esistere ma non è stata recuperata. |
| eval-0026 | L2 | 1,1,1,1,0 | `retrieval_miss_in_law` | `invention` | Articolo su PRAE adottato dal Consiglio non in contesto. Modello inventa "art.82 c.1" con citazione di chunk irrilevante. |
| eval-0027 | L3 | 2,1,0,0,0 | `chunk_split_orphan` | — | Chunk top: "1. Sono considerati stabilimenti termali quelli in cui si utilizzano:" — la lista (fanghi etc.) è nei sotto-chunk non recuperati. Modello scrive "[testo interrotto nel contesto fornito]". |
| eval-0028 | L4 | 2,1,1,1,1 | `retrieval_miss_in_law` | `invention` | Regola del subingresso non in contesto. Modello inventa fondandosi su art.13 c.4 (contributo, non autorizzazione). |
| eval-0045 | L1 | 2,2,1,0,0 | `wrong_chunk_reranked` | `retrieval_miss_in_law` | Chunk top score 2 = titolo parentetico "(Piante officinali ad uso erboristico…)" e descrizione di "corso tipo A". La definizione vera (art.5 verosimilmente) non recuperata. Reranker ingannato dalle keyword. |
| eval-0050 | L2 | 1,1,0,0,0 | `improper_hedging` | `wrong_chunk_reranked` | Il chunk-5 (rerank=0) dice esplicitamente "Commissione è rinnovata all'inizio di ogni legislatura" — è la risposta, ma il reranker l'ha messo in fondo e il modello l'ha ignorato. |
| eval-0051 | L3 | 2,2,1,0,0 | `judge_calibration_issue` | — | Modello estrae correttamente da chunk 1 ("scopi scientifici e didattici", "Soprintendente"). Judge punisce perché preferiva risposta più concisa. La risposta del modello è fattualmente corretta dai chunk. |
| eval-0055 | L3 | 2,2,2,2,2 | `retrieval_miss_in_law` | — | Tutti i 5 chunk con score 2 parlano di "azienda indenne" (concetto vicino ma diverso). Articolo su "infezione limitata ≤10%" non in contesto. Modello nota correttamente la discrepanza. |
| eval-0056 | L4 | 1,1,1,1,0 | `retrieval_miss_in_law` | `judge_calibration_issue` | Chunk in contesto parlano di "proventi delle sanzioni" non delle sanzioni stesse (sospensione/revoca qualifica). Articolo sanzioni non recuperato. Judge dice "fatto noto" ma è law-specific. |
| eval-0058 | L2 | 1,1,0,0,0 | `retrieval_miss_in_law` | — | L'articolo che cita "anni 2025-2027" (art.1#c1 della l.r. 14/2024) è recuperato in altre query (vedi eval-0014) ma qui non è stato pescato. Top chunk parla solo di "decorrere dall'anno 2024". |
| eval-0059 | L3 | 2,1,1,0,0 | `retrieval_miss_in_law` | — | Articolo che fa rinvio a deliberazione Giunta (probabilmente art.4 o 5) non in contesto. Top chunk = art.1 con descrizione generale. |
| eval-0060 | L4 | 2,2,1,1,1 | `judge_calibration_issue` | — | Modello estrae correttamente l'80%+fideiussione da chunk 2 (rerank=2). Risposta ufficiale "In più soluzioni" è troppo concisa rispetto al testo del chunk; judge punisce il dettaglio. |
| eval-0083 | L3 | 2,1,1,1,1 | `nuance_misread` | `retrieval_miss_in_law` | Chunk top dice "quota interessi rimborsata alle scadenze originarie" (= regular interest). Non c'è chunk esplicito su "interessi di mora se non pagati". Modello deduce assenza dal silenzio del chunk. |
| eval-0085 | L1 | 2,2,1,0,0 | `retrieval_miss_in_law` | `wrong_chunk_reranked` | Chunk top score 2 parlano di "diritto proporzionale" pagato alla Regione — non di cauzione ai proprietari. Articolo sulla cauzione non in contesto. Reranker ingannato dal contesto generico "concessionario deve pagare". |
| eval-0086 | L2 | 2,2,1,1,1 | `retrieval_miss_in_law` | `wrong_chunk_reranked` | Chunk top score 2 = "permesso di ricerca rilasciato dalla Giunta" + un articolo di altra legge sull'asportazione. Articolo su imbottigliamento acque minerali non recuperato. |
| eval-0088 | L4 | 2,0,0,0,0 | `retrieval_miss_in_law` | `wrong_chunk_reranked` | Top chunk score 2 = un comma di l.r.15/2014 sul "diritto proporzionale" — totalmente off-topic vs "ratio della legge". Articolo "Finalità" della legge non recuperato. |
| eval-0095 | L3 | 1,1,0,0,0 | `retrieval_miss_in_law` | `judge_calibration_issue` | Articolo su imposta di soggiorno non recuperato. Chunk in contesto sono su imposta sostitutiva (concetto diverso). Judge stesso ammette ambiguità ("typically governed by local municipal regulations…"). |
| eval-0098 | L2 | 2,1,1,1,1 | `chunk_intro_only` | `retrieval_miss_in_law` | Tutti e 5 i chunk in contesto sono titoli parentetici (`p:intro`): "(Indennizzo…)", "(Esenzione…)", "(Misure straordinarie…)". Articolo sul "differimento tributi" non recuperato. Il modello ha visto solo etichette. |
| eval-0099 | L3 | 2,1,0,0,0 | `nuance_misread` | — | Top chunk dice "imprese e liberi professionisti che accedono alle misure dell'art.3 della l.r. 4/2020 (Prime misure…sostegno per famiglie, lavoratori e imprese…)". La risposta corretta "famiglie, lavoratori, imprese" è dentro il *titolo della legge citata in parentesi*. Hard di estrazione. |

## Aggregati

### Conteggio per categoria primaria

| Categoria primaria | Conteggio | Quota su 22 |
|---|---|---|
| `retrieval_miss_in_law` | 11 | 50% |
| `nuance_misread` | 3 | 14% |
| `chunk_split_orphan` | 2 | 9% |
| `wrong_chunk_reranked` | 2 | 9% |
| `judge_calibration_issue` | 2 | 9% |
| `chunk_intro_only` | 1 | 5% |
| `improper_hedging` | 1 | 5% |
| `out_of_scope` | 1 | 5% |
| **Totale** | **22** | **100%** |

### Conteggio per "famiglia" del problema

Aggregando primaria + secondaria, la distribuzione delle famiglie causali (un caso può apparire in più famiglie):

| Famiglia | Casi | Quota |
|---|---|---|
| **Retrieval/expansion incompleto** (right law, wrong article) | 14 | 64% |
| **Chunking** (orphan intro o intro-only) | 3 | 14% |
| **Reranker miscalibrato** (score 2 a chunk non risolutivi) | 6 | 27% |
| **Generazione** (nuance/hedging/invention) | 7 | 32% |
| **Judge** (calibrazione discutibile) | 4 | 18% |
| **Out-of-scope corpus** | 2 | 9% |

I conteggi superano il 100% perché molti casi hanno cause multiple sovrapposte. La famiglia **retrieval/expansion incompleto** è la dominante: in 14 casi su 22 il problema è che la legge giusta è in contesto ma l'articolo o il comma con la risposta no. Questo include i casi `chunk_split_orphan` (la "lista" è in chunk fratelli mai recuperati) e i `chunk_intro_only` (i chunk recuperati sono titoli parentetici senza corpo).

## Tre scoperte importanti

### 1. La chunking strategy crea "orphan intro chunks"

Il chunker passa al livello passaggio (intro, c1, c1.lit_a, c1.lit_b, …). Quando un comma c1 introduce una lista (es. "1. Sono organi dell'azienda USL:"), quel chunk c1 è SEPARATO dai sotto-chunk lit_a, lit_b, lit_c che contengono le voci. Il dense embedder è felicissimo di matchare il chunk c1 perché contiene "organi" e "USL"; ma il chunk c1 da solo non risponde alla domanda. I chunk lit_a, lit_b sono ranking peggio perché sono testi atomici come "il direttore generale" senza contesto della domanda.

Si vede chiaramente in eval-0001 e eval-0027. È un problema **strutturale** del chunker, non del retrieval.

**Mitigazione possibile (no re-index)**: in fase di context build, quando un chunk è di tipo intro o termina con ":" e ha sotto-chunk fratelli (stesso `article_id`, prefisso passage), recuperare automaticamente i sotto-chunk fratelli nello stesso contesto. Questo è una micro-feature `sibling_expansion` nel runner advanced.

**Mitigazione strutturale (re-chunking)**: modificare la chunking strategy per emettere un chunk "completo" che contenga intro+lista quando la lista è breve, o aggiungere contesto sibling al `text_for_embedding` (già parzialmente fatto: il prefisso include label, ma non il testo dei fratelli).

### 2. Il reranker dà score 2 ai chunk-titolo

Casi come eval-0045 (chunk top = `(Piante officinali ad uso erboristico…)` — un titolo parentetico di articolo, non la definizione), eval-0098 (5 chunk tutti `p:intro`), eval-0024 e eval-0086 mostrano che il reranker LLM dà score 2 a chunk che matchano le keyword della domanda *anche quando il chunk è un titolo o un'introduzione vuota*. Il rubric attuale ("directly answers or strongly supports the answer") non distingue "titolo che annuncia un argomento" da "testo che enuncia la regola".

**Mitigazione**: nel prompt di rerank esplicitare:
- score 0 se il chunk è solo un titolo, una rubrica, o una lead-in che annuncia ma non contiene la risposta
- score 1 se cita o richiama la regola senza enunciarla
- score 2 solo se il chunk contiene il testo che risponde direttamente

Aggiungere 2 esempi few-shot del fallimento (chunk-titolo che non andrebbe scelto).

### 3. La maggioranza dei retrieval miss è "right law, wrong article"

In 11 casi su 22 (50%) la legge attesa è in `retrieved_law_ids` (`reference_law_hit=True`) ma l'articolo che contiene la risposta non è nei top-K chunk recuperati. Questo significa che la dense retrieval con `top_k=10` recupera spesso più chunk dello stesso articolo o articoli ridondanti, lasciando fuori articoli pertinenti dello stesso corpus normativo.

**Mitigazione 1 (no re-index)**: nel runner advanced, dopo la prima retrieval, se la legge è già "agganciata" (top-1 e top-2 condividono law_id), aumentare il top_k effettivo o fare una seconda search filtrata sulla stessa law_id ma con peso ai chunk di articoli diversi (article_id diversi).

**Mitigazione 2**: la graph expansion attuale espande SOLO cross-law (segue edges verso altre leggi). Aggiungere un'espansione **intra-law** che, dato un seed chunk in legge A, recuperi N chunk addizionali da articoli vicini della stessa legge A (es. ±2 articoli, oppure articoli che condividono il `structure_path` come "CAPO IV").

## Implicazioni per le priorità di intervento

L'audit cambia il ranking dei batch del piano originale:

| Batch piano originale | Nuova priorità | Ragione |
|---|---|---|
| **B4** (re-index bge-m3 hybrid) | **Salta a priorità ALTA** | Embedder attuale (`nomic-embed-text` 768-dim, English-leaning) sbaglia spesso il match articolo-articolo all'interno della stessa legge. bge-m3 multilingue + sparse mitiga lessico legale. |
| **B2** (graph expansion mirata) | **Estendere con `sibling_expansion` + `intra_law_expansion`** | Le mitigazioni 1 e 3 sopra sono nuove feature non previste nel piano. Vanno aggiunte al batch B2 o creare un nuovo B2-bis. |
| **B3** (rerank rinforzato) | **Priorità ALTA con few-shot anti-titolo** | La calibrazione sui chunk-titolo è un problema concreto e ricorrente, non marginale come stimato inizialmente. |
| **B1** (prompt no-hint riscritto) | **Priorità MEDIA, scope ridotto** | Solo 4-7 casi su 22 sono pure generation issues. Il prompt rinforzato vale comunque per quei casi (eval-0011, 0083, 0099, 0050) ma da solo non recupera la maggior parte dei fallimenti. |
| Nuovo: **B5 chunking** | **Da considerare** | Mitigazione strutturale di chunk_split_orphan (3 casi su 22 più diversi context_noise non auditati). Costo: re-chunk + re-index. |

## Avvertenze metodologiche per la tesi

1. **Judge calibration**: 2 dei 22 casi (eval-0051, eval-0060) sono falsi negativi dal judge. Sono ~9% del pool unknown. Sull'intero set di 100 record, il judge potrebbe avere un bias di rigore non trascurabile. Vale la pena ri-valutare 5-10 casi success a campione per capire se il judge è sistematicamente troppo severo o calibrato.

2. **Out-of-scope**: eval-0025 (e probabilmente eval-0095) sono domande la cui risposta sta fuori dal corpus regionale. Si potrebbe annotare nel dataset un campo `answerable_from_corpus: bool` per distinguere fallimenti del RAG da limiti del corpus.

3. **Reference law hit ingannevole**: `reference_law_hit=True` in 21 dei 22 casi unknown, ma metà hanno comunque un retrieval miss interno alla legge giusta. La metrica `reference_law_hit` è più ottimistica del vero. Considerare di aggiungere `reference_article_hit` se le `expected_references` includono il numero di articolo.

## File usati

- `data/rag_runs/advanced/full_100_dense_graph_rerank/no_hint_results.jsonl`
- `data/laws_dataset_clean/chunks.jsonl` (per il testo completo dei chunk in contesto)
- `data/rag_runs/advanced/full_100_dense_graph_rerank/mcq_results.jsonl` (per cross-check MCQ pari)

Dump intermedio per ispezione: `/tmp/unknown_audit_dump.md` (85 KB, 22 casi con domanda, risposta predetta, risposta corretta, judge_explanation, contesto chunk-by-chunk).
