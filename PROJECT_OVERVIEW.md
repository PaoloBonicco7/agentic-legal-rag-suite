# Implementation Summary and References

Questo lavoro verifica empiricamente se un retrieval più ricco — hybrid dense + sparse con query rewriting — migliora le risposte su un set di domande basate su un corpus di leggi regionali italiane rispetto a un Naive RAG dense-only e a un controllo no-RAG. Le sezioni 1–7 descrivono le scelte implementative dei sette step della pipeline e le motivano con la letteratura di riferimento; la sezione 8 raccoglie i risultati misurati nei notebook `06b_retrieval_diagnostics.ipynb` (diagnosi retrieval-only) e `06_advanced_graph_rag.ipynb` (verifica end-to-end sul dataset completo).

## 1. Preprocessing strutturale del corpus

### Problema

Il corpus sorgente contiene leggi regionali in HTML: leggibile per l'utente, non direttamente utilizzabile per retrieval, citazioni controllate o RAG graph-aware. Servono unità stabili, ispezionabili e recuperabili che mantengano identità normativa, struttura interna, provenienza e relazioni esplicite tra leggi.

### Approccio

Il preprocessing trasforma gli HTML in record JSONL strutturati — leggi, articoli, passaggi, note, relazioni esplicite, chunk — usando segnali deterministici (filename, heading, anchor, link, citazioni testuali) per ricostruire la struttura giuridica prima di applicare il chunking. Ogni chunk mantiene testo pulito, contesto per embedding (`text_for_embedding`), metadati di provenienza, stato normativo e relazioni esplicite. La scelta è deliberatamente *structure-aware*: prima si ricostruisce la struttura giuridica, poi si applica il chunking. Questo evita di trattare le leggi come testo piatto e permette ai passaggi recuperati di portare con sé contesto, citabilità e segnali utili per retrieval filtrato o graph-aware. La separazione `text` (visualizzato) vs `text_for_embedding` (indicizzato) lascia al modello embedding il contesto giuridico mantenendo al retrieval una fonte leggibile e filtrabile.

Il dataset risultante contiene 3.145 leggi, 17.774 articoli, 76.390 passaggi, 76.467 chunk e 35.159 edge, con quality gate e hash nel manifest. Il numero di chunk non è scelto a priori: deriva dalla granularità dei passaggi giuridici estratti e dalla finestra di chunking configurata a 600 parole con overlap di 80. Poiché la pipeline segmenta prima gli articoli in intro, commi e lettere, quasi tutti i passaggi sono già abbastanza corti da diventare un singolo chunk; solo 32 passaggi superano la finestra e vengono divisi in più chunk. La quasi equivalenza tra passaggi e chunk conferma quindi che il chunking finale serve soprattutto come salvaguardia sui passaggi lunghi, non come segmentazione cieca dell'intero HTML.

### Reference

- [LegalHTML: Semantic mark-up of legal acts using web technologies, 2023](https://www.sciencedirect.com/science/article/pii/S0267364923000985) — rappresentazione strutturata di atti normativi via HTML/semantic web.
  - **Notes**: Il paper propone di usare HTML esteso con markup giuridico, metadati semantici e supporto alla consolidazione per rappresentare struttura, semantica e visualizzazione degli atti in un unico documento. Nel progetto non è stato adottato lo standard LegalHTML completo; il contributo è stato sfruttato come riferimento concettuale per trattare gli HTML normativi come documenti strutturati, estraendo articoli, passaggi, riferimenti, stato normativo e provenienza prima del chunking e dell'indicizzazione.

- [Modelling Legislative Systems into Property Graphs to Enable Advanced Pattern Detection, 2024](https://arxiv.org/abs/2406.14935) — modellazione di leggi, articoli, citazioni, modifiche e abrogazioni come property graph.
  - **Notes**: Il paper propone di rappresentare il sistema legislativo come property graph, con leggi, articoli e allegati come nodi e relazioni come citazioni, modifiche e abrogazioni arricchite da proprietà. L'implementazione descritta usa Akoma Ntoso, uno standard XML per documenti legislativi molto più strutturato degli HTML regionali usati in questo lavoro, per estrarre entità, struttura e relazioni, e Neo4j per rendere interrogabili pattern complessi sul sistema normativo. Se il corpus sorgente fosse già disponibile in Akoma Ntoso, l'estrazione di articoli, commi, identificatori stabili, riferimenti e modifiche sarebbe più diretta; nel progetto, però, convertire gli HTML regionali in Akoma Ntoso avrebbe introdotto un lavoro preliminare troppo ampio rispetto all'obiettivo del progetto. Per questo l'approccio è stato sfruttato in forma semplificata: non è stato introdotto Neo4j né un property graph esterno, ma il preprocessing estrae edge espliciti tra norme direttamente dagli HTML e li conserva come metadati e candidate graph per la graph expansion sperimentale; i diagnostics hanno poi mostrato che questa espansione aggiungeva troppo rumore, quindi resta documentata ma disattivata nella configurazione finale.

- [Legal Chunking: Evaluating Methods for Effective Legal Text Retrieval, 2024](https://journals.sagepub.com/doi/10.3233/FAIA241255) — importanza del chunking nel dominio legale.
  - **Notes**: Il paper confronta simple splitting, recursive splitting e semantic chunking su un task di retrieval legale basato sul GDPR, mostrando che nessuna tecnica automatica produce in modo stabile chunk semanticamente molto rilevanti; anche il semantic chunking risulta debole e costoso quando non cattura bene struttura gerarchica, clausole annidate e dipendenze contestuali. Nel progetto questo risultato ha motivato una scelta più conservativa: non dividere il corpus con splitter generici, ma prima estrarre articoli, commi, lettere e passaggi dalla struttura HTML delle leggi regionali e usare il chunking a finestra solo come fallback sui passaggi troppo lunghi. La quasi equivalenza tra passaggi e chunk nel dataset conferma che il retrieval lavora soprattutto su unità normative già strutturate, non su segmenti arbitrari.

- [An Ontology-Driven Graph RAG for Legal Norms, 2026](https://journals.sagepub.com/doi/10.3233/FAIA251598) — RAG legale sensibile a struttura, temporalità e provenienza.
  - **Notes**: Il paper propone un SAT-Graph RAG, cioè un framework ontology-driven che rappresenta norme, componenti, versioni temporali, versioni linguistiche e azioni legislative per rendere interrogabili struttura, validità temporale, provenienza e causalità delle modifiche. L'obiettivo è superare il RAG piatto, che recupera testo senza sapere quale versione di una norma fosse valida in una certa data. Nel progetto questo approccio è stato sfruttato come riferimento architetturale per rendere espliciti struttura, stato normativo, relazioni e provenance nei record di preprocessing e nell'indice Qdrant; non è stata però implementata una vera ontologia temporale con versioni e Action node, e la componente graph-aware è rimasta sperimentale perché la graph expansion misurata nei diagnostics introduceva più rumore che beneficio.

- [Automatic semantic edge labeling over legal citation graphs, 2018](https://link.springer.com/article/10.1007/s10506-018-9217-1) — estrazione e classificazione di relazioni esplicite tra norme.
  - **Notes**: Il paper affronta il problema di costruire citation graph legali non solo rilevando i rinvii, ma assegnando a ogni edge un'etichetta semantica che descrive la funzione della citazione, ad esempio limitazione, eccezione, definizione, delega o modifica. L'approccio usa il testo circostante alla citazione per classificare lo scopo del collegamento, mostrando che nel dominio legale il tipo di relazione è importante quanto l'esistenza del link. Nel progetto questo principio è stato sfruttato in modo deterministico e più leggero: le citazioni e i link estratti dagli HTML vengono convertiti in edge con `relation_type` e confidenza, poi salvati nei record e nel payload dell'indice. Non è stato implementato un classificatore ML di edge labeling; le relazioni usate sono quelle riconoscibili con pattern e segnali strutturali affidabili, così da mantenere il preprocessing riproducibile e ispezionabile.

- [Retrieval-Augmented Generation for Large Language Models: A Survey, 2023/2024](https://arxiv.org/abs/2312.10997) — collocazione del preprocessing nella pipeline RAG.
  - **Notes**: Il survey organizza la letteratura RAG distinguendo Naive RAG, Advanced RAG e Modular RAG, e descrive la pipeline di base come indexing, retrieval e generation, con ottimizzazioni pre-retrieval e post-retrieval come metadata, query rewriting, hybrid retrieval, reranking e compressione del contesto. Nel progetto questo paper è stato usato come riferimento trasversale per posizionare i vari step: il preprocessing e Qdrant corrispondono alla fase di indexing, il Simple RAG alla baseline naive retrieve-read, mentre retrieval diagnostics e Advanced Graph RAG testano componenti advanced come hybrid dense+sparse, multi-query rewriting, graph expansion e reranking. È stato sfruttato soprattutto come cornice metodologica, non come algoritmo specifico.


## 2. Dataset di valutazione

### Problema

Le domande di valutazione nascono come MCQ. Dopo una prima analisi, il formato con opzioni è risultato troppo guidato per valutare bene la capacità dei sistemi RAG di generare risposte autonome; per questo è stata creata anche una versione no-hint, senza opzioni.

Lo step 2 non valuta ancora i modelli: prepara un benchmark stabile, allineando la versione MCQ originale e la versione no-hint derivata, così che tutti gli esperimenti successivi lavorino sulle stesse domande, risposte corrette e reference normative attese.

### Approccio

La pipeline normalizza i due CSV in `questions_mcq.jsonl` e `questions_no_hint.jsonl`, assegnando `qid` stabili e preservando livello, risposta corretta ed `expected_references`. I riferimenti restano stringhe leggibili in questo step; la risoluzione in identificatori di legge e articolo avviene solo negli step successivi.

Il risultato è un dataset pulito di 100 domande MCQ e 100 domande no-hint, bilanciato sui livelli L1-L4. Il manifest registra hash, conteggi e quality gate, rendendo il benchmark rigenerabile e confrontabile.

### Reference

- [LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models, 2023](https://arxiv.org/abs/2308.11462) — task legali costruiti con attenzione al ragionamento giuridico.
  - **Notes**: Il concetto ripreso da LegalBench è che un benchmark legale dovrebbe rendere esplicito quale capacità sta misurando, invece di trattare tutte le domande come semplice QA generico. Il paper costruisce task con il contributo di esperti e li organizza per forme di ragionamento giuridico; nel progetto questo principio viene riadattato in modo più semplice attraverso i livelli L1-L4 e la separazione tra formato MCQ e formato no-hint. LegalBench non viene usato come dataset perché lavora su task in inglese e non valuta direttamente il retrieval su un corpus normativo locale. Rimane però utile per motivare la struttura del benchmark: ogni domanda conserva livello, risposta corretta e riferimento atteso, così gli esperimenti successivi possono leggere i risultati non solo come accuratezza aggregata, ma anche rispetto alla difficoltà della domanda e al tipo di capacità richiesta.

- [LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain, 2024](https://arxiv.org/abs/2408.10343) — benchmark RAG legali con domande associate a riferimenti normativi attesi.
  - **Notes**: Il concetto ripreso da LegalBench-RAG è che un benchmark per Legal RAG non dovrebbe valutare solo la risposta finale, ma anche la capacità del sistema di recuperare evidenza giuridica precisa e citabile. Il paper costruisce coppie query-snippet e tratta i passaggi rilevanti come ground truth del retrieval, invece di fermarsi al documento corretto o a chunk ampi. Nel progetto questo principio viene riadattato al corpus di leggi regionali: lo step 2 conserva per ogni domanda gli `expected_references` testuali, che negli step successivi vengono risolti in legge e articolo attesi. Da qui derivano le metriche retrieval-only (`law_hit`, `article_hit`, `MRR`) e il vincolo end-to-end per cui le citazioni devono provenire dai chunk effettivamente passati al modello. Il dataset LegalBench-RAG non viene usato direttamente perché lavora su documenti legali in inglese, soprattutto contratti e privacy policies; viene però riprocessata la sua idea centrale di benchmark retrieval-aware, adattandola a normativa italiana e riferimenti articolo-legge.

- [A Reasoning-Focused Legal Retrieval Benchmark, 2025](https://arxiv.org/abs/2505.03970) — benchmark legal RAG realistici orientati a retrieval+QA.
  - **Notes**: Il concetto ripreso da questo paper è che le domande legali realistiche non sempre coincidono lessicalmente con il passaggio normativo che serve per rispondere. Nei benchmark Bar Exam QA e Housing Statute QA il retrieval deve quindi trovare evidenza utile per sostenere una risposta, non solo testo semanticamente vicino alla query. Nel progetto questa idea viene applicata in forma più semplice: il formato MCQ originale è mantenuto perché permette scoring deterministico, ma viene affiancato da una vista no-hint per verificare se il sistema riesce a produrre una risposta senza essere guidato dalle opzioni. Anche la divisione L1-L4 si legge in questa direzione: le domande più difficili dovrebbero richiedere un collegamento più robusto tra formulazione della domanda, fonte normativa e risposta. Il dataset del paper non viene usato perché riguarda il diritto statunitense; viene invece ripreso l'approccio di valutare retrieval e risposta come parti collegate dello stesso compito legale.

## 3. Indexing contract

### Problema

Lo step 1 produce chunk strutturati e citabili, ma non ancora interrogabili da una pipeline RAG. Per usarli nel retrieval serve un indice che conservi sia il testo da cercare sia le informazioni giuridiche necessarie a interpretare il risultato: legge, articolo, stato normativo, vista corrente o storica, provenienza e relazioni esplicite.

Il problema non è solo rendere i chunk cercabili per similarità semantica. Nel dominio giuridico una risposta può dipendere da parole molto precise, numeri di articolo, riferimenti normativi e vincoli di validità che un embedding dense può non rappresentare bene. L'indice deve quindi supportare ricerca semantica, segnale lessicale e filtri su metadati senza separare il testo recuperato dalla sua identità normativa.

### Approccio

Ogni chunk viene indicizzato in Qdrant come point stabile, con vettori dense e sparse e con un payload che mantiene i metadati prodotti dal preprocessing. Il testo usato per l'embedding è `text_for_embedding`, cioè il passaggio arricchito con il minimo contesto normativo necessario; il testo originale resta separato per visualizzazione, citazione e verifica.

L'embedder scelto è **BGE-M3**, perché produce con lo stesso modello una rappresentazione dense multilingue e una rappresentazione sparse lessicale. Questo permette di costruire una collection unica, `legal_chunks_bge_m3`, su cui eseguire retrieval hybrid dense+sparse senza mantenere due sistemi separati. La fusione dei ranking avviene con Reciprocal Rank Fusion tramite l'API nativa di Qdrant.

I payload index su campi come `law_id`, `law_status`, `index_views`, `article_id` e `relation_types` rendono possibile applicare vincoli giuridici quando l'esperimento lo richiede. Nella configurazione finale questi filtri non sono forzati di default, perché i diagnostics mostrano che possono escludere riferimenti storici presenti nella ground truth; restano però parte del contratto dell'indice e permettono esperimenti controllati su vigenza, struttura e relazioni.

L'output dello step è quindi un indice Qdrant riproducibile, persistente in locale, con chunk identificabili, payload ispezionabile e vettori dense+sparse pronti per gli step successivi. Il vecchio indice Utopia/Nomic dense-only resta come baseline storica; il miglioramento ottenuto dal re-index BGE-M3 è misurato in §8.1.

### Reference

- [M3-Embedding / BGE-M3, 2024](https://arxiv.org/abs/2402.03216) — modello multilingual, multi-granularity, multi-functionality con dense+sparse nello stesso encoder. Base di tutto il retrieval del progetto.
  - **Notes**: Il paper introduce BGE-M3 come modello multilingue capace di produrre, dalla stessa stringa, rappresentazioni dense, sparse e multi-vector. Nel progetto viene ripreso soprattutto il primo aspetto pratico: usare un unico encoder per coprire sia similarità semantica sia matching lessicale, evitando una pipeline separata BM25 + embedding. Questa scelta è particolarmente utile sul corpus italiano, dove il retrieval deve gestire sia formulazioni naturali delle domande sia lessico normativo, numeri di articolo e riferimenti testuali. Il progetto usa BGE-M3 in modalità locale per generare i named vector `dense` e `sparse` della collection Qdrant; non usa invece la componente multi-vector ColBERT-style, perché lo schema dell'indice resta volutamente più semplice. Il paper motiva quindi la scelta centrale dello step 3: costruire l'indice nuovo intorno a un embedder unico dense+sparse, poi misurare in §8.1 il salto rispetto al precedente indice dense-only.

- [Reciprocal Rank Fusion, 2009](https://doi.org/10.1145/1571941.1572114) — riferimento fondativo per la fusione RRF di ranking dense+sparse.
  - **Notes**: Il paper propone RRF come metodo semplice per fondere ranking diversi usando solo la posizione dei documenti, senza addestramento e senza normalizzare punteggi prodotti da modelli differenti. Nello step 3 il concetto viene ripreso per un caso preciso: combinare il ranking dense e quello sparse della stessa collection Qdrant. Il progetto non introduce pesi manuali tra semantica e lessico, perché avrebbero richiesto tuning ulteriore e avrebbero reso meno leggibile il confronto sperimentale; usa invece RRF come criterio standard, stabile e spiegabile. Il paper motiva quindi la scelta di rendere l'hybrid retrieval una proprietà dell'indice, non un post-processing ad hoc. Lo stesso principio viene poi riutilizzato negli step successivi per fondere risultati multi-query, ma in questa sezione serve soprattutto a giustificare la fusione dense+sparse.

- Qdrant documentation — [Indexing](https://qdrant.tech/documentation/manage-data/indexing/), [Hybrid Queries](https://qdrant.tech/documentation/search/hybrid-queries/), [Filtering](https://qdrant.tech/documentation/search/filtering/).
  - **Notes**: Queste reference non sono paper, ma documentano le API usate per costruire il contratto tecnico dell'indice. Da Qdrant vengono ripresi named vectors, sparse index e payload indexes per conservare `dense`, `sparse` e metadati giuridici nella stessa collection; `prefetch` + `RrfQuery` per eseguire hybrid retrieval dense+sparse lato Qdrant; e i filtri su payload per rappresentare vincoli che l'embedding non può esprimere in modo affidabile, come stato normativo, vista corrente/storica, legge o articolo. Nel progetto i filtri restano disponibili ma non obbligatori, perché i diagnostics mostrano che un vincolo statico sulla sola normativa vigente può escludere riferimenti storici presenti nella ground truth.

- [Hybrid Legal Norm Retrieval: Leveraging Knowledge Graphs and Textual Representations, 2024](https://doi.org/10.3233/FAIA241245) — retrieval giuridico ibrido che combina testo, BM25/transformer e conoscenza strutturale.
  - **Notes**: Il paper mostra che nel retrieval di norme legali segnali lessicali, semantici e strutturali risolvono problemi diversi: il lessico aiuta su riferimenti precisi e terminologia tecnica, il dense retrieval aiuta quando la domanda usa parole diverse dalla norma, la struttura aiuta quando contano collegamenti e gerarchie. Il progetto riprende questa impostazione in forma più compatta: non introduce BM25 separato né un knowledge graph esterno, ma usa la sparse nativa di BGE-M3 e conserva relazioni e metadati nel payload Qdrant. Il contributo del paper motiva quindi due scelte dello step 3: non affidarsi a un indice dense-only e non perdere la struttura giuridica prodotta dal preprocessing. Inoltre, nel caso testato da loro il grafo migliora solo di poco le performance.

- [Finding the Law, 2023](https://arxiv.org/abs/2301.12847) — nello statutory retrieval la struttura della legge conta quanto il testo.
  - **Notes**: Il paper affronta lo statutory retrieval e mostra che recuperare passaggi di legge non equivale a cercare testo generico: articoli vicini possono essere semanticamente simili ma avere funzione giuridica diversa. Il progetto riprende questa osservazione trasferendola nel contratto dell'indice: ogni chunk porta con sé identificatori di legge e articolo, stato, vista e contesto normativo minimo in `text_for_embedding`. Non viene implementato un retriever strutturale dedicato, ma la struttura viene resa disponibile sia al modello embedding sia ai filtri Qdrant. Il paper motiva quindi il collegamento tra step 1 e step 3: il preprocessing structure-aware sarebbe poco utile se l'indice appiattisse di nuovo i chunk in semplici stringhe vettorializzate.

- [LegalBench-RAG, 2024](https://arxiv.org/abs/2408.10343) — retrieval preciso di segmenti legali minimi e citabili.
  - **Notes**: LegalBench-RAG è già stato citato in §2 per motivare un benchmark con riferimenti attesi; nello step 3 viene richiamato per un aspetto diverso, cioè la granularità del retrieval. Il paper valuta sistemi RAG legali contro snippet precisi, non contro interi documenti, perché la generazione deve potersi appoggiare a fonti citabili e non a contesto generico. Il progetto adatta questo principio indicizzando chunk che corrispondono quasi sempre a passaggi normativi atomici, invece di indicizzare leggi intere o articoli troppo ampi. Non viene riusato il dataset del paper, ma il suo criterio metodologico sostiene la forma della collection: ogni point deve essere abbastanza piccolo da essere citabile e abbastanza ricco da conservare coerenza giuridica.

## 4. Baseline no-RAG

### Problema

Prima di introdurre il retrieval, serve capire quanto il modello riesca già a rispondere usando solo la propria conoscenza. Altrimenti il miglioramento di un sistema RAG sarebbe ambiguo: potrebbe dipendere dai chunk recuperati, ma anche dalla forza del modello, dal formato della domanda o dal prompt.

Lo step 4 risolve quindi un problema di controllo sperimentale. Definisce una soglia model-only sullo stesso benchmark che verrà poi usato da Simple RAG e Advanced RAG, così il valore aggiunto del retrieval può essere letto come differenza rispetto a una baseline esplicita.

### Approccio

Il modello viene valutato sui dataset puliti dello step 2 senza Qdrant, senza chunk recuperati e senza contesto normativo aggiunto. Il formato MCQ viene valutato in modo deterministico confrontando la label predetta con quella corretta; il formato no-hint produce invece una risposta aperta, poi giudicata con la stessa scala di valutazione `0-2` usata negli altri step.

Le chiamate usano output JSON strutturato e temperatura `0`, in modo da ridurre la variabilità interna della run. Gli output sono row-level results per MCQ e no-hint, un summary aggregato e un manifest con configurazione, modelli, versioni di prompt e hash dei dataset. Le metriche (`accuracy`, `coverage`, `strict_accuracy`, `mean_score`, `by_level`) mantengono lo stesso contratto degli step RAG successivi, quindi i risultati sono confrontabili senza trasformazioni ulteriori.

### Reference

- [LawBench: Benchmarking Legal Knowledge of Large Language Models, 2023](https://arxiv.org/abs/2309.16289) — benchmark legal LLM su conoscenza, comprensione e applicazione del diritto.
  - **Notes**: LawBench organizza la valutazione di vari LLM distinguendo tra conoscenza memorizzata, comprensione del testo giuridico e applicazione della norma. Il progetto non riusa il dataset, perché è centrato soprattutto sul diritto cinese e non sul corpus regionale italiano; riprende però l'idea metodologica di non trattare tutte le domande legali come un unico QA indistinto. Nello step 4 questa idea viene adattata in forma più semplice: il modello viene valutato prima senza retrieval, sullo stesso benchmark L1-L4 usato dagli step successivi. In questo modo la baseline misura quanto il modello sappia già riconoscere o produrre risposte giuridiche senza accesso alle fonti locali. La distinzione tra MCQ e no-hint traduce la stessa esigenza: separare il riconoscimento guidato dalle opzioni dalla capacità di formulare una risposta autonoma. 

- [GPT-4 Passes the Bar Exam, 2023/2024](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4389233) — valutazione zero-shot/model-only su compiti legali con componenti MCQ e risposta aperta.
  - **Notes**: Katz et al. valutano GPT-4 sul Uniform Bar Exam in modalità zero-shot, includendo sia la componente a scelta multipla sia prove a risposta aperta valutate con una griglia di valutazione. Da qui deriva la scelta dello step 4 di mantenere entrambi i formati del benchmark, MCQ e no-hint, invece di usare una sola metrica aggregata. L'MCQ misura il riconoscimento della risposta corretta in un setting guidato; il no-hint verifica se il modello riesce a produrre una risposta senza opzioni.

- [Re-evaluating GPT-4's Bar Exam Performance, 2023/2024](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4441311) — cautela e trasparenza metodologica nelle valutazioni legali.
  - **Notes**: Il paper mostra che il risultato di GPT-4 al Bar Exam è stato probabilmente sovrainterpretato perché il percentile dipende da scelte metodologiche poco trasparenti. Il punto è che una valutazione legal-LLM non basta se riporta solo un numero finale: deve chiarire dati, scoring, prompt, modello e limiti. Nella tesi questo principio viene applicato alla baseline no-RAG, che misura il comportamento model-only prima di introdurre il corpus. Per questo la run del progetto registra hash dei dataset, modelli, versioni di prompt, errori e metriche separate (`coverage`, `accuracy`, `strict_accuracy`), così il confronto con Simple RAG e Advanced RAG resta verificabile.

- [G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment, 2023](https://arxiv.org/abs/2303.16634) — giudice LLM con griglia di valutazione numerica per risposte aperte, con consapevolezza dei possibili bias.
  - **Notes**: G-Eval propone di usare un LLM come giudice per output testuali liberi, guidandolo con una griglia di valutazione esplicita invece di affidarsi a metriche lessicali come overlap o similarità superficiale. Questo concetto è usato direttamente nella valutazione no-hint dello step 4: la risposta aperta del modello viene confrontata con la risposta corretta attraverso un giudice LLM che assegna `0`, `1` o `2`. Il progetto semplifica il framework originale: non usa chain-of-thought esposto, non addestra un judge dedicato e mantiene una scala corta per ridurre ambiguità tra classi vicine. La scala di valutazione `0-2` è sufficiente per lo scopo della tesi, perché deve distinguere risposta sbagliata, parzialmente corretta e corretta in modo confrontabile tra no-RAG, Simple RAG e Advanced RAG. Il paper segnala anche bias del giudice LLM, come preferenza per risposte più lunghe o dipendenza dal modello usato. Solleva un altro punto interessante: i modelli come GPT-4 potrebbero avere un bias (un pregiudizio). In pratica, tendono a dare voti più alti ai testi scritti da altre IA rispetto a quelli scritti dagli umani, perché riconoscono e preferiscono il proprio stile o modo di ragionare.

## 5. Simple RAG

### Problema

La baseline no-RAG chiarisce quanto il modello riesce a rispondere da solo, ma non dice se una risposta sia fondata sulle norme del corpus. Nel dominio legale questo è un limite sostanziale: una risposta plausibile non basta, perché deve essere verificabile rispetto a passaggi normativi precisi.

Lo step 5 introduce quindi una baseline RAG minima. Serve a verificare che recupero, costruzione del contesto, generazione, citazioni e valutazione funzionino end-to-end prima di aggiungere hybrid retrieval, query rewriting, graph expansion o reranking. In questo modo il confronto successivo con l'Advanced RAG misura il valore delle componenti avanzate rispetto a un RAG semplice ma già completo.

### Approccio

Per ogni domanda la pipeline recupera un piccolo insieme di chunk con dense retrieval sull'indice Qdrant, costruisce un contesto limitato e chiede al modello di generare una risposta strutturata usando solo quei passaggi. Le citazioni sono vincolate ai `chunk_id` effettivamente inseriti nel contesto: se il modello cita un chunk non presente, la risposta viene tracciata come errore.

La valutazione resta allineata al no-RAG: scoring deterministico per le MCQ e scala di valutazione `0-2` con judge LLM per le domande no-hint. Lo step produce manifest, risultati JSONL a livello di singola domanda, summary aggregato e quality report. Questi output rendono leggibili sia i risultati complessivi sia i casi di fallimento, distinguendo retrieval miss, problemi di generazione ed errori di citazione.

La scelta metodologica è volutamente conservativa: lo step non cerca ancora il miglior retrieval possibile, ma una baseline chiara e controllabile. È utile nel disegno della tesi perché separa tre livelli di confronto: modello senza fonti (§4), RAG minimale (§5), RAG avanzato con componenti diagnosticate (§6-§7).

### Reference

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, 2020](https://arxiv.org/abs/2005.11401) — riferimento fondativo per il paradigma RAG.
  - **Notes**: Il paper introduce il paradigma retrieval → contesto → generazione condizionata, mostrando che un modello generativo può migliorare su task knowledge-intensive quando riceve passaggi recuperati da un corpus esterno. Nel progetto non viene ripresa l'architettura originale con DPR e BART né l'addestramento end-to-end; il principio viene adattato a una pipeline più semplice, in cui Qdrant recupera chunk normativi e un LLM Utopia risponde tramite prompt strutturato. 

- [Dense Passage Retrieval for Open-Domain QA, 2020](https://arxiv.org/abs/2004.04906) — motivazione per partire da dense vector retrieval.
  - **Notes**: DPR è il riferimento utile per trattare il dense retrieval come baseline neurale autonoma, non come scelta arbitraria. Il progetto ne riprende solo l'idea metodologica: una prima fase di recupero basata su embedding dense, senza BM25 separato, senza sparse vector e senza grafo. L'implementazione è diversa dal paper, perché usa l'indice Qdrant costruito nello step 3 e non un retriever DPR addestrato sul dataset; tuttavia la scelta di partire da una sola modalità di retrieval permette di costruire un confronto pulito. Lo step 6 potrà così misurare se hybrid retrieval, query rewriting e altre leve aggiungono valore rispetto a un dense baseline già riconoscibile in letteratura.

- [BGE M3-Embedding, 2024](https://arxiv.org/abs/2402.03216) — embedding multilingue adatto all'italiano normativo.
  - **Notes**: Il paper presenta BGE-M3 come un modello di embedding unico per tre esigenze che di solito richiedono componenti separate: recupero multilingue, gestione di testi di lunghezza diversa e produzione di rappresentazioni dense, sparse e multi-vector. Nel progetto è stato usato nella sua componente dense+sparse: la stessa stringa può essere rappresentata sia per similarità semantica sia per matching lessicale, caratteristica importante quando le domande sono in linguaggio naturale ma le fonti usano lessico normativo, numeri di articolo e formule ricorrenti. Nel progetto BGE-M3 viene usato come embedder dell'indice Qdrant; nello step 5, però, si usa solo il vettore dense, così il Simple RAG resta una baseline semantica minimale. La componente sparse viene lasciata allo step avanzato, dove serve a misurare il guadagno dell'hybrid retrieval.

- [Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools, 2024](https://arxiv.org/abs/2405.20362) — RAG riduce ma non elimina le allucinazioni; motiva citazioni vincolate.
  - **Notes**: Il paper mostra che anche strumenti legal AI basati su retrieval possono produrre allucinazioni, citazioni inventate o risposte non realmente supportate dai passaggi recuperati. Nel progetto questo risultato viene tradotto in una scelta concreta: una risposta RAG è valida solo se le citazioni puntano a chunk effettivamente presenti nel contesto della domanda. Non viene riusato il protocollo del paper, che valuta sistemi commerciali su query giuridiche diverse dal corpus di questa tesi; viene ripreso il principio di auditabilità. Per questo lo step 5 salva tracce row-level con chunk recuperati, chunk inseriti nel contesto e citazioni prodotte, rendendo possibile controllare se un errore nasce dal retrieval, dalla generazione o da una citazione non fondata.

- [LegalBench-RAG, 2024](https://arxiv.org/abs/2408.10343) — retrieval legale con snippet precisi e limiti di contesto.
  - **Notes**: LegalBench-RAG è rilevante qui per l'idea che un sistema legal RAG debba lavorare su evidenze piccole, precise e citabili, non su documenti interi o contesti indistinti. Lo step 5 adatta questo principio limitando il contesto a pochi chunk e tracciando separatamente ciò che viene recuperato e ciò che viene davvero passato al modello. Non basta recuperare la legge corretta in astratto; conta se il passaggio utile arriva nel contesto e può essere citato.

- [LRAGE: Legal Retrieval Augmented Generation Evaluation Tool, 2025](https://arxiv.org/abs/2504.01840) — valutazioni RAG legali che isolano corpus, retriever, generatore e metriche.
  - **Notes**: LRAGE è utile perché separa esplicitamente le componenti di una valutazione Legal RAG: corpus, retriever, generatore, judge e metriche. Lo step 5 riprende questa disciplina in forma semplificata: usa lo stesso corpus e lo stesso indice degli step precedenti, mantiene il retriever nella forma dense-only, usa la stessa valutazione del no-RAG e registra ogni configurazione nel manifest. Non viene riusato il tool LRAGE né il suo set di metriche; il progetto mantiene metriche e contratti propri, adatti al corpus italiano e alle reference articolo-legge. Il paper motiva però la scelta metodologica di avere un Simple RAG separato: senza una baseline con componenti fissate e tracciate, il confronto con l'Advanced RAG sarebbe difficile da attribuire al retrieval.

## 6. Retrieval diagnostics (notebook 06b)

### Problema

Il Simple RAG dello step 5 recupera pochi chunk con ricerca dense-only. Questa baseline è utile, ma lascia aperta una domanda metodologica: quali componenti avanzate migliorano davvero il retrieval e quali aggiungono solo costo, rumore o complessità?

Lo step 6 risponde a questa domanda prima della generazione finale. L'obiettivo non è ancora produrre risposte migliori, ma verificare se il sistema riesce a recuperare l'articolo normativo atteso, quanto in alto lo posiziona e quali leve meritano di passare allo step end-to-end. In questo modo l'Advanced RAG non nasce come somma di tecniche, ma come configurazione selezionata dalla diagnostica.

### Approccio

La diagnosi separa il retrieval dalla generazione. Per ogni configurazione misura tre segnali: `article_hit`, cioè presenza dell'articolo atteso tra i candidati; `law_hit`, cioè presenza almeno della legge corretta; `MRR`, cioè posizione del primo riferimento utile nel ranking. Le metriche sono calcolate sugli `expected_references` preparati nello step 2, quindi il confronto resta legato a evidenza normativa esplicita e non a una valutazione generica di similarità.

Il notebook confronta in modo incrementale le principali leve disponibili: aumento di `top_k` nella ricerca dense, filtri metadata, espansione tramite edge normativi, retrieval hybrid dense+sparse, reranking LLM e riformulazione della query. Ogni leva viene valutata contro la stessa baseline BGE-M3 dense, così il miglioramento è attribuibile alla componente testata e non a cambiamenti di corpus, modello embedding o dataset.

Il risultato è una selezione netta. L'hybrid retrieval viene promosso: rispetto a `dense@10` porta `article_hit` da 73% a 89% e migliora anche il ranking (`MRR` da 0.519 a 0.551). La strategia `multi_query` viene promossa sul pilot perché aggiunge un ulteriore guadagno rispetto all'hybrid puro e raggiunge `law_hit=100%` sul campione. I filtri statici su stato normativo vengono scartati perché escludono alcuni riferimenti storici presenti nella ground truth; la graph expansion viene scartata perché aggiunge quasi solo candidati non rilevanti; il reranking LLM migliora la precisione in alto ma perde recall e produce troppi errori di output strutturato.

Gli output principali sono `scenarios.csv`, le diagnostics a livello di domanda, le cache versionate per reranking e query rewriting, e `recommended_advanced_config.json`. Quest'ultimo è il passaggio operativo verso lo step 7: attiva hybrid retrieval e `multi_query`, lascia disattivati filtri metadata, graph expansion e reranking LLM, e rende tracciabile la ragione di ogni scelta.

### Reference

- [Reciprocal Rank Fusion, 2009](https://doi.org/10.1145/1571941.1572114) — fonda l'Esperimento F.
  - **Notes**: Il paper introduce RRF come metodo semplice per fondere ranking diversi usando la posizione dei risultati, non i punteggi grezzi prodotti da sistemi eterogenei. Nello step 6 questo concetto viene adattato al caso dense+sparse: la vista dense di BGE-M3 cattura similarità semantica, la vista sparse recupera termini normativi, numeri e formulazioni precise, e RRF combina i due ranking senza pesi addestrati. Non vengono riusati né Condorcet Fuse né learning-to-rank, perché avrebbero richiesto tuning aggiuntivo e avrebbero reso meno leggibile il waterfall. Il risultato dell'Esperimento F motiva la scelta: l'hybrid migliora sia la copertura dell'articolo atteso sia la posizione media dei risultati utili, quindi viene promosso come base dello step 7.

- [BGE M3-Embedding, 2024](https://arxiv.org/abs/2402.03216) — base dense+sparse di tutto il waterfall.
  - **Notes**: Il paper presenta BGE-M3 come modello multilingue capace di produrre rappresentazioni dense, sparse e multi-vector dalla stessa stringa. Nello step 6 viene usata soprattutto la doppia rappresentazione dense+sparse: il confronto resta controllato perché entrambe le viste derivano dallo stesso embedder e dallo stesso indice Qdrant. La componente multi-vector non viene usata, coerentemente con la scelta di mantenere l'indice più semplice. Il paper motiva quindi il disegno del waterfall: prima si misura la baseline dense BGE-M3, poi si verifica se aggiungere la vista sparse tramite RRF produce un miglioramento reale. Il salto rispetto al vecchio embedder è discusso separatamente in §8.1, così lo step 6 misura le leve di retrieval avanzato senza confonderle con il cambio di modello.

- [Query Rewriting for Retrieval-Augmented Large Language Models, 2023](https://arxiv.org/abs/2305.14283) — pattern rewrite-retrieve-read alla base dell'Esperimento H.
  - **Notes**: Il paper propone di inserire una fase di riformulazione prima del retrieval, perché la domanda dell'utente e il testo da recuperare spesso non condividono la stessa forma linguistica. Nello step 6 questo principio viene ripreso con un LLM rewriter congelato: non viene addestrato un modello dedicato, non viene usato reinforcement learning e non viene cambiato il retriever. La riformulazione è trattata come leva diagnostica isolata e riproducibile, con prompt e cache versionati. Il paper motiva quindi l'Esperimento H nel suo insieme: testare se una domanda legale, riscritta in forma più vicina al lessico normativo, recupera articoli che la query originale lascia fuori o mette troppo in basso.

- [Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE), 2023](https://aclanthology.org/2023.acl-long.99/) — strategia `hyde`.
  - **Notes**: HyDE genera un documento ipotetico a partire dalla query e usa l'embedding di quel testo per cercare documenti reali simili. Il progetto riprende questa idea come una delle strategie dell'Esperimento H, chiedendo al modello di produrre un passaggio normativo plausibile prima del retrieval. Non viene riusato il setup originale con Contriever né viene addestrato un retriever; HyDE è implementato come trasformazione della query dentro la stessa pipeline BGE-M3. La strategia è stata utile da testare ma non da promuovere: nel corpus di leggi regionali italiane il passaggio ipotetico può introdurre contenuti plausibili ma non ancorati alla norma specifica, aumentando il rumore. La nota del paper sulla natura "fake" del documento generato spiega bene perché questo rischio è particolarmente delicato in ambito legale.

- [DMQR-RAG: Diverse Multi-Query Rewriting for RAG, 2024](https://arxiv.org/abs/2411.13154) — multi-query rewriting e diversificazione delle formulazioni; motiva `multi_query` come strategia preferita.
  - **Notes**: DMQR-RAG estende il rewriting singolo generando più riformulazioni, con l'idea che query diverse ma coerenti possano coprire parti diverse del vocabolario del corpus. Lo step 6 adatta questo concetto in forma compatta: usa `n=3` riformulazioni generate gpt-oss, poi confronta la strategia contro query originale, rewrite singolo e HyDE. Non viene riusata la selezione adattiva delle strategie proposta dal paper, perché il progetto ha bisogno di una configurazione semplice e ripetibile. Il risultato del pilot rende il collegamento concreto: `multi_query` è la strategia di query rewriting che produce il miglior guadagno sopra l'hybrid puro, quindi viene promossa nella configurazione raccomandata. Il paper motiva anche il rischio da controllare nello step 7: più varianti aumentano recall, ma richiedono una fusione attenta per non portare troppi distrattori nel contesto.

- [RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs, 2024](https://arxiv.org/abs/2407.02485) — LLM reranking come componente da misurare prima di ridurre il contesto.
  - **Notes**: RankRAG mostra che ranking del contesto e generazione possono essere trattati come funzioni collegate dentro una pipeline RAG. Il progetto non riprende il framework completo: non addestra un LLM multi-task e non integra ranking e risposta nello stesso modello. Ne riprende però l'idea metodologica da verificare: prima di tagliare il contesto, un LLM può stimare quali candidati siano più rilevanti. L'Esperimento G implementa questa idea in modo leggero, con scoring `0-2` sui candidati hybrid. Il risultato è negativo per la pipeline finale: il reranking migliora la precisione dei primi risultati, ma non recupera articoli che l'hybrid non aveva già portato nei candidati utili, riduce la recall e introduce errori di output strutturato. Il paper resta quindi utile non per promuovere il rerank LLM, ma per motivare perché fosse una leva ragionevole da misurare e perché la sua esclusione dallo step 7 sia una decisione sperimentale.

- [Hybrid Legal Norm Retrieval, 2024](https://doi.org/10.3233/FAIA241245) — combinare segnali lessicali, semantici e conoscenza strutturata.
  - **Notes**: Il paper combina testo, modelli di retrieval e conoscenza strutturata per il recupero di norme giuridiche, mostrando che segnali lessicali, semantici e relazionali possono essere complementari. Nel progetto questo schema viene semplificato: non vengono usati BM25 separato, precedenti o knowledge graph esterno; la parte lessicale è coperta dalla sparse di BGE-M3, mentre la parte strutturale è rappresentata dagli edge estratti nel preprocessing. Lo step 6 riprende quindi dal paper due domande sperimentali: l'hybrid dense+sparse aiuta davvero? seguire gli edge normativi migliora il candidate set? I risultati separano le due risposte: l'hybrid viene promosso, la graph expansion viene scartata per rumore. La citazione è utile perché motiva il test congiunto di segnali testuali e strutturali, ma anche perché chiarisce il limite dell'adattamento: un grafo usato come semplice espansione post-retrieval non equivale a una pipeline knowledge-graph pienamente integrata.

- [LegalBench-RAG, 2024](https://arxiv.org/abs/2408.10343) — retrieval di snippet legali minimi e citabili; criterio retrieval-only.
  - **Notes**: LegalBench-RAG distingue esplicitamente la valutazione del retrieval dalla valutazione della risposta generata e insiste su segmenti legali piccoli, precisi e citabili. Lo step 6 adatta questo principio al corpus regionale italiano: non usa il dataset LegalBench-RAG, ma valuta ogni configurazione contro gli `expected_references` dello step 2. La metrica centrale diventa `article_hit`, perché recuperare solo la legge corretta non basta se la risposta deve citare un passaggio normativo specifico. `law_hit` resta come segnale più debole e `MRR` misura quanto presto compare il riferimento utile. Il paper motiva quindi la separazione tra §6 e §7: un retrieval buono non garantisce una risposta corretta, ma senza retrieval diagnostico non sarebbe possibile capire se un fallimento nasce dal recupero o dalla generazione.

- [An Ontology-Driven Graph RAG for Legal Norms, 2026](https://journals.sagepub.com/doi/10.3233/FAIA251598) — Graph RAG legale structure-aware; framework per cui la graph expansion va vincolata e non usata in modo indiscriminato.
  - **Notes**: Il paper propone un Graph RAG giuridico in cui struttura gerarchica, versioni temporali, provenienza e azioni legislative sono modellate esplicitamente. Lo step 6 non implementa questa architettura: non costruisce una vera ontologia, non modella versioni temporali come nodi separati e non usa Action node per rappresentare le modifiche. Riprende solo l'idea minima di non trattare i chunk come testo isolato, testando se gli edge normativi estratti nello step 1 possano espandere il retrieval. L'esito negativo della graph expansion non smentisce il paper; mostra piuttosto che una semplice espansione post-retrieval è troppo debole e rumorosa per sostituire un grafo giuridico vincolato da struttura, tempo e causalità. La citazione resta utile perché spiega perché gli edge vengono comunque conservati nel payload e perché la componente graph rimane una direzione di Future Work, non una parte attiva della configurazione finale.

## 7. Advanced Graph RAG end-to-end (notebook 06)

### Problema

Lo step 6 ha mostrato quali tecniche migliorano il retrieval prima della generazione, ma quel risultato non basta: recuperare meglio non significa automaticamente rispondere meglio. Lo step 7 verifica quindi se le componenti promosse dai diagnostics producono un miglioramento reale sul benchmark completo, confrontandole con il Simple RAG sullo stesso corpus, sulle stesse domande e con le stesse metriche.

Il problema principale è evitare una pipeline avanzata solo in apparenza. Ogni componente deve portare un beneficio misurabile oppure restare fuori dalla configurazione finale. Per questo lo step non prova a usare tutte le tecniche disponibili, ma seleziona solo quelle che hanno mostrato un segnale utile nella fase diagnostica.

### Approccio

La configurazione finale modifica il Simple RAG soprattutto nella fase di recupero. Usa un retrieval ibrido, che combina il segnale semantico degli embedding dense con il segnale lessicale della componente sparse, e applica una strategia multi-query: la domanda viene riformulata in più modi, i risultati vengono recuperati separatamente e poi uniti in un unico ranking.

L'idea è semplice: una domanda giuridica può non usare le stesse parole della norma che serve per rispondere. Il retrieval ibrido aiuta quando contano termini normativi precisi, numeri di articolo o formule ricorrenti; il multi-query aiuta quando una sola formulazione della domanda non basta a raggiungere il passaggio corretto. Il contesto passato al modello resta comunque limitato e citabile: non basta recuperare molti candidati, bisogna portare nel prompt pochi chunk utili.

Altre componenti sono state escluse dalla configurazione finale. I filtri statici sui metadati rischiavano di eliminare riferimenti storici presenti nella ground truth; la graph expansion aggiungeva quasi solo rumore; il reranking LLM non recuperava abbastanza chunk utili da giustificarne costo e instabilità. Restano quindi disponibili come varianti sperimentali, ma non fanno parte del risultato principale.

Lo step produce gli stessi output confrontabili degli esperimenti precedenti — risultati MCQ, risultati no-hint, summary aggregato e diagnostica delle failure — ma aggiunge una lettura più fine del punto in cui la pipeline perde informazione: chunk recuperati, chunk effettivamente inseriti nel contesto, risposta finale e citazioni.

### Risultati principali

La configurazione finale migliora il Simple RAG, ma in modo più contenuto rispetto all'aumento osservato nel solo retrieval. Sul benchmark completo l'Advanced RAG passa da 0.79 a 0.84 di accuracy MCQ, da 0.59 a 0.63 di accuracy no-hint e da 1.18 a 1.26 di mean score no-hint. Il guadagno più netto è sulle domande L4, dove il no-hint migliora di 26 punti, mentre L3 peggiora rispetto al Simple RAG.

Il dato più importante è il divario tra recupero e contesto. La configurazione finale trova l'articolo atteso nel 93% dei casi tra i candidati recuperati, ma solo nel 72% dei casi tra i chunk davvero passati al modello. Questo significa che il limite principale non è più solo trovare la norma corretta: spesso la norma viene trovata, ma resta troppo in basso nel ranking o viene esclusa dal contesto finale.

La conclusione dello step è quindi doppia. Hybrid retrieval e multi-query sono utili e migliorano il risultato end-to-end; tuttavia non bastano da soli a trasformare un buon retrieval in una risposta sempre corretta. Il collo di bottiglia residuo è la selezione del contesto: bisogna distinguere meglio i chunk davvero risolutivi dai chunk solo vicini, introduttivi o ridondanti.

### Reference

- [Lost in the Middle: How Language Models Use Long Contexts, 2023](https://arxiv.org/abs/2307.03172) — spiega perché un contesto più ampio non garantisce risposte migliori.
  - **Notes**: Il paper mostra che aumentare il contesto non garantisce che il modello usi meglio l'informazione: i passaggi rilevanti possono essere ignorati se finiscono in mezzo a molti documenti simili o ridondanti. Nel progetto questo concetto motiva la separazione tra candidati recuperati e chunk effettivamente passati al modello. Il risultato di §8.3 conferma il problema: il 93% di article hit tra i candidati scende al 72% nel contesto finale. Il paper non viene riusato come tecnica, ma come chiave interpretativa del limite osservato: il collo di bottiglia non è solo la copertura del retrieval, ma l'ordine e la selezione dei chunk che arrivano al generatore.

- [LegalBench-RAG, 2024](https://arxiv.org/abs/2408.10343) — retrieval di snippet legali minimi e citabili; vincola le citazioni ai chunk del contesto.
  - **Notes**: Il contributo ripreso è la distinzione tra recuperare documenti genericamente pertinenti e recuperare evidenza legale minima, precisa e citabile. Nello step 7 questa idea motiva la doppia lettura dei risultati: non basta sapere se l'articolo atteso è tra i candidati recuperati, bisogna sapere se entra davvero nel contesto usato per rispondere. Per questo il gap 93% → 72% è centrale nella discussione: misura la perdita tra retrieval ampio e grounding effettivo della risposta. Il dataset del paper non viene riusato, ma il criterio di valutazione retrieval-aware viene adattato al corpus normativo italiano.

- [LRAGE, 2025](https://arxiv.org/abs/2504.01840) — valutazione legale che isola corpus, retriever, reranker, generatore e metriche.
  - **Notes**: LRAGE è utile per la disciplina sperimentale: separare corpus, retriever, generatore, judge e metriche permette di capire quale parte della pipeline sta contribuendo al risultato. Lo step 7 riprende questa impostazione confrontando Advanced RAG e Simple RAG a parità di dataset, indice e contratto di valutazione. Non viene riusato il tool del paper; viene ripreso il principio di tracciabilità delle configurazioni e dei risultati, necessario per leggere il miglioramento finale come effetto delle componenti promosse dai diagnostics e non come variazione non controllata.

- [Hallucination-Free?, 2024](https://arxiv.org/abs/2405.20362) — RAG riduce ma non elimina le allucinazioni; trace row-level e citazioni vincolate sono richieste.
  - **Notes**: Il paper mostra che anche strumenti legali con retrieval possono produrre risposte non supportate o citazioni non affidabili. Nello step 7 questo principio motiva due scelte: mantenere le citazioni legate ai chunk effettivamente usati come contesto e analizzare i fallimenti invece di limitarsi a una metrica aggregata. La tesi non riusa il protocollo del paper, che riguarda prodotti commerciali e casi d'uso diversi; ne riprende però l'esigenza di auditabilità. Anche quando l'Advanced RAG migliora il risultato medio, i casi di errore restano parte dell'evidenza sperimentale.

## 8. Risultati

### 8.1 Embedder upgrade (step 3)

Confronto retrieval-only sullo stesso corpus e stesse 100 domande, vecchio indice Utopia/Nomic dense-only vs nuovo indice BGE-M3 dense+sparse (fonte: `docs/results/06b_embedding_index_comparison.md`).

| indice | configurazione | article_hit | law_hit | MRR |
|---|---|---:|---:|---:|
| Utopia/Nomic dense | top_k=10 | 45.0% | 72.0% | 0.288 |
| Utopia/Nomic dense | top_k=100 | 66.0% | 87.0% | 0.295 |
| **BGE-M3 dense** | **top_k=10** | **73.0%** | **97.0%** | **0.519** |
| BGE-M3 dense | top_k=100 | 88.0% | 99.0% | 0.524 |
| **BGE-M3 hybrid** | top_k=100, rrf_k=30 | **89.0%** | 99.0% | **0.551** |

Il re-index BGE-M3 porta +28pp di `article_hit@10` e +0.231 di MRR rispetto a Nomic dense@10, prima ancora di attivare hybrid. L'attivazione hybrid aggiunge +16pp di `article_hit` sopra la nuova baseline dense@10 e +0.032 di MRR. Il salto giustifica il cambio di embedder come scelta architetturale: senza BGE-M3 le altre leve della pipeline (multi-query, fix F1/F2/F3) avrebbero un effetto trascurabile.

### 8.2 Retrieval diagnostics (step 6, notebook 06b)

Waterfall completo della run `diagnostic_full_utopia_throttled__20260525T091921Z`. Baseline = `dense@10` su 100 domande; LLM per G e H = `SLURM.gpt-oss:120b` via Utopia.

| stage | scenario | n | article_hit | law_hit | MRR | δ vs baseline | esito |
|---|---|---:|---:|---:|---:|---:|---|
| baseline | Dense@10 (no filter) | 100 | 73.0% | 97.0% | 0.519 | — | riferimento |
| direct ceiling | Dense top_k=100 | 100 | 88.0% | 99.0% | 0.524 | +15.0pp | informativo |
| B — filter | + Filter `law_status=current` | 100 | 71.0% | 93.0% | 0.507 | −2.0pp | scartato |
| C — graph | + Best graph from sweep | 100 | 74.0% | 97.0% | 0.520 | +1.0pp | scartato (noise=99.7%) |
| **F — hybrid** | **Hybrid top_k=100, rrf_k=30** | 100 | **89.0%** | 99.0% | **0.551** | **+16.0pp** | **promosso** |
| G — rerank | + LLM rerank (input_k=20, output_k=10) | 23 (pilot) | 82.6% | 100.0% | 0.711 | +9.6pp | scartato (−6.4pp vs hybrid) |
| **H — rewriting** | **+ multi-query (n=3)** | 30 (pilot) | **93.3%** | 100.0% | 0.467 | **+20.3pp** | **promosso (+4.3pp vs hybrid)** |

Sintesi degli esiti:

- **Hybrid (F) promosso**: +16pp `article_hit` sopra dense@10 con MRR che cresce da 0.519 a 0.551. Il guadagno non è solo recall: dense@100 raggiunge 88% di hit ma resta a MRR 0.524 — RRF promuove in cima i chunk che entrambe le viste (dense e sparse) considerano rilevanti.
- **Multi-query (H) promossa**: +4.3pp sopra hybrid puro sul pilot di 30 domande, con `law_hit=100%`. Le riformulazioni alternative coprono meglio la varietà del vocabolario normativo della stessa intent informativa.
- **Filtri metadata (B) scartati**: `law_status=current` esclude 4 domande dell'evaluation set i cui riferimenti puntano a leggi abrogate, abbassando il hit di 2pp.
- **Graph expansion (C) scartata**: la best config ha `expansion_noise_ratio=0.997` — oltre il 99% dei chunk aggiunti non è rilevante.
- **LLM reranking (G) scartato**: `recovered=0` su tutte le 18 configurazioni pilot — l'LLM non promuove mai in `output_k` un articolo che hybrid aveva escluso da `input_k`. Migliora la precision-at-top (MRR 0.71 vs 0.55) ma a costo della recall e con `failure_rate=30.6%` sull'output strutturato.

Configurazione promossa (`recommended_advanced_config.json` della run):

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

### 8.3 End-to-end Advanced RAG (step 7, notebook 06)

Run finale `full_100__answer_slurm_gpt_oss_120b__judge_slurm_gpt_oss_120b__a4_combined_best_v2` — variante A4 (F1+F2+F3 combinati): `top_k=100`, `rrf_k=60`, `max_context_chunks=15`, multi-query attiva con n=3.

Confronto sui 200 record di valutazione (100 MCQ + 100 no_hint):

| metrica | no-RAG | Simple RAG | Advanced (A4) | δ vs Simple | δ vs no-RAG |
|---|---:|---:|---:|---:|---:|
| MCQ accuracy / strict | 0.81 | 0.79 | **0.84** | **+0.05** | +0.03 |
| no_hint accuracy | 0.485 | 0.59 | **0.63** | **+0.04** | +0.145 |
| no_hint mean score (0–2) | 0.97 | 1.18 | **1.26** | **+0.08** | +0.29 |

Strict accuracy per livello di difficoltà (no_hint):

| livello | no-RAG | Simple RAG | Advanced (A4) | δ vs Simple |
|---|---:|---:|---:|---:|
| L1 | 0.34 | 0.66 | **0.68** | +0.02 |
| L2 | 0.54 | 0.68 | **0.72** | +0.04 |
| L3 | 0.46 | **0.64** | 0.48 | **−0.16** |
| L4 | 0.60 | 0.38 | **0.64** | **+0.26** |

Diagnostics retrieval (Advanced A4, su 200 domande):

| metrica | valore |
|---|---:|
| `reference_article_hit` retrieved (top_k=100, post multi-query fusion) | 93.0% (186/200) |
| `reference_article_hit` in context (top 15 chunk) | 72.0% (144/200) |
| `reference_law_hit` retrieved | 95.0% (190/200) |
| `context_sufficient` (no_hint, judge) | 86/100 |

Failure categories (Advanced A4, 200 domande):

| categoria | conteggio |
|---|---:|
| nessun fallimento (`none`) | 147 |
| `unknown` (risposta non riconducibile a categoria specifica) | 43 |
| `context_noise` | 5 |
| `generation_error` (errore di plumbing) | 4 |
| `abstention` | 1 |
| `citation_error: invalid_chunk_ids` (errori operativi) | 4 (3 MCQ + 1 no_hint) |

Cache multi-query: 200 hit / 0 miss / 0 failure (cache versionata interamente riutilizzata dalla pipeline 06b).

Funnel di perdita tra retrieval, context selection e risposta:

| stato della pipeline | conteggio | interpretazione |
|---|---:|---|
| articolo atteso non recuperato nei top-100 | 14/200 | limite residuo del retrieval anche dopo hybrid + multi-query |
| articolo atteso recuperato ma tagliato fuori dal contesto top-15 | 42/200 | collo di bottiglia principale: `hit@100` positivo, ma il generatore non vede il chunk |
| articolo atteso nel contesto ma risposta non strict-correct | 27/200 | limite di estrazione/generazione, chunk incompleti o judge severo |
| articolo atteso nel contesto e risposta strict-correct | 117/200 | successo end-to-end completo |

La distribuzione del primo chunk appartenente all'articolo atteso conferma il punto: 116/200 record hanno l'articolo atteso entro rank 1-3, 28/200 entro rank 4-15, 42/200 solo tra rank 16-100 e 14/200 mai nei top-100. La metrica retrieval-only di 06b conta come successo sia rank 3 sia rank 90; la pipeline end-to-end, invece, usa solo i primi 15 chunk. Per questo `reference_article_hit_retrieved=93%` non può essere letto come probabilità che il modello abbia davvero l'evidenza risolutiva nel prompt.

La differenza MCQ/no_hint mostra quanto questo collo di bottiglia pesi sulla generazione. Quando l'articolo atteso è nel contesto, MCQ sale a 91.7% e no_hint a 73.6% di accuracy; quando non è nel contesto, MCQ resta comunque a 64.3% grazie a segnali parziali o scelta per esclusione, mentre no_hint scende a 35.7%. Quindi la performance MCQ può mascherare una debolezza di grounding che diventa evidente nelle risposte aperte.

I 43 casi `unknown` vanno letti in questa luce: non indicano solo "errore generativo" generico. Una parte rilevante deriva da articolo o comma corretti recuperati troppo in basso, chunk dell'articolo giusto ma non risolutivi, oppure chunk introduttivi/di rubrica che attivano `reference_article_hit_context=true` senza contenere davvero la risposta. Anche `context_sufficient` è ottimistico: su no_hint il modello marca 86 contesti come `yes`, ma 22 di questi ricevono comunque judge score 0. Il limite residuo è quindi soprattutto di **context selection e answer-bearing evidence**, non di pura copertura top-100.

### 8.4 Letture chiave

**Cosa funziona.** Sul totale, Advanced batte sia Simple RAG sia no-RAG: +5pp MCQ accuracy, +4pp no_hint accuracy, +0.08 mean score no_hint. Il salto più consistente è su L4 no_hint (+26pp vs Simple), il livello a maggiore richiesta di ragionamento giuridico: hybrid + multi-query portano nel contesto i passaggi normativi che Simple RAG con top_k=3 lasciava fuori. A livello retrieval, il `reference_article_hit` retrieved del 93% e il `reference_law_hit` del 95% confermano che la candidate list — dopo fusione cross-query e RRF — copre quasi tutta la ground truth: la diagnosi di 06b si trasferisce end-to-end.

**Cosa non funziona ancora.** L3 no_hint regredisce di 16pp rispetto a Simple RAG (0.48 vs 0.64). Il gap tra `article_hit` retrieved (93%) e `article_hit` in context (72%) indica che 21pp di chunk corretti sono nei 100 candidati ma cadono fuori dai 15 nel contesto: F3 (`max_context_chunks=15`) ha attenuato il problema ma non l'ha risolto. La regressione è concentrata su L3, dove i chunk rilevanti competono con distrattori semanticamente vicini che hybrid non separa. In più, `article_hit_context` è ancora una metrica ottimistica: se il chunk incluso è solo una rubrica, un'introduzione tipo "Sono organi..." o un comma che rinvia ad altri sotto-chunk, l'articolo è formalmente presente ma la risposta non è davvero estraibile. Questo spiega perché 27 record con articolo atteso nel contesto non siano strict-correct.

**Gap rispetto ai target della roadmap.** I target dichiarati erano MCQ +10pp, no_hint judge score +5pp, `reference_article_hit` +20pp, `reference_law_hit` +15pp vs Simple RAG. I risultati sono parzialmente raggiunti: il target retrieval-only su `article_hit` è centrato (93% retrieved corrisponde a +20pp sopra il `dense@10=73%` baseline di 06b), ma i delta downstream sono più contenuti (MCQ +5pp invece di +10pp, no_hint mean score +0.08 invece di +5pp sul 200-score, corrispondente a circa +4pp). Il significato è quello atteso dalla letteratura su Lost in the Middle: un retrieval migliore non si traduce automaticamente in risposte migliori se il context window mette al margine i chunk rilevanti, e una riformulazione multi-query introduce distrattori che il generatore non sempre disambigua. Le direzioni di Future Work sono quindi mirate: (1) cross-encoder locale o reranker dedicato prima del taglio a 15, non LLM rerank zero-shot già scartato; (2) MMR/diversity cap per `law_id` e `article_id`, così da evitare che varianti dello stesso tema saturino il contesto; (3) sibling expansion per chunk introduttivi e liste spezzate; (4) metrica aggiuntiva di `answer-bearing_chunk_hit`, più severa di `article_hit`, per distinguere articolo formalmente presente da evidenza realmente sufficiente.

## Bibliografia di riferimento

I paper che sostengono più scelte della pipeline e meritano lettura preliminare prima della stesura finale:

- [BGE M3-Embedding, 2024](https://arxiv.org/abs/2402.03216) — embedder multilingue dense+sparse. Step 3, 5, 6 (base di tutto il retrieval).
- [Reciprocal Rank Fusion, 2009](https://doi.org/10.1145/1571941.1572114) — fusione di ranking eterogenei. Step 3, 6 (Esperimento F).
- [LegalBench-RAG, 2024](https://arxiv.org/abs/2408.10343) — benchmark RAG legale, snippet minimi e citazioni vincolate. Step 2, 3, 4, 5, 6, 7.
- [Query Rewriting for Retrieval-Augmented Large Language Models, 2023](https://arxiv.org/abs/2305.14283) + [DMQR-RAG, 2024](https://arxiv.org/abs/2411.13154) — pattern rewrite-retrieve-read e multi-query. Step 6 (Esperimento H).
- [Lost in the Middle, 2023](https://arxiv.org/abs/2307.03172) — i modelli usano peggio i contesti lunghi. Step 7 (motiva il controllo sul contesto finale).
- [Hallucination-Free?, 2024](https://arxiv.org/abs/2405.20362) — RAG riduce ma non elimina le allucinazioni. Step 5, 7 (motiva trace row-level e citazioni vincolate).
- [An Ontology-Driven Graph RAG for Legal Norms, 2026](https://journals.sagepub.com/doi/10.3233/FAIA251598) — framework di riferimento per Graph RAG legale. Step 1, 6 (Esperimento C; motiva perché in questo progetto il graph resta off ma è esposto per Future Work).

Per le direzioni di **Future Work** (§8.4) restano utili due riferimenti dedicati che non rientrano nella lista *top picks* e non motivano la configurazione finale di §7: [MMR — Carbonell & Goldstein, 1998](https://dl.acm.org/doi/10.1145/290941.291025) per la diversificazione del contesto (cap per legge), e [BGE-Reranker / C-Pack, 2024](https://arxiv.org/abs/2309.07597) come alternativa cross-encoder al LLM reranking scartato.
