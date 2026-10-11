# Revisione della strategia di retrieval — 11 ottobre 2026

**Il RAG migliora il risultato osservato rispetto al No-RAG; non abbiamo ancora
dimostrato che combinare grafo, multi-query e reranking migliori Hybrid.**
Hybrid è il riferimento operativo più semplice da cui ripartire. Le modifiche
sono pianificate nella [specifica 11](../specs/11_retrieval_optimization.md), non
implementate. Non sono state effettuate nuove chiamate a pagamento per questa analisi.

## Cosa dimostrano i 20 test

Generatore `gpt-6-luna`, giudice primario `gpt-5.6-terra`, cinque domande per livello.
Denominatore fisso 20; errori di generazione inclusi come insuccessi. Gold legacy
invariati. La [nota del pilot](09_terra20_pilot.md) conserva protocollo e risultati.

| Profilo | Risposte pienamente corrette | Errori di generazione/citazione |
|---|---:|---:|
| No-RAG | 9/20 — 45% | 0 |
| Dense | 16/20 — 80% | 2 |
| Sparse | 14/20 — 70% | 4 |
| Hybrid | 17/20 — 85% | 1 |
| Multi-query | 15/20 — 75% | 1 |
| Graph | 17/20 — 85% | 2 |
| Rerank | 17/20 — 85% | 1 |

Hybrid–No-RAG: **+40 punti percentuali**, IC95% appaiato +15/+65. Hybrid–Dense:
**+5 punti**, IC95% 0/+15, una sola domanda passa a pienamente corretta. Questa
domanda, Q0046, aveva già l'articolo atteso nei due contesti: non prova un nuovo
recupero di evidenza. Graph mantiene gli stessi esiti pienamente corretti di Hybrid;
Rerank guadagna Q0100 ma perde Q0046; Multi-query guadagna Q0100 ma perde Q0016,
Q0036 e Q0046. I giudizi su Q0046/Q0100 presentano incoerenze da revisionare.

Le risposte giudicate contemporaneamente corrette e supportate sono 16/20 Dense,
16/20 Hybrid, 17/20 Graph e 15/20 Rerank. Nei 20 quesiti l'articolo atteso compare
in 18 contesti Dense e 17 Hybrid: la sua presenza non basta a spiegare il punteggio.
Anche il supporto è un giudizio automatico da verificare, distinto dalla semplice
validità formale della citazione.

Il segnale a favore del RAG è utile, ma resta esplorativo: i quesiti sono già stati
usati nello sviluppo, mancano controllo indipendente e validazione umana dei casi
dubbi. Terra è un secondo modello rispetto al generatore, non un secondo giudice.
Nessun trattamento è stato promosso. Non possiamo dichiarare una strategia ottimale
o un vantaggio certo delle combinazioni sulla base di questi 20 risultati.

## Diagnostica locale estesa ai 100 intenti

È stato eseguito soltanto retrieval, con embedding BGE-M3 locale e lo stesso indice,
filtro, limite di 100 candidati e contesto 4096 token/12 unità. Quattro profili,
400 record, zero errori di retrieval, **zero nuove risposte, giudizi o chiamate OpenAI**.

| Profilo | Articolo atteso nei candidati iniziali, @100 | Articolo atteso nel contesto |
|---|---:|---:|
| Dense | 87/100 | 75/100 |
| Sparse | 87/100 | 78/100 |
| Hybrid | 91/100 | 81/100 |
| Graph, pool Hybrid prima dell'espansione | 91/100 | 79/100 |

Hybrid recupera otto contesti mancanti a Dense: Q0012, Q0020, Q0033, Q0035, Q0056,
Q0057, Q0068, Q0097. Perde Q0008 e Q0095. Il saldo è +6 punti, IC95% appaiato
0/+12; sensibilità per 21 gruppi di legge 0/+12. Graph non aggiunge successi di
contesto rispetto a Hybrid e perde Q0056/Q0097. La tabella distingue il pool
iniziale dal ranking successivo all'espansione.

Queste misure sono **proxy basati sui riferimenti legacy**, non accuratezza né
copertura completa delle evidenze. Q0097 mostra l'articolo 2, ma non il comma 1
utile al gold; Q0012 recupera una norma che consente un regolamento autonomo,
contraddicendo il gold che considera obbligatorio il tipo regionale. Migliore
retrieval può quindi rivelare un difetto del benchmark anziché aumentare il punteggio.
Un passaggio non citato nel gold resta non giudicato, non automaticamente errato.

I nove riferimenti assenti dai 100 candidati Hybrid hanno cause diverse:
Q0013–Q0016 puntano a chunk marcati `ceased_confirmed` della LR 44/1991, esclusi
dalla policy corrente; Q0076 non ha un chunk indicizzabile per l'articolo atteso;
Q0019/Q0043/Q0044/Q0081 hanno chunk ammessi dalla policy ma non recuperati nei 100.
Non vanno risolti tutti disabilitando il filtro. Per Q0081 la ricerca può usare
altre norme che definiscono Finaosta: il riferimento legacy non è un giudizio
esaustivo. Questa classificazione descrive gli artefatti, non certifica l'attualità
delle disposizioni né stabilisce che manchino altre evidenze sufficienti.

## Sei esempi prima/dopo, con prove e limiti

Sono casi di **miglioramento osservato della selezione**, scelti dai 100 diagnostici.
Tutti gli articoli seguenti mancano nel contesto Dense e compaiono in Hybrid.
I ranghi indicano candidati prima del context builder; `E` identifica il blocco
effettivamente mostrato. Due casi sanitari richiedono riconciliazione dell'identità,
uno recupera solo parte delle condizioni. Non sono sei nuove risposte corrette.

| QID | Disposizione attesa | Dense → Sparse → Hybrid | Evidenza nel contesto Hybrid |
|---|---|---|---|
| 0020 | LR 30/2005, art. 8 | 13 → 5 → 6 | E6: revoca parziale proporzionale |
| 0033 | LR 5/2000, art. 9 | 32 → 1 → 4 | E4: USL regionale; identità duplicata da verificare |
| 0035 | LR 5/2000, art. 38 | 40 → 8 → 9 | E9: accreditamento e accordi; identità duplicata da verificare |
| 0056 | LR 4/2012, art. 8 | 39 → 4 → 11 | E11: revoca; condizioni della sospensione ancora mancanti |
| 0057 | LR 14/2024, art. 1 | 21 → 1 → 5 | E5: finalità delle competizioni UCI a La Thuile |
| 0068 | LR 10/2022, art. 4 | assente nei 100 → 2 → 10 | E10: urgenza ed entrata in vigore il giorno successivo |

### Q0020 — Revoca parziale del contributo

Domanda: l'inadempimento comporta sempre la restituzione dell'intero contributo?
Dense privilegia LR 27/1998 art. 52 e LR 84/1993 art. 14, disposizioni di altri
settori sui contributi. LR 30/2005 art. 8 resta al rango 13 e fuori dal prompt.
Sparse lo trova al rango 5; la fusione RRF lo porta al rango 6/E6.

Estratto mostrato: «La revoca del contributo può essere disposta anche in misura
parziale, in modo proporzionale all'inadempimento riscontrato.»

Il meccanismo è la complementarità lessicale/semantica, non il grafo. La domanda
non specifica settore o legge: le altre norme non sono certificabili come
irrilevanti. In un'applicazione reale va esplicitato l'ambito, senza indovinarlo dal gold.

### Q0033 — Individuazione dell'Azienda USL

Dense mostra disposizioni generiche sulla programmazione sanitaria, tra cui
l'articolo 2 nelle due rappresentazioni LR 5/2000 e LR 5/2002. L'articolo 9 della
LR 5/2000 resta al rango 32. Sparse lo classifica primo; Hybrid lo mostra in E4.

Estratto: «La Regione assicura i livelli essenziali ed appropriati di assistenza
attraverso una unità sanitaria locale coincidente con l'ambito territoriale della regione.»
Il comma 2, nello stesso blocco, contiene la denominazione «Azienda regionale sanitaria
USL della Valle d'Aosta».

Il passaggio risponde all'identificazione richiesta, ma E5 ripete il corpo sotto
l'identità conflittuale LR 5/2002. Il guadagno di ranking è reale; l'attribuzione
canonica deve essere verificata prima di presentare il caso come fonte legale validata.

### Q0035 — Prestazioni sanitarie erogate da privati

Dense favorisce LR 65/1977 art. 15, relativa a organismi consultoriali, e altri
passaggi sanitari. L'articolo 38 della LR 5/2000 è al rango 40. Sparse lo porta a 8,
Hybrid a 9/E9.

Estratto: «nonché di soggetti erogatori pubblici o privati accreditati, previa
stipula di accordi o contratti.»

La clausola aggiunge una condizione concreta al breve gold sull'accreditamento.
È un miglioramento nell'evidenza specifica, con il medesimo duplicato d'identità
sanitaria in E10. Non ridurre la risposta a una parola chiave ignorando accordi/contratti.

### Q0056 — Sanzioni per la qualifica sanitaria degli allevamenti

Dense recupera la legge giusta, LR 4/2012, ma privilegia articolo 2 (definizioni)
e 12 (contributi). Il chunk dell'articolo 8 sui provvedimenti sanitari resta a 39.
Sparse lo trova a 4; Hybrid lo porta a 11/E11.

Estratto: «La qualifica di cui all'articolo 6, comma 2, è revocata dal servizio
veterinario competente dell'Azienda USL nel caso in cui le condizioni di cui agli
articoli 6, comma 1, e 7, comma 1, non siano rispettate.»

Il contesto mostra commi 3–5 e una menzione della sospensione, ma mancano i suoi
presupposti nei commi 1–2. Serve completare la disposizione, non aggiungere altre
leggi genericamente simili. Il grafo attuale inserisce due unità prima del chunk,
lo sposta a 13 e lo elimina dal contesto: esempio concreto di regressione da combinazione.

### Q0057 — Finalità della LR 14/2024

La domanda identifica esplicitamente LR 29 luglio 2024 n. 14. Dense mette al
primo posto LR n. 15 dello stesso giorno, sull'autoconsumo diffuso, quindi
disposizioni finanziarie della legge richiesta. L'articolo 1 è a 21.
Sparse lo trova primo; Hybrid lo mostra a 5/E5.

Estratto: «la presente legge disciplina gli interventi regionali finalizzati al
sostegno dell'organizzazione, svolgimento e promozione delle competizioni dell'Unione
ciclistica internazionale (UCI) Mountain bike World Series».
Il medesimo blocco precisa La Thuile e gli anni 2025–2027.

Qui LR 15/2024 è effettivamente un atto diverso da quello chiesto. Il caso motiva
un ramo di ricerca guidato dall'identificativo normativo risolto, mantenendo il
ramo globale per eventuali norme collegate. Questo ramo è una modifica pianificata;
il miglioramento già osservato deriva dal ramo sparse e da RRF.

### Q0068 — Urgenza della LR 10/2022

Dense favorisce disposizioni transitorie di LR 22/2010 art. 76 e LR 15/2021 art. 55.
L'articolo 4 della legge espressamente chiesta manca anche nei 100 candidati.
Sparse lo recupera a 2; Hybrid lo include a 10/E10.

Estratto: «entra in vigore il giorno successivo a quello della sua pubblicazione
nel Bollettino ufficiale della Regione.» La frase completa dichiara l'urgenza
ai sensi dell'articolo 31, comma terzo, dello Statuto speciale.

È un vero aumento del recupero nei candidati, non soltanto un riordinamento.
Il grafo non trova una nuova evidenza attesa: sposta il medesimo chunk a 12/E12.
Non è stata generata una nuova risposta Luna per questa diagnostica.

Le schede JSON conservano URL ufficiale dello snapshot, chunk/unità, offset degli
estratti, ranking dei rami, contesto completo e anomalie. Il report permette di
aprire ogni caso e confrontare questi passaggi senza chiamare modelli.

## Cosa cambiare e cosa non adottare ancora

| Problema osservato | Modifica proposta | Verifica necessaria |
|---|---|---|
| Q0018: sei risposte RAG falliscono per `allegate` invece di `allegata`, pur avendo la fonte in E1. | Selezione di segmenti tramite ID e citazioni materializzate dal backend. | Elimina la trascrizione inventata senza confondere validità formale e supporto. Nuovo protocollo comune a tutti i baseline. |
| Q0057/Q0068: identificativo esplicito non sufficiente a guidare Dense. | Ramo Hybrid sull'atto risolto, mantenendo ricerca globale. | Migliora il contesto richiesto senza perdere eccezioni e atti collegati; zero riferimenti gold in runtime. |
| Q0056: articolo presente ma condizioni mancanti. | Completamento di commi, introduzioni ed eccezioni con span strutturali. | Copertura delle clausole verificate, stesso budget completo, nessuna omissione silenziosa. |
| Q0036: requisiti già nel contesto, ma omessi nella risposta. | Diagnosi e ablation separata del prompt comune di risposta. | Distinguere un errore di generazione dalla mancanza di evidenza; nessuna checklist ricavata dal gold nei prompt. |
| Q0033/Q0035/Q0036: testi sanitari duplicati sotto identità diverse. | Alias canonici verificati e deduplica con provenienza multipla. | Non fondere leggi diverse solo perché il testo coincide; preservare raw e conflitti. |
| Q0007: il grafo aggiunge fitopatie e avversità agricole a una domanda sui veicoli. | Espansione dalla menzione locale, tipo operativo e rilevanza del target; promozione di target già nel pool. | Un hop, massimo due unità; distinguere articolo nuovo, promozione e comparsa effettiva nel contesto. |
| Reranker: restituisce soltanto i primi 20 e perde la coda; mancano identità/rubrica negli input. | Riordinare il prefisso con metadati e conservare gli altri 80. | Identico pool iniziale; condizioni utili oltre 20 ancora selezionabili. |
| Multi-query: beneficio non misurato, esiti inferiori a Hybrid. | Varianti orientate a condizioni/eccezioni, con conservazione di attori, atti e negazioni. | Ablation separata; adozione soltanto se migliora copertura e poi risposte. |
| Q0016: disposizione di legge cessata recuperata come citazione in un atto modificativo. | Identità e ambito temporale del testo citato distinti dal suo contenitore. | Nessuna certificazione di attualità dal semplice stato del contenitore. |

Nei 20 test il grafo ha aggiunto **26 chunk in 14 domande**, tutti mostrati e nessuno
citato. L'assenza di guadagno non dipende qui da aggiunte scartate per budget.
Il problema osservato è la selezione: seed documentali troppo ampi e ordine per ID,
senza legame sufficiente con la domanda. L'assenza di citazione da sola non prova
che ogni aggiunta sia inutile; va controllato il supporto delle affermazioni.

È stata inoltre riprodotta la fusione multi-query dai ranking salvati: controllo
1:1:1 identico nei 20 contesti. Con pesi 2:1:1 e 4:1:1 l'articolo nel contesto
resta **17/20**, mentre Article Success@5 peggiora da 17 a 16. Q0004 passa da
rango 5 a 6/7; Q0095 da 14 a 16/17, sempre fuori dal contesto.
**Aumentare il peso della domanda originale non è quindi una soluzione adottata.**
Cambiano alcuni testi del contesto: senza nuova generazione non possiamo inferire
che le risposte rimangano identiche. La formula provata è RRF lineare esterna fra
query, distinta dalla variante pesata nativa Qdrant.

## Sequenza di lavoro e criteri di scelta

1. Correggere provenienza/citazioni e revisionare i casi gold/giudice problematici.
   Sono interventi comuni, da separare dal guadagno attribuito al retrieval.
2. Misurare localmente identificativi, completamento, grafo mirato e reranking
   preservando la coda; una modifica per volta. Non ricostruire l'indice se cambiano
   soltanto ranking e contesto. Nuovi payload/testi richiedono identità nuova.
3. Combinare inizialmente al massimo due interventi utili, quindi rimuovere ciascuno
   per capire che contributo porta. Il multi-query resta disabilitato se non aiuta.
4. Congelare il candidato e confrontare No-RAG/Dense/Hybrid/Advanced sui medesimi
   20 QID con Luna/Terra e nuovi baseline coerenti. Massimo 160 chiamate per
   risposte/giudizi più, se necessarie, 40 trasformazioni. Nessuna chiamata eseguita ora.
5. Passare ai 100 soltanto per i finalisti, con controllo indipendente e budget
   separatamente verificato. Nuovi quesiti realmente non usati serviranno per
   valutare generalizzazione; gli 80 legacy fuori dal pilot restano già esplorati.

Obiettivo pratico: +10 punti rispetto a Dense, senza regressioni note di fonte,
citazione o esclusione. Non è una promessa. Il vincitore deve migliorare la
fully-correct rate **e** il supporto, con costi e IC95% visibili; a parità si conserva
la soluzione semplice. Sei casi favorevoli non possono sostituire il confronto
completo. Q0012/Q0097 e le perdite Q0008/Q0095 resteranno visibili nei report.

Per il prossimo pilot si propone un tetto incrementale di 2 EUR, entro il cap
cumulativo di 5 EUR con riserva 0,50. Il ledger resta **425 voci / 1,963799125 EUR**,
immutato durante questa revisione. Prezzi e limiti di input/output saranno
verificati prima di un batch autorizzato; un limite esaurito produce checkpoint e
report incompleto, non risultati inventati. Questa nota pianifica il lavoro e non
avvia né autorizza nuovi batch.

La letteratura sostiene le ipotesi, senza garantire il risultato locale:
[DMQR-RAG](https://arxiv.org/abs/2411.13154) per la diversità delle query,
[SAT-Graph RAG](https://journals.sagepub.com/doi/10.3233/FAIA251598) per relazioni
strutturali/temporali e [Lost in the Middle](https://arxiv.org/abs/2307.03172)
per verificare l'ordine delle evidenze. Il secondo propone anche un sistema di
versioni/ontologia più ampio e una valutazione qualitativa simulata: non dimostra
un incremento quantitativo sul nostro benchmark. Le applicazioni e i limiti sono
distinti nella [bibliografia](../notes/01_literature.md).

## Artefatti e riproducibilità della revisione

- `data/diagnostics/strategy_review_20261011/full100/{manifest,metrics}.json`
  e `records.jsonl`: diagnostica completa, run `9abac847c05738dbfceb5328a7df46e7ca918050f4abf6b0d1d1906508364fb7`.
- `retrieval_cases.json`, `evaluation_analysis.json`, `architecture_review.json`
  nello stesso contenitore: audit automatici, nessuna approvazione umana.
- `multiquery_replay.json` e `multiquery_replay_records.jsonl`: controllo esatto
  e due replay, 60 record, tokenizer locale senza embedding o chiamate API.
- `analysis_summary.json`: hash degli input, confronti appaiati e baseline del ledger.
- `candidate_misses.json`: distinzione tra cessazione, riferimento senza chunk
  e chunk ammesso dalla policy ma assente dai primi 100.
- `docs/results/10_case_comments.jsonl`: commenti durevoli separati dall'HTML.
- `data/reports/strategy_review_20261011/index.html`: report dei 20 risultati
  congelati e dei 100 diagnostici, chiaramente distinti.

Source SHA256: `3cc5bb20e001962f46fd65f35386fb19a35c3ea2ac8b04c3f17e453536656b8a`;
corpus manifest: `282cfa61dbcf49ab3fc544d096afe8f22deb1114d5bd689a40f154b941be20bb`.
La nuova analisi non modifica codice runtime, corpus, indice, domande, gold,
risposte, giudizi, report precedenti o archivio congelato del pilot.
