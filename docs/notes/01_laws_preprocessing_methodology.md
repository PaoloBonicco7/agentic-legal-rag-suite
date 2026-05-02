# 01 - Metodologia di Preprocessing delle Leggi

## Obiettivo

Questa fase trasforma il corpus HTML delle leggi regionali in un dataset pulito, strutturato e pronto per le fasi successive di indicizzazione e RAG.

L'obiettivo metodologico non e produrre una copia semplificata del testo, ma costruire una rappresentazione esplicita della struttura giuridica contenuta nei file originali. Ogni unita recuperabile deve mantenere il collegamento con la legge, l'articolo, il passaggio, la provenienza del file sorgente, lo status legale e le relazioni esplicite con altre leggi.

Il principio guida e conservativo: si estraggono solo informazioni che hanno un'evidenza osservabile nel corpus. Non vengono inferiti significati giuridici non presenti nel testo, nei link o nelle citazioni.

## Dataset iniziale

Il dataset iniziale si trova in `data/laws_html/` ed e composto da file HTML reali, uno per legge. Ogni file contiene sia testo giuridico sia markup utile a ricostruire la struttura del documento.

La struttura iniziale ha tre caratteristiche importanti:

- il nome del file contiene una parte dell'identita della legge, per esempio data e numero;
- il contenuto HTML contiene heading, paragrafi, anchor interne, link e note;
- il corpus puo contenere file non sorgente, come `.DS_Store`, che non fanno parte del dataset legale.

Il primo passaggio metodologico e quindi distinguere il corpus giuridico vero dai file accessori e verificare che ogni HTML valido possa essere associato a una legge con identita stabile.

## Pattern osservabili nel corpus

Il corpus non e un formato tabellare gia pronto. La struttura deve essere riconosciuta da pattern ricorrenti nel markup e nel testo.

### Filename

Il filename codifica data e numero della legge. Questa informazione viene usata per costruire un identificatore stabile della forma concettuale:

```text
vda:lr:<data-legge>:<numero-legge>
```

L'identificatore non dipende dalla posizione assoluta del file nel filesystem, quindi resta stabile tra macchine diverse.

### Heading e preambolo

Gli heading HTML e i primi paragrafi permettono di recuperare il titolo della legge e il testo introduttivo. Il preambolo e utile per intercettare informazioni generali, come indicazioni di abrogazione dell'intera legge o riferimenti ad altre leggi.

### Indice

Molte leggi contengono un indice iniziale con link agli articoli. L'indice e utile come segnale di struttura, ma non deve essere scambiato per il testo normativo vero e proprio. La metodologia lo riconosce e lo salta quando costruisce articoli e passaggi, evitando duplicazioni.

### Anchor degli articoli

Gli articoli sono spesso marcati da anchor interne, come `articolo_1__`. Questo e uno dei segnali piu forti per riconoscere l'inizio di un articolo.

Quando il markup e meno regolare, l'articolo puo essere riconosciuto anche da righe testuali come `Art. 1` o `Articolo 1`. In entrambi i casi l'etichetta dell'articolo viene normalizzata, per esempio `Art. 4 bis` diventa `4bis`.

### Paragrafi, commi e lettere

Il testo dell'articolo viene suddiviso in passaggi piu piccoli seguendo pattern giuridici ricorrenti:

- testo prima del primo comma: introduzione dell'articolo;
- righe che iniziano con `1.`, `2.`, `3.`: commi;
- righe che iniziano con `a)`, `b)`, `c)`: lettere.

Questa scelta evita di costruire chunk arbitrari direttamente sul testo completo dell'articolo. Prima si preserva la struttura giuridica, poi si applica il chunking solo quando un passaggio e troppo lungo.

### Note

Le note sono riconosciute tramite anchor come `nota_...` e tramite link interni che rimandano alla nota. Vengono mantenute come componente separata perche spesso contengono informazioni redazionali rilevanti, come modifiche, sostituzioni, inserimenti o abrogazioni.

Quando una nota e collegata a un passaggio o a un articolo, il nuovo dataset conserva questo collegamento.

### Hyperlink e citazioni testuali

Le relazioni tra leggi vengono ricavate da due tipi di evidenza:

- hyperlink HTML che contengono parametri riferiti ad altre leggi;
- citazioni testuali riconoscibili, come `Legge regionale ... n. ...` o `L.R. n/anno`.

Un riferimento viene considerato risolto solo quando puo essere collegato a una legge presente nel corpus locale. I riferimenti non risolti non vengono ignorati silenziosamente: sono contati e riportati nel manifest e nel quality report.

## Principi di pulizia

La pulizia del dataset segue alcuni principi espliciti.

### Preservazione del dato sorgente

I file in `data/laws_html/` sono trattati come sorgenti. La pipeline li legge, ma non li modifica. Il dataset pulito viene scritto in `data/laws_dataset_clean/`, che e un artifact rigenerabile.

### Determinismo

A parita di corpus e configurazione, il processo deve produrre gli stessi identificatori, gli stessi ordinamenti e gli stessi file. Per questo gli input vengono ordinati, gli ID sono derivati da caratteristiche stabili e gli output JSONL sono scritti in ordine deterministico.

### Separazione tra sorgente e artifact

Il corpus HTML e il dataset pulito hanno ruoli diversi. Il primo e il materiale originale da conservare; il secondo e una rappresentazione generata per retrieval, filtro, analisi e spiegabilita.

### Nessuna inferenza speculativa

Le relazioni del grafo vengono create solo quando esiste evidenza esplicita: link, citazione testuale o nota. Il sistema non deduce relazioni implicite e non prova a interpretare giuridicamente il contenuto oltre i pattern osservabili.

### Errori visibili

File non validi, riferimenti non risolti, ID duplicati e campi mancanti devono emergere come diagnostica o come fallimenti di validazione. L'obiettivo e evitare che dati ambigui entrino silenziosamente nelle fasi successive.

## Costruzione del nuovo dataset

Il dataset pulito e diviso in componenti, ognuna con una responsabilita precisa.

### `laws.jsonl`

Rappresenta le leggi del corpus.

Ogni record contiene l'identita della legge, data, numero, titolo, status, file sorgente e preambolo. Questa tabella e il livello piu alto del dataset e permette di collegare ogni elemento successivo alla sua fonte normativa.

Serve nelle fasi successive per filtri per legge, citazioni, spiegazione delle fonti e ricostruzione del contesto.

### `articles.jsonl`

Rappresenta gli articoli estratti da ogni legge.

Ogni articolo nasce da anchor HTML o da pattern testuali equivalenti. L'articolo conserva etichetta normalizzata, titolo o rubrica quando presente, percorso strutturale e stato dell'articolo.

Serve per mantenere il recupero aderente alla struttura normativa, invece di trattare la legge come testo piatto.

### `passages.jsonl`

Rappresenta le porzioni interne di un articolo.

I passaggi corrispondono a introduzioni, commi e lettere. Questa granularita e utile per retrieval perche un articolo puo essere lungo e contenere molte informazioni distinte. Il passaggio e il livello concettuale prima del chunking.

Serve per creare unita di testo abbastanza precise da recuperare, ma ancora spiegabili come parte di una struttura legale.

### `notes.jsonl`

Rappresenta le note redazionali.

Le note vengono estratte separatamente perche possono contenere informazioni sullo stato del testo o sulle modifiche introdotte da altre leggi. Quando possibile, sono collegate agli articoli o ai passaggi che le richiamano.

Servono per preservare evidenze che sarebbero facili da perdere se il testo venisse solo concatenato.

### `edges.jsonl`

Rappresenta il grafo delle relazioni esplicite.

Ogni edge collega una legge, un articolo o un passaggio sorgente a una legge destinazione, eventualmente anche a un articolo destinazione. Il tipo relazione viene assegnato in base al testo di evidenza: riferimento generico, abrogazione, modifica, sostituzione o inserimento.

Serve per retrieval graph-aware, espansione del contesto e analisi dei collegamenti normativi.

### `chunks.jsonl`

Rappresenta le unita pronte per indicizzazione e retrieval.

Ogni chunk deriva da un passaggio. Se il passaggio e lungo, viene diviso con una finestra deterministica a parole; se e breve, resta un singolo chunk. Il chunk contiene sia il testo pulito sia `text_for_embedding`, che aggiunge contesto come legge, articolo e percorso strutturale.

I chunk includono metadati denormalizzati: status, provenienza, viste di indicizzazione, relazioni entranti e uscenti, tipi relazione e identificatori stabili. Questo evita che lo step di indicizzazione debba ricostruire il contratto.

## Status legale e viste di indicizzazione

Il dataset distingue tra materiale storico e materiale adatto alla vista corrente.

Ogni chunk include sempre la vista `historical`, perche anche una legge passata puo essere utile per ricostruire evoluzione normativa o relazioni. La vista `current` viene aggiunta solo quando la legge e l'articolo risultano adatti al recupero come diritto corrente.

Gli status ammessi per le leggi sono:

- `current`;
- `past`;
- `unknown`;
- `index_or_empty`.

Lo status non pretende di sostituire una valutazione giuridica completa. E una classificazione operativa e documentata, utile per filtri e retrieval, basata su segnali espliciti osservabili nel corpus.

## Validazione e tracciabilita

La pipeline produce un `manifest.json` che registra configurazione, conteggi, hash del corpus sorgente, hash degli output e quality gates.

La validazione controlla che:

- esista almeno un file HTML valido;
- gli ID siano non vuoti e senza duplicati;
- i chunk contengano tutti i campi richiesti;
- i metadati lista siano liste reali e non stringhe serializzate;
- il grafo pulito non contenga self-loop;
- i tipi relazione e gli status appartengano ai valori ammessi;
- gli output dichiarati esistano e abbiano hash;
- `chunks.jsonl` non sia vuoto.

Il campo `ready_for_indexing` sintetizza l'esito di questi controlli. Le fasi successive devono consumare il dataset pulito solo quando questo flag e vero.

## Risultato metodologico

Il risultato non e solo un file di chunk, ma un dataset multilivello:

```text
legge -> articolo -> passaggio -> chunk
              \-> note
              \-> relazioni esplicite
```

Questa struttura permette di fare retrieval mantenendo:

- provenienza del testo;
- granularita giuridica;
- filtri per status e vista;
- relazioni con altre leggi;
- spiegabilita del percorso dalla fonte HTML al chunk indicizzato.

## Limiti metodologici

La metodologia e intenzionalmente conservativa.

Alcuni riferimenti possono rimanere non risolti se citano leggi fuori dal corpus, se usano forme testuali non coperte dai pattern o se sono ambigui. Questi casi vengono conteggiati, non trasformati in relazioni speculative.

Lo status legale e una classificazione operativa basata su segnali testuali e note. Non sostituisce una verifica normativa manuale.

La qualita dell'estrazione dipende dalla regolarita del markup HTML sorgente. La pipeline gestisce varianti ricorrenti, ma documenti molto malformati o strutture non standard possono richiedere revisione o regole aggiuntive.

Infine, il dataset pulito e un artifact generato: deve essere ricostruibile dal corpus e dalla configurazione, non modificato manualmente.
