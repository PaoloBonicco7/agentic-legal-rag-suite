# Protocollo sperimentale e stato delle evidenze

Al 10 ottobre 2026 la nuova pipeline implementa il protocollo di confronto; non
ha ancora prodotto un confronto QA completo con modelli reali. Il preflight
Utopia salvato in `data/diagnostics/utopia_preflight.json` registra `http_403`.
Non è stata fornita una chiave OpenAI per la valutazione. Nessun miglioramento
della correttezza generativa è quindi dichiarabile dalla nuova implementazione.
I risultati della vecchia codebase restano evidenze storiche con un altro protocollo.

## Protocollo implementato

- I CSV originali restano intatti: 100 intenti, con no-hint primario e MCQ
  diagnostico. Le due forme non costituiscono 200 osservazioni indipendenti.
- No-RAG, dense, sparse e hybrid condividono generatore, domande, corpus,
  filtro, embedding e budget. Le ablation registrano il trattamento distinto.
- La metrica primaria è la quota di score 2; la media 0–2 è secondaria.
  Correttezza rispetto al gold legacy, grounding sul contesto realmente inviato
  e validità formale delle citazioni sono misure distinte.
- Un secondo giudice controlla 20 QID stabili, stratificati per livello dove
  possibile. Il disaccordo rimane visibile; nessun giudice sostituisce una verifica
  umana. Per la promozione il secondo giudice deve avere un diverso provider o
  modello; un timeout diverso non basta. Confronti appaiati e IC95% descrivono
  domande già esplorate.
- La vista corrente è separata e richiede revisioni umane attribuite allo snapshot.
  Non può essere promossa finché tutti gli intenti originali non sono classificati;
  i casi storici esclusi restano conservati. Una domanda riscritta richiede nuova
  generazione; una correzione del solo gold può rivalutare risposte congelate.
  Le approvazioni richiedono motivazione e fonti; le correzioni e i casi storici
  richiedono anche prove riportate nel companion, senza validazione automatica
  del gold da parte del codice.
- Il diagnostico oracle usa esclusivamente chunk approvati, verificati contro il
  corpus. È un limite diagnostico e non un candidato alla promozione operativa.

## Cosa è verificato

Test con client e retriever fittizi verificano allineamento, assenza di gold nelle
chiamate di retrieval/generazione, checkpoint, ripresa e limiti di budget. Un blocco
prima del dispatch resta `not_attempted`, con score mancante; un HTTP403 tentato
è un singolo fallimento prima di interrompere nuove chiamate. I QID non visitati
non diventano errori o zeri artificiali. In presenza di score mancanti le quote
sul denominatore previsto sono limiti inferiori, non confronti completi.

Il report corpus legacy verificato contiene 3145 documenti e 19770 chunk.
La verifica statica ha esaminato `data/reports/legacy-final` (3147 pagine HTML)
e `data/reports/current-corpus` (3178 pagine, da un corpus corrente intermedio).
Non ha trovato link locali interrotti, URL eseguibili, risorse remote, handler
HTML, elementi incorporati attivi, script estranei al filtro locale previsto o
checksum di input/output incoerenti. Il controllo riguarda queste due versioni;
un report rigenerato richiede una nuova verifica. Il rendering visivo nel browser
non è stato verificato: la policy del browser blocca URL `file:`. Nessun aggiramento
è stato tentato. Queste verifiche non dimostrano accuratezza giuridica del parser,
leggibilità visiva nel browser o correttezza delle risposte.

Le diagnosi retrieval senza LLM registrano separatamente pool originale, selezione
e contesto, usando Article Success/MRR come proxy di riferimenti incompleti.
Non producono score di correttezza, precision, nDCG o recall delle evidenze.
Gli artefatti reali di tali diagnosi, una volta completato l'indice, costituiscono
misure di retrieval; il guadagno QA resta da misurare con generatore e giudici.

I commenti interpretativi sono JSONL separati, con autore e scope opzionale per
QID, run o documento. Il report li legge e li mostra con testo escapato; conserva
percorso e checksum senza modificare annotazioni, score o gold. La verifica dei
pacchetti controlla anche la catena interna snapshot/catalogo/raw/corpus/run,
i CSV del benchmark dichiarati e il companion raw di ogni revisione utilizzata.
Le revisioni vengono conservate senza riscrittura in `source/reviews/<sha>.jsonl`;
se un run ne dichiara l'hash senza il file, il pacchetto viene rifiutato. Ricalcolare
soltanto gli hash esterni non rende valido un artefatto internamente incoerente.

I costi mostrati sono contabilizzati: possono essere stime o riserve prudenziali,
con basi esplicite nei record. Il ledger persistente è la fonte contabile e governa
il tetto complessivo di 20 EUR; le verifiche dei prezzi possono aggiornarsi senza
cambiare l'identità scientifica o impedire la ripresa dei checkpoint.

## Funzioni implementate e misure ancora necessarie

L'interfaccia comune no-RAG/dense/sparse/hybrid, le ablation di riscrittura,
espansione del grafo e reranking LLM, il diagnostico oracle, i due giudici e la
rivalutazione del gold corretto sono implementati e verificati con fakes.
Non costituiscono ancora risultati QA misurati con modelli reali. I report
affiancano risposte e checkpoint per QID/formato, mantenendo descrittivi i
confronti incompleti o incompatibili. Le pagine retrieval affiancano proxy e
contesti senza attribuire uno score generativo.

I blocchi esterni alla verifica QA sono l'accesso Utopia rifiutato e l'assenza della
chiave OpenAI. Il corpus corrente e il relativo indice devono inoltre essere
identificati dalla stessa versione finale prima di una replica. La revisione
umana del companion e il controllo dei giudici restano da eseguire: nessuna
proposta automatica o risultato di un fake vale come approvazione umana.
