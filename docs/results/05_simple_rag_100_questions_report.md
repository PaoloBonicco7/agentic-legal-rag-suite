# Simple RAG - Resoconto 100 domande

Run sorgente: `data/rag_runs/simple/`.

## Metriche globali
- `mcq`: processed=100, judged=100, score_sum=80/100, accuracy=0.8, strict_accuracy=0.8, coverage=1.0, errors=0
- `no_hint`: processed=100, judged=100, score_sum=88/200, accuracy=0.44, strict_accuracy=0.44, coverage=1.0, errors=0

## Metriche per livello

| dataset | level | processed | judged | score | accuracy | strict_accuracy | errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| mcq | L1 | 25 | 25 | 19/25 | 0.76 | 0.76 | 0 |
| mcq | L2 | 25 | 25 | 21/25 | 0.84 | 0.84 | 0 |
| mcq | L3 | 25 | 25 | 20/25 | 0.8 | 0.8 | 0 |
| mcq | L4 | 25 | 25 | 20/25 | 0.8 | 0.8 | 0 |
| no_hint | L1 | 25 | 25 | 18/50 | 0.36 | 0.36 | 0 |
| no_hint | L2 | 25 | 25 | 24/50 | 0.48 | 0.48 | 0 |
| no_hint | L3 | 25 | 25 | 22/50 | 0.44 | 0.44 | 0 |
| no_hint | L4 | 25 | 25 | 24/50 | 0.48 | 0.48 | 0 |

## Distribuzione score

- MCQ score: {0: 20, 1: 80}
- No-hint judge_score: {0: 43, 1: 26, 2: 31}

## Righe MCQ errate

| qid | level | correct | predicted | question |
|---|---|---|---|---|
| eval-0004 | L4 | B | A | Quali istituzioni partecipano alla politica socio-sanitaria regionale? |
| eval-0007 | L3 | B | C | Io ho 32 anni, sono un soggetto privato che non esercita un'attività economica e ho acquistato un veicolo a bassa emissione nuovo di fabbric |
| eval-0012 | L4 | D | C | I Comuni sono liberi di adottare i propri regolamenti edilizi? |
| eval-0017 | L1 | C | A | Con quale atto giuridico vengono concessi i contributi per la tutela, la conservazione e la valorizzazione dei borghi in Valle d'Aosta? |
| eval-0019 | L3 | E | D | Se ho effettuato un lavoro di ristrutturazione della facciata di un edificio storico e tale lavoro è costato 100.000 euro, posso chiedere il |
| eval-0025 | L1 | D | C | Quanti anni dura l'autorizzazione alla coltivazione di cave e torbiere? |
| eval-0036 | L4 | D | C | Autorizzazione e accreditamento delle strutture sanitarie sono lo stesso atto giuridico? |
| eval-0041 | L1 | B | D | Che cos’è il PTP? |
| eval-0050 | L2 | C | F | La Commissione regionale per il patrimonio culturale è permanente? |
| eval-0055 | L3 | C | F | Quando l’infezione da virus BHV-1 è considerata limitata? |
| eval-0058 | L2 | A | D | Fino a quale anno è previsto un contributo regionale a sostegno degli eventi di cui alla Legge regionale 29 luglio 2024, n. 14? |
| eval-0063 | L3 | E | A | Il proprietario o l’avente diritto dei fondi confinanti con la sede stradale regionale può aprire canali, fossi ed eseguire escavazioni nei  |
| eval-0065 | L1 | A | D | A quali soggetti si applica la Legge regionale 30 maggio 2022, n. 10 |
| eval-0068 | L4 | D | C | Alla Legge regionale 30 maggio 2022, n. 10 si applica il periodo di 15 giorni di vacatio legis previsto dallo Statuto regionale? |
| eval-0085 | L1 | E | F | In caso di accesso ai fondi, cosa deve pagare il ricercatore o il concessionario ai proprietari dei terreni ai sensi della Legge regionale 1 |
| eval-0086 | L2 | C | D | Per quale attività è richiesta l’autorizzazione della Giunta regionale ai sensi della legge regionale 13 marzo 2008, n. 5? |
| eval-0087 | L3 | B | A | La vigilanza sulla coltivazione delle cave, delle miniere e delle acque minerali naturali, di sorgente e termali è effettuata solo ed esclus |
| eval-0089 | L1 | B | F | Quale struttura ha istituito a Bruxelles la Regione Valle d’Aosta? |
| eval-0094 | L2 | D | F | La Regione promuove l’utilizzo di alloggi a fini turistici? |
| eval-0100 | L4 | A | F | Con quali provvedimenti la Regione viene in aiuto alle imprese tenute a sospendere l’attività a causa della pandemia da COVID-19? |

## Righe no-hint con score < 2

| qid | level | score | question | predicted_answer |
|---|---|---:|---|---|
| eval-0001 | L1 | 0 | Quali sono gli organi dell'azienda Usl? | Il contesto normativo fornito non elenca esplicitamente gli organi dell'azienda USL. Tuttavia, menziona il Collegio sindacale come destinatario di atti aziendal |
| eval-0003 | L3 | 1 | Cosa accade in caso di vacanza dell'ufficio o assenza o impedimento del direttore generale, qualora  | In caso di vacanza dell'ufficio o di assenza o impedimento del direttore generale, le relative funzioni sono svolte dal direttore amministrativo o dal direttore |
| eval-0004 | L4 | 1 | Quali istituzioni partecipano alla politica socio-sanitaria regionale? | La Regione, attraverso l'Azienda regionale sanitaria USL della Valle d'Aosta (USL), e gli utenti e le loro associazioni, partecipano alla politica socio-sanitar |
| eval-0006 | L2 | 0 | Con quali dei seguenti strumenti la Regione incentiva la mobilità sostenibile? | La Regione incentiva la mobilità sostenibile mediante strumenti volti a favorire le condizioni per uno sviluppo sostenibile. |
| eval-0007 | L3 | 0 | Io ho 32 anni, sono un soggetto privato che non esercita un'attività economica e ho acquistato un ve | Il contesto fornito non contiene informazioni relative a contributi regionali per l'acquisto di veicoli a bassa emissione da parte di soggetti privati non impre |
| eval-0008 | L4 | 0 | Io ho comprato un'automobile nel 2025 beneficiando di un contributo regionale. Dopo sei mesi vendo l | Il contesto fornito non contiene disposizioni specifiche relative alla vendita di un'automobile acquistata con contributo regionale entro un certo periodo, né s |
| eval-0011 | L3 | 0 | Un accordo di programma può essere stipulato anche da soggetti privati? | Sì, un accordo di programma può essere stipulato anche da soggetti privati. La norma prevede che, qualora l'accordo riguardi opere o interventi alla cui realizz |
| eval-0012 | L4 | 0 | I Comuni sono liberi di adottare i propri regolamenti edilizi? | La libertà dei Comuni di adottare regolamenti edilizi non è espressamente trattata nel contesto fornito. Tuttavia, il riferimento alla legge regionale n. 63/199 |
| eval-0015 | L3 | 1 | Una società cooperativa che esercita attività di produzione artigianale tipica ha acquistato materie | No, non può chiedere il contributo nella misura massima del 60% delle spese. Il contributo del 60% è previsto solo per la realizzazione di piani di sviluppo del |
| eval-0016 | L4 | 0 | Una società cooperativa che esercita attività di produzione artigianale tipica può chiedere un contr | La normativa regionale non prevede in via generale un contributo per l'assunzione a tempo indeterminato da parte di società cooperative operanti nel settore art |
| eval-0017 | L1 | 1 | Con quale atto giuridico vengono concessi i contributi per la tutela, la conservazione e la valorizz | I contributi per la tutela, la conservazione e la valorizzazione dei borghi in Valle d'Aosta sono concessi ai sensi della legge regionale 18 novembre 2005, n. 3 |
| eval-0018 | L2 | 0 | Quale organo regionale deve effettuare l'istruttoria sulle domande presentate ai fini della concessi | La legge regionale 18 novembre 2005, n. 30, che disciplina la tutela, la conservazione e la valorizzazione dei borghi in Valle d’Aosta, non specifica esplicitam |
| eval-0019 | L3 | 0 | Se ho effettuato un lavoro di ristrutturazione della facciata di un edificio storico e tale lavoro è | No, non è possibile chiedere il rimborso totale delle spese. Le norme citate prevedono specifici importi massimi di rimborso per attività di protezione civile s |
| eval-0021 | L1 | 0 | Che cos'è il GAP? | Il contesto fornito non contiene informazioni sul significato o la definizione di 'GAP'. |
| eval-0022 | L2 | 0 | Posso aprire una sala da gioco a 300 metri da una scuola? | Il testo fornito non contiene norme specifiche che vietano o regolano la distanza tra una sala da gioco e una scuola. Le norme citate riguardano requisiti per l |
| eval-0023 | L3 | 1 | Posso mettere una slot-machine davanti alla vetrina del mio bar, in modo tale che coloro che cammina | No, non è consentito installare una slot-machine davanti alla vetrina di un bar in modo che sia visibile dai passanti. Secondo la normativa regionale, le slot-m |
| eval-0024 | L4 | 0 | A quali condizioni è ammessa la pubblicità per le sale da gioco? | Il testo fornito non contiene disposizioni specifiche riguardanti la pubblicità per le sale da gioco. |
| eval-0025 | L1 | 0 | Quanti anni dura l'autorizzazione alla coltivazione di cave e torbiere? | Il contesto fornito non specifica la durata dell'autorizzazione alla coltivazione di cave e torbiere. |
| eval-0026 | L2 | 0 | Con quale atto della Regione viene adottato il PRAE, il Piano Regionale delle attività estrattive? | Il Piano Regionale delle attività estrattive (PRAE) viene adottato con deliberazione della Giunta regionale, ai sensi dell'art. 3, comma 1, lettera d), della le |
| eval-0027 | L3 | 1 | Quali dei seguenti materiali deve utilizzare una struttura per poter essere qualificata "stabiliment | La qualificazione di stabilimento termale richiede l'utilizzo di acque minerali o termali, sorgenti, fanghi, gas naturali o altri prodotti naturali aventi propr |
| eval-0028 | L4 | 1 | Nel caso di vendita di una cava, quali sono le autorizzazioni di cui deve munirsi il compratore? | Nel caso di vendita di una cava, il compratore deve munirsi dell'autorizzazione alla coltivazione di cava. Ai sensi dell'art. 13, comma 4, della legge regionale |
| eval-0030 | L2 | 1 | Quali sono le iniziative che la Regione intraprende al fine di prevenire la violenza di genere? | La Regione intraprende iniziative tra cui colloqui con donne vittime di violenza di genere per individuarne i bisogni e fornire loro le prime indicazioni utili, |
| eval-0033 | L1 | 1 | Qual è l’organo che la Regione utilizza per assicurare i livelli essenziali e appropriati di assiste | La Regione utilizza il Servizio sanitario regionale (SSR) per assicurare i livelli essenziali e appropriati di assistenza sanitaria. |
| eval-0034 | L2 | 1 | Quali prestazioni eroga l’Azienda USL? | L’Azienda USL eroga prestazioni gratuite per i compiti di Istituto e nei termini di legge, come previsto dalla normativa regionale. |
| eval-0037 | L1 | 0 | Quando la farmacia è aperta al pubblico, con quali modalità deve essere assicurato il servizio farma | La norma citata non disciplina esplicitamente le modalità di erogazione del servizio farmaceutico quando la farmacia è aperta al pubblico, ma si riferisce esclu |
| eval-0039 | L3 | 1 | Quali farmaci il cittadino può richiedere quando la farmacia eroga il servizio a battenti chiusi? | Il cittadino può richiedere soltanto le prescrizioni di ricette sulle quali il medico ha fatto esplicita menzione del carattere di urgenza, nonché di ricette o  |
| eval-0040 | L4 | 0 | Il farmacista può chiudere per ferie la sua farmacia per 34 giorni l’anno? | La normativa regionale citata non prevede esplicitamente un limite massimo di chiusura annuale per ferie della farmacia. Tuttavia, l'art. 4 della Legge regional |
| eval-0041 | L1 | 0 | Che cos’è il PTP? | Il contesto fornito non contiene alcuna definizione o riferimento esplicito al termine "PTP". |
| eval-0042 | L2 | 0 | Che cos’è lo sviluppo sostenibile ai sensi della Legge regionale 6 aprile 1998, n. 11? | La Legge regionale 6 aprile 1998, n. 11 non definisce esplicitamente il concetto di «sviluppo sostenibile», ma lo indica come uno degli obiettivi fondamentali d |
| eval-0044 | L4 | 1 | Il Piano Regolatore Generale del Comune è uno strumento “flessibile”? | Il Piano Regolatore Generale del Comune è uno strumento flessibile, in quanto può essere aggiornato e modificato in base alle esigenze evolutive del territorio, |
| eval-0045 | L1 | 0 | Che cosa sono le “piante officinali” ai sensi della Legge regionale 16 febbraio 2011, n. 2? | La Legge regionale 16 febbraio 2011, n. 2 intitola l'articolo 6 come «(Piante officinali ad uso erboristico, alimentare e domestico)», ma non fornisce una defin |
| eval-0046 | L2 | 0 | Di quali autorizzazioni ho bisogno per svolgere attività di coltivazione e di raccolta delle piante  | Per svolgere attività di raccolta ad uso commerciale di specie incluse negli allegati D ed E delle piante officinali è necessaria un'autorizzazione prevista dal |
| eval-0047 | L3 | 1 | Come intende procedere la Regione per favorire lo sviluppo e la qualificazione della produzione regi | La Regione, al fine di favorire lo sviluppo e la qualificazione della produzione regionale delle piante officinali, la struttura regionale competente in materia |
| eval-0050 | L2 | 0 | La Commissione regionale per il patrimonio culturale è permanente? | Il contesto normativo fornito non contiene informazioni esplicite sull'eventuale natura permanente o straordinaria della Commissione regionale per il patrimonio |
| eval-0053 | L1 | 1 | Che cosa indica la sigla BHV-1 nella Legge regionale 13 febbraio 2012, n. 4 | La sigla BHV-1 non è esplicitamente definita nella Legge regionale 13 febbraio 2012, n. 4; tuttavia, la legge la utilizza come riferimento a un virus rispetto a |
| eval-0055 | L3 | 0 | Quando l’infezione da virus BHV-1 è considerata limitata? | La legge regionale 13 febbraio 2012, n. 4, definisce un’azienda come 'positiva' quando presenta uno o più animali positivi al virus BHV-1, ma non fornisce una d |
| eval-0056 | L4 | 1 | Quali sono le sanzioni giuridiche previste se un’azienda di allevamenti non rispetta le prescrizioni | La Legge regionale 13 febbraio 2012, n. 4 prevede sanzioni amministrative per la violazione delle prescrizioni in essa contenute; le somme derivanti dall'applic |
| eval-0057 | L1 | 0 | Qual è la finalità della Legge regionale 29 luglio 2024, n. 14? | La finalità della Legge regionale 29 luglio 2024, n. 14 non è indicata nel contesto fornito. |
| eval-0058 | L2 | 0 | Fino a quale anno è previsto un contributo regionale a sostegno degli eventi di cui alla Legge regio | La Legge regionale 29 luglio 2024, n. 14, prevede attività promozionali dell'evento e dell'offerta turistica a decorrere dall'anno 2024, ma non indica un termin |
| eval-0059 | L3 | 0 | A quale altro atto giuridico rinvia la Legge regionale 29 luglio 2024, n. 14 per la disciplina degli | La Legge regionale 29 luglio 2024, n. 14 non rinvia ad altro atto giuridico per la disciplina degli interventi a sostegno dell’organizzazione, svolgimento e pro |
| eval-0060 | L4 | 0 | Con quali modalità viene erogato il contributo annuale di cui alla Legge regionale 29 luglio 2024, n | La Legge regionale 29 luglio 2024, n. 14 non è presente nel contesto fornito; tuttavia, per la Legge regionale 29 luglio 2024, n. 12 (vda:lr:2024-07-29:12), all |
| eval-0063 | L3 | 1 | Il proprietario o l’avente diritto dei fondi confinanti con la sede stradale regionale può aprire ca | No, è fatto divieto di aprire canali, fossi ed eseguire escavazioni nei terreni laterali alle strade regionali, salvo quanto previsto dall'articolo 13, che stab |
| eval-0064 | L4 | 1 | Cosa accade in caso di mancato pagamento del canone concessorio di cui alla Legge regionale 20 novem | In caso di mancato pagamento del canone concessorio, si verifica una violazione della normativa regionale che può comportare sanzioni amministrative, anche ai s |
| eval-0065 | L1 | 0 | A quali soggetti si applica la Legge regionale 30 maggio 2022, n. 10 | La Legge regionale 30 maggio 2022, n. 10 non è presente nel contesto fornito; l'unico riferimento alla legge regionale n. 11 del 30 maggio 2022 è nel comma a) d |
| eval-0066 | L2 | 0 | Quale organo regionale definisce le modalità applicative dell’accesso gratuito ai servizi di traspor | Nessuno degli organi regionali indicati nel contesto fornisce informazioni relative alla definizione delle modalità applicative dell’accesso gratuito ai servizi |
| eval-0067 | L3 | 0 | Quale potrebbe essere lo scopo di consentire ai profughi di guerra l’accesso gratuito ai servizi di  | Il contesto normativo fornito non contiene disposizioni specifiche riguardanti l'accesso gratuito ai servizi di trasporto pubblico per i profughi di guerra, né  |
| eval-0068 | L4 | 1 | Alla Legge regionale 30 maggio 2022, n. 10 si applica il periodo di 15 giorni di vacatio legis previ | No, alla Legge regionale 30 maggio 2022, n. 10 non si applica il periodo di 15 giorni di vacatio legis previsto dallo Statuto regionale, poiché la legge in ogge |
| eval-0069 | L1 | 0 | Qual è l’ente di governo della comunità locale nel sistema delle autonomie della Valle d’Aosta? | La Chambre è l'ente di governo della comunità locale nel sistema delle autonomie della Valle d’Aosta. |
| eval-0072 | L4 | 1 | Oltre allo Statuto comunale, qual è un altro atto normativo del Comune? | Oltre allo Statuto comunale, gli altri atti normativi del Comune devono conformarsi ad esso, come previsto dall'art. 32, comma 3, della Legge regionale 7 dicemb |
| eval-0074 | L2 | 1 | La Commissione regionale per il patrimonio culturale decade automaticamente con la fine della legisl | No, la Commissione regionale per il patrimonio culturale non decade automaticamente con la fine della legislatura, poiché non è prevista alcuna disposizione nel |
| eval-0077 | L1 | 0 | Cosa sono le Consorterie valdostane? | Il testo fornito non contiene informazioni sulle Consorterie valdostane. |
| eval-0079 | L3 | 1 | A chi si applica, se compatibile, il regime giuridico delle Consorterie? | Alle Unités des Communes valdôtaines si applicano, in quanto compatibili, le disposizioni sulle Consorterie. |
| eval-0081 | L1 | 0 | Che cos’è Finaosta S.p.A? | Nel contesto normativo fornito non è presente alcuna informazione riguardante Finaosta S.p.A. |
| eval-0083 | L3 | 1 | Quali interessi di mora sono dovuti anche durante il regime di sospensione del pagamento della quota | Durante il regime di sospensione del pagamento della quota capitale, la quota interessi è comunque dovuta e calcolata al tasso contrattuale sul debito residuo a |
| eval-0084 | L4 | 0 | Quante volte il mutuatario può chiedere la sospensione del pagamento della quota capitale delle rate | La Legge regionale 9 aprile 2021, n. 5 non prevede un limite numerico esplicito al numero di volte per cui il mutuatario può chiedere la sospensione del pagamen |
| eval-0085 | L1 | 0 | In caso di accesso ai fondi, cosa deve pagare il ricercatore o il concessionario ai proprietari dei  | Nessuna compensazione economica è prevista: il ricercatore o il concessionario deve soltanto notificare copia del permesso di ricerca o della concessione ai pro |
| eval-0086 | L2 | 0 | Per quale attività è richiesta l’autorizzazione della Giunta regionale ai sensi della legge regional | L’autorizzazione della Giunta regionale è richiesta per l’esercizio di attività economiche, commerciali o artigianali, nonché per l’apertura di stabilimenti, se |
| eval-0087 | L3 | 0 | La vigilanza sulla coltivazione delle cave, delle miniere e delle acque minerali naturali, di sorgen | Sì, la vigilanza sulla coltivazione delle cave, delle miniere e delle acque minerali naturali, di sorgente e termali è effettuata dalla struttura competente, co |
| eval-0088 | L4 | 0 | Qual è la ratio complessiva della Legge regionale 13 marzo 2008, n. 5? | La ratio complessiva della Legge regionale 13 marzo 2008, n. 5 non è esplicitamente indicata nel contesto fornito. |
| eval-0089 | L1 | 0 | Quale struttura ha istituito a Bruxelles la Regione Valle d’Aosta? | Il contesto fornito non contiene informazioni relative all'istituzione di una struttura da parte della Regione Valle d’Aosta a Bruxelles. |
| eval-0090 | L2 | 1 | Con quale strumento tecnico-giuridico la Regione Valle d’Aosta dà attuazione al diritto europeo? | La Regione Valle d’Aosta dà attuazione al diritto europeo mediante decreti regionali, che assumono la forma di atti normativi adottati in esecuzione di obblighi |
| eval-0091 | L3 | 1 | Il Presidente della Giunta regionale può impugnare direttamente dinanzi alla Corte di giustizia dell | No, il Presidente della Giunta regionale non può impugnare direttamente dinanzi alla Corte di giustizia dell’Unione europea un atto normativo europeo ritenuto i |
| eval-0092 | L4 | 1 | Quali potrebbero essere le ricadute positive per i cittadini valdostani dell’attività della Regione  | Le ricadute positive per i cittadini valdostani potrebbero includere una maggiore conoscenza delle istituzioni e delle politiche dell'Unione europea, nonché una |
| eval-0093 | L1 | 0 | Chi è l’ospite secondo la Legge regionale 18 luglio 2023, n. 11? | La Legge regionale 18 luglio 2023, n. 11 non definisce esplicitamente il termine «ospite». Tuttavia, l'art. 2, comma 1, prevede che «Ai fini della presente legg |
| eval-0095 | L3 | 0 | La Regione giova di un incremento delle imposte dall’attività di locazione breve a fini turistici? | No, la Regione non ne trae vantaggio diretto in termini di incremento delle imposte, poiché le imposte (esclusa l’IVA) e le spese relative al conferimento sono  |
| eval-0096 | L4 | 0 | Chi affitta, per 15 giorni e per uso turistico, una stanza della propria casa di abitazione deve pag | No, chi affitta una stanza della propria casa di abitazione per uso turistico per 15 giorni non è soggetto alla disciplina dell'esercizio dell'attività di affit |
| eval-0097 | L1 | 1 | Qual è lo scopo della Legge regionale 21 aprile 2020, n. 5? | La Legge regionale 21 aprile 2020, n. 5 disciplina, tra l'altro, l'erogazione di contributi nel rispetto della disciplina europea in materia di aiuti di Stato e |
| eval-0098 | L2 | 0 | Quale strumento introduce la Legge regionale 21 aprile 2020, n. 5? | La Legge regionale 21 aprile 2020, n. 5 introduce l'esenzione dall'addizionale regionale all'IRPEF per l'anno 2020 e l'indennizzo ai titolari di contratti di lo |
| eval-0099 | L3 | 0 | Quali categorie di beneficiari sono coperte dalle misure di sostegno previste dalla Legge regionale  | La Legge regionale 21 aprile 2020, n. 5 non elenca esplicitamente le categorie di beneficiari nel brano fornito; tuttavia, il riferimento alla «categoria e posi |
