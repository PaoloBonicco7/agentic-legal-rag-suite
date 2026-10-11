# Verifica end-to-end — 11 ottobre 2026

## Protocollo congelato

Obiettivo: verificare la pipeline dopo le correzioni del primo pilot e misurare
segnali iniziali delle singole strategie Advanced. Nessuna promessa di guadagno
positivo. I risultati del [pilot precedente](07_luna_pilot.md) restano invariati.

Campione prima delle chiamate: `Random(20261012)`, popolazioni di QID ordinate,
livelli ordinati L1–L4, `sample(..., 1)` per livello sui 100 intenti originali.
QID: L1 `eval-0053`, L2 `eval-0002`, L3 `eval-0007`, L4 `eval-0004`.
Domande e gold legacy invariati; i quattro intenti non sono un test indipendente
mai esplorato. Non sono scelti per i risultati del precedente pilot.

Sette profili separati sullo stesso campione: No-RAG, Dense (RAG naive), Sparse,
Hybrid, Multi-query, Graph e Rerank. Le tre ablation Advanced partono da Hybrid:
nessuna combinazione di leve in questo test. Stesso generatore/giudice Luna,
reasoning low, massimo 4.096 output token, nessun retry o secondo modello.
Stesso corpus e indice, policy di cessazione e budget finale: 100 candidati,
4.096 token BGE, 12 blocchi; reranking sui primi 20, grafo un hop/due aggiunte,
multi-query originale e due varianti. Registrare il costo aggiuntivo delle leve.

Nuovi contratti: `answer-v3`, citazioni whitespace normalizzate solo verso un
unico span originale; `legal-judge-v2`, completezza e condizioni essenziali,
con stato della risposta visibile. Tutti i profili ricevono la stessa
indicazione di ambito Valle d'Aosta, senza
evidenze normative nel No-RAG; `rewrite-v2` vieta esiti e condizioni presunte.
Il confronto con il precedente pilot sarà descrittivo: prompt, campione e
corpus sono cambiati. Nessuna rivalutazione
retroattiva dei suoi score, né approvazione automatica del benchmark.

Limiti: massimo **72 nuove chiamate**, ledger comune cumulativo massimo
**0,50 EUR**, di cui 0,10 EUR di riserva; comprende anche il pilot precedente.
Solo `gpt-6-luna`; embedding locali riusati dalla cache verificata.
Massimo previsto per i sette profili: 28 generazioni, 28 giudizi, quattro
rewriting e quattro reranking (64 chiamate). Smoke separati: preflight,
CLI, UI, una proposta enrichment e tre generazioni MCQ (massimo altre sette).
Uno stop prima del dispatch conserva score mancanti e checkpoint.

## Verifiche richieste

- Suite offline completa dopo le correzioni di citazioni, judge, riferimenti
  composti, errori terminali, diagnostica provider e riproducibilità.
- Nuovo corpus dal medesimo snapshot congelato; tutte le quality gate, conteggi,
  differenze dal corpus precedente e ripetibilità dei JSONL.
- Nuovo indice immutabile con cache modello/testo e payload riconciliati;
  nessuna modifica all'indice del precedente pilot.
- Run comparabili dei sette profili; ranking, contesto, citazioni, errori,
  tempi, usage, costi e IC appaiati sul campione ristretto.
- Smoke reali delle interfacce e MCQ separati dalle metriche principali.
  Enrichment soltanto proposte non adottate; oracle e revisioni richiedono
  approvazione umana e sono verificati dai test di contratto, non inventati.
- Report statico offline con audit dei casi; notebook con switch disabilitati.
- Bundle e ripristino del corpus e dei run congelati, con checksum e provenienza
  del codice riconciliati; nessuna chiave, peso del modello o indice incluso.

## Risultati misurati

I sette run sono completi, con quattro score ciascuno: 26 giudizi e due zeri
operativi. Nessuno score manca e nessun giudizio previsto resta in sospeso.
Tutti gli output riusciti identificano `gpt-6-luna`; una chiamata
No-RAG fallita non restituisce identità o contatori. I gold non sono modificati.

| Profilo | Score 0 / 1 / 2 | Fully correct | Media 0–2 | Errori generativi | Chiamate | Stima EUR |
| --- | --- | --- | --- | --- | --- | --- |
| No-RAG | 2 / 0 / 2 | 2/4, 50% | 1,00 | 1 | 7 | 0,003810 |
| Dense | 0 / 1 / 3 | 3/4, 75% | 1,75 | 0 | 8 | 0,006101 |
| Sparse | 0 / 1 / 3 | 3/4, 75% | 1,75 | 0 | 8 | 0,006381 |
| Hybrid | 0 / 0 / 4 | 4/4, 100% | 2,00 | 0 | 8 | 0,005981 |
| Multi-query | 1 / 1 / 2 | 2/4, 50% | 1,25 | 0 | 12 | 0,006370 |
| Graph | 1 / 0 / 3 | 3/4, 75% | 1,50 | 1 | 7 | 0,005483 |
| Rerank | 1 / 0 / 3 | 3/4, 75% | 1,50 | 0 | 12 | 0,010091 |

| QID | No-RAG | Dense | Sparse | Hybrid | Multi-query | Graph | Rerank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| eval-0002 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| eval-0004 | 2 | 1 | 1 | 2 | 0 | 2 | 0 |
| eval-0007 | 0 | 2 | 2 | 2 | 2 | 2 | 2 |
| eval-0053 | 0 | 2 | 2 | 2 | 1 | 0 | 2 |

Hybrid supera numericamente No-RAG di 50 punti (IC95% appaiato [0; 100])
e Dense di 25 (IC95% [0; 75]); la sensibilità per tre gruppi di legge dà
rispettivamente [0; 100] e [0; 50]. Gli intervalli comprendono zero. Nessuna
configurazione è promossa: campione piccolo, già esplorato e giudice comune.
Le singole leve Advanced non migliorano Hybrid in questo test.

Tutti i RAG mostrano nel contesto un articolo atteso in 4/4 casi: è una proxy,
non una prova di sufficienza o correttezza. Le 57 citazioni accettate sono
letteralmente nel corpo del chunk; il denominatore esclude il payload Graph
respinto. Nessuna normalizzazione whitespace è stata necessaria in questi
output: il nuovo validatore è verificato anche dai test sintetici.

## Diagnosi e limiti

- **0004:** il giudice assegna da zero a due a risposte che includono gli enti
  locali e distinguono i ruoli. Il gold breve non è esclusivo; le differenze
  richiedono revisione umana. Il ranking corretto non spiega tutti i voti.
- **0007:** No-RAG si astiene; i RAG recuperano il massimale di 14.000 euro.
  Spesa ammissibile e modulazione restano condizioni, non un pagamento garantito.
  Luna applica in modo disomogeneo le cautele temporali nel giudizio di grounding.
- **0053:** lo zero No-RAG deriva da `provider_error`, senza causa specifica
  accertabile dagli artefatti; resta una riserva di 0,00246475 EUR. Lo zero
  Graph deriva da una citazione dell'header anziché del corpo. Entrambi restano
  insuccessi end-to-end, senza rigenerazione; non provano ignoranza del modello.
  Multi-query confonde virus e malattia pur avendo evidenza disponibile.
- Le riscritture analizzate non aggiungono condizioni: le regressioni non
  dimostrano automaticamente un cambiamento della domanda.
- Alcune fonti sanitarie presentano identità conflittuali 2002/2000 già
  segnalate nel corpus. Quote esatte e gate strutturali non riconciliano
  l'identità dell'atto. L'audit automatico le segnala per revisione della fonte.

I [commenti di caso](08_case_comments.jsonl) sono annotazioni automatiche
separate da gold e voti, portabili insieme alla documentazione.

## Verifiche tecniche e contabilità

313 test offline superati dopo le modifiche. Corpus: 3.176 documenti,
21.322 chunk e 17 gate; otto file canonici identici tra due esecuzioni.
Testi, chunk e status sono identici al corpus precedente. Relazioni:
119.630 → 119.398, irrisolte 2.265 → 4.010; 420 eventi cambiano soltanto
`related_document_id`. I due riferimenti composti corretti sono verificati
sui casi reali; non segue una certificazione globale delle relazioni.
Indice nuovo: 21.322 vettori riusati dalla cache verificata, nessuna inferenza
documentale aggiuntiva. Diagnostica locale: 16 record, zero chiamate LLM;
il grafo aggiunge unità reali in tutti e quattro i casi, senza benefici misurati.

Preflight, CLI e UI funzionali superati. Un batch enrichment propone quattro
fatti con evidenza esatta e nessuna adozione; il corpus resta immutato.
MCQ separato su 0002: 1/1 per No-RAG, Dense e Hybrid, senza giudice LLM.
Lo script sperimentale ha usato un attributo errato dopo tutte le chiamate
principali: la diagnostica è ripresa senza API né rigenerazioni dei run.

Totale nuovo: **69 chiamate, 0,046987625 EUR**; cumulativo con il primo pilot:
**115 chiamate, 0,068710875 EUR**, 114 settled e una riserva incerta.
I costi sono stime prudenti e includono i controlli separati, non fatture.
Nessun retry, modello diverso o embedding OpenAI. L'esecuzione delle altre
modalità provider resta coperta dai test offline; questo pilot usa solo Luna.

Report principale: 3.241 pagine HTML, 9.766 collegamenti locali; report MCQ:
3.185 pagine, 9.557 collegamenti. Checksum, destinazioni e risorse offline
superano i controlli statici. I tre notebook sono eseguiti integralmente nel
kernel Jupyter del progetto, con gli switch disabilitati e zero chiamate;
copie eseguite e anteprime HTML sono conservate separatamente.

Il pacchetto portabile contiene 7.795 file verificati: sorgenti, benchmark,
snapshot, corpus, sette run no-hint, tre MCQ e diagnostica retrieval. Ripristino
e rigenerazione del report dai sorgenti ripristinati riproducono tutti i 21
confronti, senza API. Pesi, indice, credenziali e ledger privato non sono inclusi.
Il fingerprint Python è `3cc5bb20e001962f46fd65f35386fb19a35c3ea2ac8b04c3f17e453536656b8a`;
le modifiche successive ai run riguardano soltanto la documentazione.
La lettura visiva nel browser resta da effettuare. Oracle e revisioni approvate
restano procedure con revisione umana.
