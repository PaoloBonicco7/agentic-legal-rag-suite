# Revisione manuale dei riferimenti composti

Questa nota documenta due problemi osservati leggendo gli artefatti già prodotti
di `data/corpora/current-final`, snapshot `vda-efe319281d52e13a`, acquisito il
10 ottobre 2026. Non è una nuova esecuzione né una validazione giuridica.
Corpus, manifest, indice e gold restano invariati.

## LR10/2026, articolo 9, comma 1

Fonte: [pagina ufficiale](https://www.consiglio.vda.it/app/leggieregolamenti/dettaglio?pk_lr=11824&versione=V),
sorgente locale `data/snapshots/current_20261010/sources/11824.html`.

Il testo letto contiene:

> I Comuni, entro novanta giorni dalla data di approvazione delle deliberazioni della Giunta regionale di cui agli articoli 6, comma 3, e 13, comma 2, della l.r. 20/1999, come modificati, rispettivamente, dal comma 1 e dal comma 4 dell'articolo 5 della presente legge, provvedono conseguentemente ad adeguare le deliberazioni di cui all'articolo 11 della medesima l.r. 20/1999.

In `relations.jsonl`, il record `rel:2923813afe31549c180d734a` ha origine
`vda:lr:2026-06-17:10#art:9#p:1`, target `vda:lr:1999-08-02:20`, risoluzione
`confirmed`, evidenza nel blocco `vda:lr:2026-06-17:10#b:00131`, span `[3,378)`.
È una risoluzione alla legge; non documenta due collegamenti agli articoli 6
e 13. L'espansione attuale, limitata a target articolo, non ottiene quei due
articoli da questo edge.

Prossimo controllo manuale: distinguere i riferimenti agli articoli 6 e 13,
il riferimento interno all'articolo 5 e quello all'articolo 11; verificare
ciascuna identità e il suo ambito nella fonte. Non assumere che un unico edge
alla legge rappresenti tutti i riferimenti presenti nella frase.

## LR54/1998, articolo 21, comma 3

Fonte: testo conservato in `blocks.jsonl` e `relations.jsonl` del corpus sopra
indicato, documento `vda:lr:1998-12-07:54`.

Il testo letto contiene:

> Fermo restando quanto disposto dal comma 1 e dall'articolo 23 e nel rispetto del principio della separazione tra funzioni di direzione politica e funzioni di direzione amministrativa di cui agli articoli 4 e 5 della legge regionale 23 ottobre 1995, n. 45 (Riforma dell'organizzazione dell'Amministrazione regionale della Valle d'Aosta e revisione della disciplina del personale), lo statuto può attribuire al Consiglio la competenza di ulteriori atti, tra i quali, in particolare:

Il record `rel:7173e271632db8366652c082` ha origine
`vda:lr:1998-12-07:54#art:21#p:3`, target
`vda:lr:1995-10-23:45#art:23`, risoluzione `confirmed`; span
`vda:lr:1998-12-07:54#b:00393 [3,483)`.
Il collegamento associa l'articolo 23 alla LR45/1995, mentre il riferimento
esplicito a quella legge nella medesima frase riguarda gli articoli 4 e 5.
È un possibile aggancio errato da verificare, non un link approvato.

Prossimo controllo manuale: consultare il documento originale e i suoi link
HTML; separare il riferimento iniziale all'articolo 23 dalla menzione della
LR45/1995. Solo dopo la verifica stabilire i target e una regressione mirata.
Nessuna correzione o suite è stata eseguita dopo lo stop richiesto.

## Implicazione per il confronto con grafo

L'esistenza del nodo target e la validità degli span sono controlli strutturali;
non provano la correttezza semantica di un edge né la rilevanza del vicino per
una domanda. I due casi vanno verificati prima di utilizzare graph expansion
come evidenza sperimentale affidabile. Una eventuale correzione richiede un
nuovo corpus versionato: non si modifica quello congelato in-place.

La [guida al preprocessing](05_preprocessing_walkthrough.html) mostra invece
un'associazione plurima osservata e supportata: il backlink della nota (2) della
LR42/1992 produce due eventi sul solo articolo 3 e sul solo articolo 4. Questo
non certifica la correttezza generale delle relazioni di grafo.
