= Riproduzione dei Risultati (Baseline)
<riproduzione-dei-risultati-baseline>
In questo paragrafo si espongono i risultati ottenuti replicando gli
esperimenti originali del paper di riferimento, al fine di stabilire una
baseline affidabile per le successive comparazioni e validare la
correttezza dell'implementazione.

La prima fase operativa del progetto ha riguardato la riproduzione
esatta (#emph[bit-perfect];) degli esperimenti descritti dagli autori,
utilizzando i parametri originali (`ldim=10`) e le stesse metriche di
valutazione.

=== Metodologia di Validazione
<metodologia-di-validazione>
È stato eseguito lo script `reproduce_baseline.py`, che orchestra
l'intero processo: 1. Generazione dei dataset latenti per diverse
configurazioni dell'AAE (variando il parametro $tau$, numero di
componenti della mistura di Gaussiane). 2. Addestramento dei
classificatori (NB, RF, SVM) su tali dataset. 3. Calcolo della metrica
#strong[MAUC] (Multi-class AUC).

=== Risultati Ottenuti
<risultati-ottenuti>
I risultati sperimentali confermano sostanzialmente i trend riportati
nel paper originale. Di seguito si riporta la tabella di sintesi dei
MAUC score ottenuti:

#figure(
  align(center)[#table(
    columns: (16.67%, 20.83%, 20.83%, 20.83%, 20.83%),
    align: (left,center,center,center,center,),
    table.header([Classificatore], [Dati Grezzi (Raw)], [AAE
      ($tau = 5$)], [AAE ($tau = 10$)], [AAE ($tau = 20$)],),
    table.hline(),
    [#strong[Naive Bayes];], [0.5126], [0.8362], [0.9427], [0.9517],
    [#strong[Random Forest];], [0.9972], [0.9345], [0.9772], [0.9687],
    [#strong[SVM];], [0.9937], [0.9215], [0.8356], [#strong[0.9950];],
  )]
  , kind: table
  )

=== Analisi della Baseline
<analisi-della-baseline>
L'osservazione dei dati grezzi evidenzia un punto fondamentale:
#strong[Random Forest e SVM sono già estremamente efficaci sui dati
originali] (MAUC \> 0.99). Questo suggerisce che, nonostante lo
sbilanciamento e l'eterogeneità, le feature originali contengono già
segnali discriminanti forti.

Tuttavia, l'esperimento conferma che l'AAE è in grado di distillare
queste informazioni in sole 10 dimensioni, mantenendo performance
elevate. Si notano due comportamenti interessanti: 1. #strong[Naive
Bayes];: Ha performance quasi casuali sui dati grezzi (0.51), ma eccelle
sulle feature latenti (0.95), provando che l'AAE riesce a
"gaussianizzare" efficacemente i dati, soddisfacendo l'assunzione di
indipendenza del classificatore Bayesiano. 2. #strong[Sensibilità della
SVM];: La SVM soffre significativamente con $tau = 10$ (MAUC 0.83), ma
recupera performance eccellenti con $tau = 20$ (MAUC 0.995). Questo
indica che aumentando il numero di cluster (componenti della mistura),
si "apre" lo spazio latente rendendolo linearmente separabile, mentre
con $tau = 10$ i dati sono probabilmente troppo compressi o sovrapposti.

Questi risultati validano l'ambiente di test e forniscono il punto di
partenza per l'analisi critica della fase successiva.
