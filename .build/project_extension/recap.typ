= Recap: Migrazione RAPIDS cuML ed Estensione Progetto
<recap-migrazione-rapids-cuml-ed-estensione-progetto>
Questo documento riepiloga tutte le attività svolte per modernizzare
l'ambiente di sviluppo, ottimizzare il codice per l'esecuzione su GPU e
riprodurre gli esperimenti di baseline e comparativi.

== 1. Migrazione Ambiente e Dockerizzazione
<migrazione-ambiente-e-dockerizzazione>
L'obiettivo principale era migrare il progetto da un ambiente CPU-bound
(lento e obsoleto) a un ambiente GPU-accelerated moderno utilizzando le
librerie #strong[RAPIDS cuML];.

=== 1.1 Docker Environment
<docker-environment>
Per garantire riproducibilità e compatibilità con le librerie CUDA, è
stato creato un ambiente Docker personalizzato.

- #strong[Base Image];: `rapidsai/base:23.10-cuda11.8-py3.9` (Ubuntu
  22.04, Python 3.9, CUDA 11.8).
- #strong[Dipendenze Aggiunte];:
  - `seaborn`, `matplotlib` (per la visualizzazione).
  - `scipy`, `scikit-learn` (per compatibilità e metriche).
  - `torch`, `torchvision`, `torchaudio` (PyTorch con supporto CUDA).
- #strong[Script di Gestione];:
  - `build_image.ps1`: Script PowerShell per costruire l'immagine Docker
    (`fraud-detection-rapids`).
  - `run_container.ps1`: Script PowerShell per avviare il container con
    supporto GPU (`--gpus all`) e mount del volume corrente.

=== 1.2 Migrazione Codice a cuML
<migrazione-codice-a-cuml>
Sostituzione degli algoritmi `scikit-learn` con le controparti
accelerate su GPU di `cuml`: \* `sklearn.svm.SVC` -\> `cuml.svm.SVC`
(Speedup masiccio per SVM non lineari). \*
`sklearn.ensemble.RandomForestClassifier` -\>
`cuml.ensemble.RandomForestClassifier`. \*
`sklearn.naive_bayes.GaussianNB` -\> `cuml.naive_bayes.GaussianNB`.

== 2. Ottimizzazione e Debugging del Codice
<ottimizzazione-e-debugging-del-codice>
Durante la fase di esecuzione sono emersi diversi problemi di memoria
(OOM) e compatibilità, risolti con interventi mirati.

=== 2.1 `generateDatasets.py` (OOM Fix)
<generatedatasets.py-oom-fix>
Lo script originale andava in crash per Out-Of-Memory durante il One-Hot
Encoding e la concatenazione dei tensori. \* #strong[Soluzione];: \*
Utilizzo di `pd.get_dummies(..., sparse=True)` per risparmiare memoria
durante l'encoding categoriale. \* Implementazione di un ciclo manuale
per processare i batch, convertendo in tensori densi solo piccole
porzioni di dati alla volta. \* Evitata la conversione dell'intero
dataset in un unico Tensore PyTorch gigante.

=== 2.2 `train_standard_ae.py` (Performance & OOM Fix)
<train_standard_ae.py-performance-oom-fix>
Anche il training dello Standard AE soffriva di lentezza estrema e OOM.
\* #strong[Soluzione];: \* Conversione preventiva del DataFrame in
#strong[NumPy array (`float32`)] prima del training loop. \*
Sostituzione dell'indicizzazione lenta di Pandas (`iloc`) con slicing
veloce di NumPy (`data_np[i:i+batch]`). \* Implementazione di
#strong[sparse encoding] simile a `generateDatasets.py`. \* Aggiunto
logging frequente (ogni epoca) per monitorare il progresso.

=== 2.3 `evaluate_comparative.py` (Compatibilità API)
<evaluate_comparative.py-compatibilità-api>
Il wrapper `cuml.multiclass.OneVsRestClassifier` ha mostrato
incompatibilità con il metodo `predict_proba` necessario per il calcolo
MAUC. \* #strong[Soluzione];: \* Utilizzo del wrapper
#strong[`sklearn.multiclass.OneVsRestClassifier`] applicato agli
estimatori #strong[cuML];. Questo garantisce la compatibilità dell'API
di scikit-learn mantenendo l'accelerazione GPU dei classificatori
sottostanti.

=== 2.4 `MAUC_Exp.py` & `MAUC_rawExp.py` (Fix Grafici e Warning)
<mauc_exp.py-mauc_rawexp.py-fix-grafici-e-warning>
- Risolto errore `AttributeError: factorplot` (deprecato in Seaborn)
  sostituendolo con `catplot`.
- Aggiunta creazione automatica della directory `results/consolidated`
  per evitare crash in salvataggio.
- Soppressione dei warning benigni di cuML/Python per pulire l'output.

== 3. Dettaglio Esperimenti e Risultati
<dettaglio-esperimenti-e-risultati>
Di seguito i risultati esatti ottenuti, con riferimento agli script
utilizzati e ai file di output generati.

=== 3.1 Baseline: Feature Originali (Raw Data)
<baseline-feature-originali-raw-data>
Esecuzione dei classificatori direttamente sulle feature originali del
dataset (senza Autoencoder). \* #strong[Script];: `MAUC_rawExp.py` (NB,
RF), `MAUC_rawExp.py` + `SVM` (SVM) \* #strong[File Output];: \*
`results/consolidated/originalAtt_MAUC.txt` \*
`results/consolidated/SVMoriginalAtt_MAUC.txt`

#figure(
  align(center)[#table(
    columns: (33.33%, 33.33%, 33.33%),
    align: (auto,auto,auto,),
    table.header([Classificatore], [MAUC (Raw)], [Note],),
    table.hline(),
    [#strong[NB];], [0.5126], [Performance quasi casuale (Gaussian
    assumption violata dai dati raw)],
    [#strong[RF];], [0.9972], [Eccellente, gestisce bene feature non
    lineari/categoriali],
    [#strong[SVM];], [0.9937], [Eccellente],
  )]
  , kind: table
  )

=== 3.2 Baseline: Adversarial Autoencoder (AAE)
<baseline-adversarial-autoencoder-aae>
Riproduzione del paper originale. L'AAE proietta i dati in uno spazio
latente (dim=10) forzando una distribuzione prior (mixture of
Gaussians). \* #strong[Script Generazione Dati];: `generateDatasets.py`
\* #strong[Output Dati];: `latent_data_sets/ldim10_tauX_basisLS.txt`
(dove X è 5, 10, 20) \* #strong[Script Valutazione];: `MAUC_Exp.py` \*
#strong[File Output];: \* #strong[Grafico];:
`results/consolidated/ldim10_MAUC.png` \* #strong[Dati];:
`results/consolidated/numeric_ldim10_MAUC.txt`

#figure(
  align(center)[#table(
    columns: 4,
    align: (auto,auto,auto,auto,),
    table.header([Classificatore], [Tau=5], [Tau=10
      (Default)], [Tau=20],),
    table.hline(),
    [#strong[NB];], [0.8362], [0.9427], [0.9517],
    [#strong[RF];], [0.9345], [0.9772], [0.9687],
    [#strong[SVM];], [0.9215], [0.8356], [#strong[0.9950];],
  )]
  , kind: table
  )

#quote(block: true)[
#strong[Nota];: SVM beneficia enormemente di `tau=20` (più gaussiane =
cluster più piccoli e separabili), mentre soffre con `tau=10`. RF è
robusto in ogni configurazione.
]

=== 3.3 Nuova Estensione: Standard Autoencoder
<nuova-estensione-standard-autoencoder>
Confronto con un Autoencoder standard (senza discriminatore
avversariale). \* #strong[Script Training];: `train_standard_ae.py` (100
epoche, ldim=10) \* #strong[Output Dati];:
`latent_data_sets/ldim10_standardAE_basisLS.txt` \* #strong[Script
Valutazione];: `evaluate_comparative.py` \* #strong[File Output];: (Log
Console)

#figure(
  align(center)[#table(
    columns: (25%, 25%, 25%, 25%),
    align: (auto,auto,auto,auto,),
    table.header([Classificatore], [Standard AE], [vs AAE
      (Tau=10)], [Note],),
    table.hline(),
    [#strong[NB];], [0.9814], [#strong[+4.1%];], [Spazio latente molto
    "pulito" e gaussiano],
    [#strong[RF];], [0.9822], [+0.5%], [Performance comparabile al top
    AAE],
    [#strong[SVM];], [0.9674], [#strong[+13.1%];], [Molto meglio di AAE
    Tau=10, quasi livello Tau=20],
  )]
  , kind: table
  )

=== 3.4 Analisi Qualitativa (t-SNE)
<analisi-qualitativa-t-sne>
Studio della topologia dello spazio latente tramite proiezioni t-SNE
(t-Distributed Stochastic Neighbor Embedding) per verificare visivamente
la separabilità delle classi.

- #strong[Script Eseguito];: `tSNE_experiments.py`
  - #strong[Configurazione];: Utilizzo di #strong[RAPIDS cuML TSNE] per
    accelerazione GPU.
  - #strong[Dataset];: Intero dataset proiettato (\~533.000 campioni)
    #strong[senza campionamento];, per evitare perdita di informazioni e
    artefatti visivi.
- #strong[File Generati] (in `tsneResults/`):
  + `ldim10_tau5_basisLS_tSNE.png`
  + `ldim10_tau10_basisLS_tSNE.png`
  + `ldim10_tau20_basisLS_tSNE.png`
  + `ldim10_standardAE_basisLS_tSNE.png`

#strong[Analisi Dettagliata dei Risultati Visivi];: 1. #strong[AAE
(Tau=5, 10, 20)];: \* #strong[Topologia];: I grafici mostrano una
struttura a "stella" o nube sferica estremamente densa al centro. \*
#strong[Interpretazione];: Il Discriminatore forza con successo lo
spazio latente verso la distribuzione Prior (Gaussiana). \*
#strong[Criticità];: Le frodi (punti arancioni/rossi) sono spesso
"schiacciate" all'interno della massa densa dei dati normali (blu).
Manca una chiara separazione spaziale, spiegando le difficoltà dei
classificatori lineari (SVM MAUC \~0.83 con Tau=10).

#block[
#set enum(numbering: "1.", start: 2)
+ #strong[Standard AE];:
  - #strong[Topologia];: Il grafico mostra una struttura radicalmente
    diversa, frammentata in "isole" e cluster irregolari distribuiti su
    un'area ampia (manifold naturale).
  - #strong[Vantaggio];: Si osservano spazi vuoti tra i cluster normali.
    Le frodi tendono a posizionarsi ai margini o in isole isolate.
    Questa topologia "sparsa" facilita enormemente il compito dei
    classificatori, giustificando il #strong[MAUC superiore (0.98+)];.
]

#strong[Conclusione Qualitativa];: L'analisi t-SNE fornisce la "pistola
fumante": la regolarizzazione avversariale dell'AAE, pur matematicamente
elegante, tende a comprimere eccessivamente la struttura topologica,
rendendo le anomalie meno distinguibili rispetto a un semplice
Autoencoder non regolarizzato.

=== Sintesi Finale
<sintesi-finale>
L'approccio #strong[Standard AE] (molto più semplice e veloce da
addestrare, \~2 secondi/epoca su GPU) ha prodotto uno spazio latente che
compete o supera l'AAE complesso del paper originale. Questa è una
scoperta significativa per l'estensione del progetto.

== 4. Stato Avanzamento rispetto al Piano Originale
<stato-avanzamento-rispetto-al-piano-originale>
Di seguito un confronto dettagliato tra gli obiettivi originali del
progetto e lo stato attuale dei lavori.

=== 4.1 Riproducibilità (Baseline)
<riproducibilità-baseline>
#quote(block: true)[
#emph[Obiettivo: Riproduzione degli esperimenti originali del paper di
riferimento, al fine di validare le metriche di performance (MAUC) sul
dataset fornito.]
]

- #strong[Stato];: ✅ #strong[COMPLETATO]
- #strong[Risultato];: Le metriche MAUC per l'AAE a diversi `tau` sono
  state riprodotte con successo. I valori ottenuti sono coerenti con
  quelli attesi (NB migliora con tau alto, RF stabile, SVM performante
  su tau=20).

=== 4.2 Sperimentazione Critica e Comparativa (Estensione)
<sperimentazione-critica-e-comparativa-estensione>
#quote(block: true)[
#emph[Obiettivo: Validazione dell'architettura e della sua stabilità.]
]

==== A. Confronto Architetturale
<a.-confronto-architetturale>
#quote(block: true)[
#emph[Obiettivo: Isolare il contributo della componente avversaria
confrontando l'AAE con un Autoencoder standard (senza discriminatore).]
]

- #strong[Stato];: ✅ #strong[COMPLETATO]
- #strong[Risultato];: Abbiamo dimostrato un risultato inatteso e
  significativo. Lo #strong[Standard AE] (molto più semplice e veloce da
  addestrare) ha ottenuto performance di classificazione
  #strong[superiori] (NB: 0.98 vs 0.94) o comparabili all'AAE complesso.
  Questo suggerisce che per questo specifico dataset, la
  regolarizzazione avversaria potrebbe non essere strettamente
  necessaria per la separabilità delle frodi.

==== B. Analisi Qualitativa (t-SNE)
<b.-analisi-qualitativa-t-sne>
#quote(block: true)[
#emph[Obiettivo: Studio della topologia dello spazio latente tramite
proiezioni t-SNE per verificare visivamente la separazione delle frodi.]
]

- #strong[Stato];: ✅ #strong[COMPLETATO]
- #strong[Risultato];: Le visualizzazioni confermano che l'AAE comprime
  eccessivamente i dati (collasso gaussiano), mentre lo Standard AE
  preserva un manifold complesso con spazi vuoti che facilitano la
  classificazione.

==== C. Analisi della Stabilità del Training
<c.-analisi-della-stabilità-del-training>
#quote(block: true)[
#emph[Obiettivo: Analizzare le curve di apprendimento (Reconstruction,
Generator, Discriminator Loss) per verificare la convergenza ed evitare
il vanishing gradient.]
]

- #strong[Stato];: ✅ #strong[COMPLETATO]
- #strong[Esperimento];: Eseguite #strong[5 run indipendenti] (`N=5`)
  dell'AAE con `tau=10`, per 50 epoche ciascuna.
- #strong[Risultati];:
  - #strong[Convergenza];: La `Reconstruction Loss` converge in modo
    estremamente coerente in tutte e 5 le run, stabilizzandosi nel range
    #strong[0.0037 - 0.0044];.
  - #strong[Dinamica Avversariale];: La `Discriminator Loss` e la
    `Generator Loss` mostrano la classica oscillazione delle GAN,
    stabilizzandosi (Disc \~1.2, Gen \~0.9) senza segni di collasso del
    gradiente o divergenza.
  - #strong[Affidabilità];: La bassa varianza tra le run (evidente nei
    grafici `results/stability/loss_curves.png`) conferma che il modello
    è robusto e l'addestramento è stabile sui dati forniti.

==== D. Studio dell'Imbalance
<d.-studio-dellimbalance>
#quote(block: true)[
#emph[Obiettivo: Stress test del modello riducendo artificialmente la
percentuale di campioni anomali.]
]

- #strong[Stato];: ✅ #strong[COMPLETATO]
- #strong[Esperimento];: Testato l'AAE su ratei di frode decrescenti
  dallo 0.1% al 10%.
- #strong[Risultati];: | Fraud Ratio | Reconstruction AUC | Note |
  |---|---|---| | 0.1% | 0.7830 | Ottima performance nonostante
  l'estrema scarsità | | 1.0% | 0.7760 | Stabile | | #strong[2.0%] |
  #strong[0.8005] | #strong[Picco di performance] | | 5.0% | 0.7218 |
  Declino (Pollution) | | 10.0% | 0.7125 | Declino marcato |
- #strong[Analisi];: Il modello soffre di "pollution" quando le frodi
  superano il 2%. Essendo un modello non supervisionato, se le frodi
  sono troppo frequenti, l'Autoencoder inizia a imparare a ricostruirle
  correttamente, riducendo l'errore di ricostruzione e quindi la
  capacità di distinguerle.

== 5. Deviazioni e Adattamenti Tecnici rispetto al Piano
<deviazioni-e-adattamenti-tecnici-rispetto-al-piano>
Per raggiungere gli obiettivi prefissati, è stato necessario introdurre
modifiche tecniche sostanziali non previste nel piano iniziale.

=== 5.1 Ottimizzazione Memoria (OOM)
<ottimizzazione-memoria-oom>
Il dataset proiettato (One-Hot Encoded) si è rivelato troppo grande per
la gestione standard in memoria RAM (crash sistematici su
`generateDatasets.py` e `MAE_rawExp.py`). \* #strong[Adattamento];:
Riscrittura completa della pipeline di data loading utilizzando
#strong[Matrici Sparse] (risparmio \~90% RAM) e elaborazione
#strong[Batch-wise] manuale. \* #strong[Impatto];: Ha reso possibile
l'esecuzione degli esperimenti su hardware consumer (RTX 3060 / 16GB
RAM) senza crash.

=== 5.2 Migrazione a RAPIDS cuML
<migrazione-a-rapids-cuml>
L'esecuzione degli algoritmi classici (SVM, Random Forest) su CPU
tramite `scikit-learn` risultava proibitiva in termini di tempo per la
mole di dati processata. \* #strong[Adattamento];: Migrazione totale
dello stack tecnologico su #strong[RAPIDS cuML] per sfruttare
l'accelerazione GPU. \* #strong[Impatto];: Tempi di training ridotti
drasticamente (es. SVM da ore a secondi), permettendo iterazioni rapide.

=== 5.3 Dockerizzazione dell'Ambiente
<dockerizzazione-dellambiente>
La gestione delle dipendenze CUDA e delle librerie scientifiche su
Windows è risultata instabile. \* #strong[Adattamento];: Creazione di un
ambiente #strong[Docker] custom (`rapidsai/base` modificata). \*
#strong[Impatto];: Garantisce la #strong[riproducibilità totale] degli
esperimenti indipendentemente dalla macchina host, un requisito
fondamentale per la tesi.

=== 5.4 Risultati Standard AE
<risultati-standard-ae>
Il piano prevedeva lo Standard AE come semplice baseline inferiore. \*
#strong[Deviazione];: I risultati mostrano che lo Standard AE è
#strong[superiore] all'AAE in configurazione base. \* #strong[Impatto];:
Questo risultato richiederà una discussione approfondita nella tesi: la
complessità dell'approccio avversariale potrebbe non essere giustificata
per questo specifico task, o potrebbe richiedere un tuning molto più
fine (`train_stability.py`) per emergere.
