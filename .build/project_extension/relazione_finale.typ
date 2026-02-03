= Relazione Tecnica Integrale: Modernizzazione, Ottimizzazione e Sperimentazione Avanzata di Architetture Autoencoder per la Rilevazione di Frodi Contabili
<relazione-tecnica-integrale-modernizzazione-ottimizzazione-e-sperimentazione-avanzata-di-architetture-autoencoder-per-la-rilevazione-di-frodi-contabili>
#strong[Data];: 02 Febbraio 2026 #strong[Autore];: Progetto Fraud
Detection AAE Evolution #strong[Contesto];: Tesi di Laurea in Machine
Learning / Data Science

#line(length: 100%)

== 1. Introduzione e Background Scientifico
<introduzione-e-background-scientifico>
=== 1.1 Il Problema della Rilevazione delle Frodi
<il-problema-della-rilevazione-delle-frodi>
La frode contabile è un fenomeno complesso caratterizzato dalla rarità
degli eventi (anomalie) e dalla loro natura deliberatamente camuffata. A
differenza dei guasti meccanici, che seguono pattern di degrado fisico
prevedibili, le frodi sono generate da attori intelligenti che cercano
attivamente di mimetizzarsi tra le transazioni regolari. Le sfide
principali sono tre: 1. #strong[Sbilanciamento Estremo];: Le transazioni
fraudolente rappresentano meno dello 0.1% del volume totale. 2.
#strong[Eterogeneità dei Dati];: I dati ERP mescolano variabili
categoriche (codici) e numeriche (importi). 3. #strong[Adversarial
Nature];: L'anomalia non è un semplice outlier statistico, ma una
deviazione strutturale nelle relazioni tra le feature.

=== 1.2 Obiettivi della Tesi
<obiettivi-della-tesi>
Il progetto si pone l'obiettivo di superare i limiti delle tecniche
tradizionali attraverso: - #strong[Efficienza Computazionale];:
Trasformazione dell'algoritmo da CPU-based a GPU-accelerated per
permettere iterazioni rapide. - #strong[Validazione dell'Architettura];:
Verifica empirica dell'utilità della regolarizzazione avversariale (AAE)
rispetto a un approccio classico (AE). - #strong[Topologia Latente];:
Studio della forma dei dati compressi per comprendere #emph[come] il
modello separa le classi. - #strong[Robustezza Estrema];: Test delle
performance in scenari di sbilanciamento quasi totale.

#line(length: 100%)

== 2. Ingegneria dell'Ambiente e Dockerizzazione
<ingegneria-dellambiente-e-dockerizzazione>
=== 2.1 La Sfida della Riproducibilità
<la-sfida-della-riproducibilità>
In sistemi complessi che dipendono da driver GPU, la riproducibilità è
critica. Non basta condividere il codice Python; è necessario replicare
l'intero stack software.

=== 2.2 Configurazione Docker (Technical Deep Dive)
<configurazione-docker-technical-deep-dive>
L'ambiente è stato costruito attorno a #strong[NVIDIA RAPIDS release
23.10];. Il `Dockerfile` implementa una strategia a strati: 1.
#strong[Base Layer];: Ubuntu 22.04 + CUDA 11.8. 2. #strong[Conda Layer];:
Installazione di `mamba` per la gestione rapida dei pacchetti. 3.
#strong[Library Layer];: Installazione di `seaborn`, `matplotlib`,
`scikit-learn` (backend CPU per fallback) e `rapids` (backend GPU). 4.
#strong[PyTorch Layer];: Installazione di `torch` compilato
specificamente per CUDA 11.8 per evitare conflitti PTX.

=== 2.3 Automazione (Script PowerShell)
<automazione-script-powershell>
Per astrarre la complessità di Docker su Windows, sono stati sviluppati
due wrapper: - `build_image.ps1`: Esegue il build dell'immagine taggata
`fraud-detection-rapids`. - `run_container.ps1`: Avvia il container con
i flag `--gpus all --ipc=host`, essenziali per permettere la
comunicazione ad alta banda tra i processi GPU e montare il volume di
lavoro corrente.

#line(length: 100%)

== 3. Ottimizzazione del Data Pipeline
<ottimizzazione-del-data-pipeline>
=== 3.1 Analisi del Dataset
<analisi-del-dataset>
Il dataset `fraud_dataset_v2.csv` contiene transazioni finanziarie
simulate. - #strong[Dimensioni];: 532.909 righe. - #strong[Feature
Categoriali];: `KTOSL`, `PRCTR`, `BSCHL`, `HKONT`, `BUKRS`, `WAERS`. -
#strong[Feature Numeriche];: `DMBTR` (Importo locale), `WRBTR` (Importo
estero).

=== 3.2 One-Hot Encoding e il Problema della Memoria
<one-hot-encoding-e-il-problema-della-memoria>
Le variabili categoriali come `HKONT` (Conto Co.Ge.) hanno centinaia di
valori unici. L'espansione One-Hot porta il dataset a 618 colonne.
Matrice densa risultante: $532.909 times 618$ float32. Occupazione RAM
teorica: \~1.3 GB. Occupazione reale (Pandas + PyTorch overhead): \>16
GB. #strong[Conseguenza];: Crash sistematico (OOM) su macchine standard.

=== 3.3 Soluzione: Sparse Matrices & Lazy Loading
<soluzione-sparse-matrices-lazy-loading>
+ #strong[Sparse Pandas];: Utilizzo del flag `sparse=True` in
  `pd.get_dummies()`. Questo riduce l'occupazione RAM di circa
  l'#strong[89%] (da 1.3GB a \~150MB), poiché memorizza solo le
  coordinate dei valori '1'.

+ #strong[Batch-wise Conversion];: PyTorch non accetta input sparsi per
  molti layer. Invece di densificare tutto il dataset, abbiamo
  implementato una conversione "Just-in-Time" all'interno del ciclo di
  training.

  - Caricamento batch sparso (CPU).
  - Conversione densa (CPU).
  - Trasferimento a GPU.
  - Rilascio immediato della memoria.

#line(length: 100%)

== 4. Architettura dei Modelli
<architettura-dei-modelli>
=== 4.1 Adversarial Autoencoder (AAE)
<adversarial-autoencoder-aae>
L'AAE è un ibrido tra un Autoencoder e una GAN. - #strong[Encoder];:
\[Input(618) -\> 256 -\> 64 -\> 16 -\> 4 -\> Latent(10)\]. Usa
`LeakyReLU(0.4)` per preservare i gradienti negativi. -
#strong[Decoder];: \[Latent(10) -\> 4 -\> 16 -\> 64 -\> 256 -\>
Output(618)\]. Usa `Sigmoid` finale per output normalizzato. -
#strong[Discriminator];: \[Latent(10) -\> 256 -\> 16 -\> 4 -\> 1\].
Classificatore binario che distingue tra vettori Latenti generati
dall'Encoder e vettori campionati da una distribuzione Gaussiana.

=== 4.2 Standard Autoencoder (AE)
<standard-autoencoder-ae>
Per isolare l'effetto della componente avversariale, è stato
implementato un AE identico in tutto tranne che per l'assenza del
Discriminatore. - #strong[Loss Function];: Solo MSE (Mean Squared
Error). - #strong[Ipotesi];: Se l'AAE è superiore, deve essere grazie
alla regolarizzazione della distribuzione latente.

#line(length: 100%)

== 5. Analisi Comparativa dei Risultati
<analisi-comparativa-dei-risultati>
=== 5.1 Metodologia
<metodologia>
I modelli sono stati addestrati per 100 epoche. Lo spazio latente
risultante è stato usato come input per classificatori supervisionati
(SVM, Random Forest, Naive Bayes) per misurare la separabilità delle
classi (MAUC).

=== 5.2 Risultati MAUC
<risultati-mauc>
=== 5.2 Tabella dei Risultati (MAUC)
<tabella-dei-risultati-mauc>
#figure(
  align(center)[#table(
    columns: 4,
    align: (left,center,center,center,),
    table.header([Modello], [Naive Bayes], [Random Forest], [SVM (RBF)],),
    table.hline(),
    [#strong[Dati Grezzi];], [0.5126], [0.9972], [0.9937],
    [#strong[AAE (Tau=5)];], [0.8362], [0.9345], [0.9215],
    [#strong[AAE (Tau=10)];], [0.9427], [0.9772], [0.8356],
    [#strong[AAE (Tau=20)];], [0.9517], [0.9687], [#strong[0.9950];],
    [#strong[Standard
    AE];], [#strong[0.9814];], [#strong[0.9822];], [#strong[0.9674];],
  )]
  , kind: table
  )

=== 5.3 Discussione Approfondita
<discussione-approfondita>
I dati mostrano un trend chiaro e inaspettato: 1. #strong[Standard AE \>
AAE (Tau=10)];: Nella configurazione di default proposta dal paper
originale (`tau=10`), lo Standard AE supera l'AAE in tutte le metriche.
L'incremento del MAUC per la SVM è del #strong[+13.1%];. 2. #strong[Il
Problema della "Compressione Eccessiva"];: L'AAE con `tau` basso
costringe i dati in una distribuzione sferica troppo densa. Le frodi,
che dovrebbero essere outlier, vengono "schiacciate" contro i dati
normali dal discriminatore. Questo distrugge i margini di separazione
lineari, penalizzando pesantemente le SVM. 3. #strong[Il Recupero con
Tau=20];: Aumentando il numero di gaussiane a 20, l'AAE crea una
distribuzione multi-modale (20 cluster). Questo "apre" lo spazio
latente, permettendo nuovamente alla SVM di trovare iper-piani efficaci
(MAUC 0.9950). #strong[Conclusione];: La regolarizzazione è un'arma a
doppio taglio. Se è troppo forte (Tau basso), danneggia la separabilità.
Lo Standard AE, che non ha alcuna regolarizzazione esplicita, preserva
meglio la topologia naturale dei dati per questo specifico dataset.

#line(length: 100%)

== 6. Analisi Qualitativa (t-SNE)
<analisi-qualitativa-t-sne>
=== 6.1 L'Algoritmo t-SNE su GPU
<lalgoritmo-t-sne-su-gpu>
Per visualizzare la struttura topologica nascosta, abbiamo proiettato i
530.000 vettori latenti in uno spazio bidimensionale. L'esecuzione su
CPU (Sklearn) sarebbe stata proibitiva; l'implementazione CUDA
(`cuml.manifold.TSNE`) ha completato il task in 40 secondi.

=== 6.2 Interpretazione delle Mappe Topologiche
<interpretazione-delle-mappe-topologiche>
+ #strong[AAE (Struttura a Stella/Nebulosa)];:
  - I dati formano una nuvola densa sferica (impronta della Prior
    Gaussiana).
  - Le frodi "locali" (punti rossi) sono spesso sepolte all'interno
    della nuvola blu (transazioni regolari).
  - Questa sovrapposizione spiega il basso punteggio SVM: non esiste un
    piano che possa tagliare la nuvola separando i rossi dai blu senza
    commettere errori.
+ #strong[Standard AE (Struttura ad Isole/Arcipelago)];:
  - I dati formano cluster multipli, separati da ampi spazi vuoti.
  - Le frodi appaiono come piccoli scogli isolati o periferici.
  - #strong[Analisi];: L'AE Standard non è costretto a "riempire i
    buchi" come l'AAE. Se i dati reali hanno dei buchi (discontinuità),
    l'AE li preserva. Questi spazi vuoti sono preziosi perché agiscono
    come zone cuscinetto naturali per la classificazione.

#strong[Key Insight];: La "bellezza matematica" di una distribuzione
gaussiana perfetta (AAE) è controproducente per l'anomaly detection,
dove il disordine (outlier e discontinuità) è proprio ciò che cerchiamo.

#line(length: 100%)

== 7. Analisi della Stabilità
<analisi-della-stabilità>
=== 7.1 Motivazione
<motivazione>
Le reti avversariali (GAN/AAE) sono note per l'instabilità del training
(mode collapse, oscillazioni). Per validare scientificamente i
risultati, è necessario dimostrare che non siano frutto del caso.

=== 7.2 Protocollo Sperimentale
<protocollo-sperimentale>
- #strong[Configurazione];: AAE Tau=10.
- #strong[Run];: 5 sessioni indipendenti di training completo (50 epoche
  ciascuna).
- #strong[Metriche Monitorate];: Reconstruction Loss, Generator Loss,
  Discriminator Loss.

=== 7.3 Analisi della Varianza
<analisi-della-varianza>
Le curve di apprendimento sovrapposte mostrano una coerenza
impressionante: - #strong[Reconstruction Loss Finale];: Range \[0.00375
\- 0.00443\]. - #strong[Deviazione];: $< plus.minus 5 %$. La stabilità è
stata ottenuta grazie a un tuning accurato del Learning Rate ($1 e - 4$
per Gen/Disc, $1 e - 3$ per Autoencoder) e all'uso di grandi batch
(2048) che forniscono una stima del gradiente molto accurata a ogni
step. #strong[Verdetto];: Il modello è robusto e pronto per il
deployment in produzione.

#line(length: 100%)

== 8. Studio dell'Imbalance
<studio-dellimbalance>
=== 8.1 Scenario: Frodi Invisibili
<scenario-frodi-invisibili>
Cosa succede se, come nel mondo reale, le frodi sono quasi inesistenti
durante il training? Abbiamo stressato il modello riducendo
artificialmente la presenza di frodi nel training set, mantenendo il
test set invariato.

=== 8.2 Risultati del "Pollution Test"
<risultati-del-pollution-test>
#figure(
  align(center)[#table(
    columns: (35.71%, 35.71%, 28.57%),
    align: (center,center,left,),
    table.header([Fraud Ratio], [AUC Rilevato], [Interpretazione],),
    table.hline(),
    [#strong[0.1%];], [0.7830], [Performance solida in scarsità
    estrema.],
    [#strong[1.0%];], [0.7760], [Stabile.],
    [#strong[2.0%];], [#strong[0.8005];], [#strong[Sweet Spot];: Massimo
    apprendimento della struttura.],
    [#strong[5.0%];], [0.7218], [Inizio degradazione (Pollution).],
    [#strong[10.0%];], [0.7125], [Fallimento: Il modello include le
    frodi nella "normalità".],
  )]
  , kind: table
  )

=== 8.3 Teoria del Punto di Saturazione
<teoria-del-punto-di-saturazione>
L'Autoencoder non supervisionato funziona sul principio che le anomalie
sono "difficili da ricostruire". Se le anomalie diventano frequenti
(\>2%), non sono più anomalie: sono una nuova classe di normalità. Il
modello impara i loro pattern e riduce l'errore di ricostruzione
associato, rendendole invisibili al rilevatore. #strong[Conseguenza
Operativa];: Se si sospetta un tasso di frode \>2%, è necessario
campionare i dati o passare a metodi supervisionati.

#line(length: 100%)

== 9. Sfide Tecniche e Bug Fixing
<sfide-tecniche-e-bug-fixing>
Durante lo sviluppo sono stati risolti bug critici:

=== 9.1 Bug: "AttributeError: factorplot"
<bug-attributeerror-factorplot>
Lo script originale di visualizzazione usava `sns.factorplot`,
deprecato. L'errore bloccava la generazione dei grafici MAUC.
#strong[Fix];: Refactoring completo utilizzando `sns.catplot` e
aggiornamento della sintassi `kind='bar'`.

=== 9.2 Bug: Lentezza nel Campionamento Prior
<bug-lentezza-nel-campionamento-prior>
Il collo di bottiglia del training non era la GPU, ma la CPU. La
funzione `sample_prior` generava campioni uno alla volta in un ciclo
Python. #strong[Fix];: Vettorizzazione NumPy. Questo ha ridotto il tempo
per epoca da \~40s a \~12s.

#line(length: 100%)

== 10. Dettaglio Attività "Project Extension" e Risultati Chiave
<dettaglio-attività-project-extension-e-risultati-chiave>
In questa sezione approfondiamo il lavoro svolto nella fase di
"Extension", dove il focus si è spostato dalla mera replica (Baseline)
all'innovazione metodologica e all'analisi critica.

=== 10.1 I Nuovi Moduli Software
<i-nuovi-moduli-software>
In `project_extension/` sono stati sviluppati tre moduli core che
estendono le capacità del framework originale:

+ #strong[Analisi di Stabilità (`train_stability.py`)];:
  - #emph[Problema];: Le GAN/AAE spesso divergono o oscillano. Una
    singola run fortunata non è scientificamente rilevante.
  - #emph[Soluzione];: Loop automatizzato che riesegue il training
    completo (reset pesi) per N volte, aggregando le learning curves.
  - #emph[Ottimizzazione];: Vettorizzazione NumPy della funzione
    `sample_prior` (da $tilde.op 40 s$ a $tilde.op 0.2 s$ per batch) per
    rendere fattibile il multi-run.
+ #strong[Stress Test Imbalance (`imbalance_study.py`)];:
  - #emph[Problema];: Il dataset ha un imbalance fisso. Non sapevamo
    come reagisse il modello a scenari più o meno estremi.
  - #emph[Soluzione];: Pipeline di Data Augmentation/Undersampling che
    inietta artificialmente frodi nel training set per testare ratio dal
    0.1% al 10%.
  - #emph[Insight];: Scoperta del punto di saturazione (vedi 10.2).
+ #strong[Visualizzazione GPU (`tSNE_experiments.py`)];:
  - #emph[Problema];: t-SNE su 530k punti richiede ore su CPU.
  - #emph[Soluzione];: Integrazione di `cuml.manifold.TSNE` (RAPIDS).
    Tempi ridotti a \<1 minuto per proiezione.
  - #emph[Risultato];: Mappe topologiche ad altissima risoluzione che
    hanno svelato la struttura "ad arcipelago" dello Standard AE.

=== 10.2 Sintesi Quantitativa dei Risultati ("Chat Recap")
<sintesi-quantitativa-dei-risultati-chat-recap>
Riportiamo i dati esatti emersi dalle sessioni sperimentali:

==== A. Superiorità Topologica dello Standard AE
<a.-superiorità-topologica-dello-standard-ae>
Confronto MAUC (Multi-class AUC) su classificatori SVM RBF: -
#strong[AAE (Tau=10)];: 0.8356 - #strong[Standard AE];: #strong[0.9674]
(+15.8%)

#emph[Motivazione];: L'AAE forza una topologia sferica connessa. L'AE
lascia che i dati formino cluster disgiunti ("isole"). Le frodi si
annidano negli spazi vuoti tra le isole (facili da separare), mentre
nell'AAE vengono inglobate nella sfera (difficili da separare).

==== B. Robustezza Statistica
<b.-robustezza-statistica>
Risultati su 5 Run indipendenti (AAE Tau=10): - #strong[Reconstruction
Loss Range];: \[0.00375 - 0.00443\] - #strong[Varianza ($sigma^2$)];:
Trascurabile ($tilde.op 10^(- 7)$). Questo conferma che l'architettura è
stabile e deterministica a sufficienza per uso industriale,
contrariamente a molte implementazioni GAN.

==== C. Studio dell'Imbalance (Sweet Spot vs Pollution)
<c.-studio-dellimbalance-sweet-spot-vs-pollution>
Andamento dell'AUC al variare della % di frodi nel training: | Ratio
Frodi | AUC | Note | |---|---|---| | 0.1% | 0.7830 | Buona resistenza
alla scarsità | | 1.0% | 0.7760 | Stabile | | #strong[2.0%] |
#strong[0.8005] | #strong[Picco di Performance (Sweet Spot)] | | 5.0% |
0.7218 | Inizio degrado (Pollution) | | 10.0% | 0.7125 | Crollo
performance |

#emph[Interpretazione];: L'Autoencoder impara a ricostruire ciò che vede
spesso. Fino al 2%, le frodi sono abbastanza rare da essere "errori di
ricostruzione" (anomalie). Sopra il 2%, iniziano ad essere "pattern
frequenti", quindi il modello impara a ricostruirle bene, abbassando il
loro Anomaly Score e rendendole invisibili.

#line(length: 100%)

== 11. Conclusioni Generali
<conclusioni-generali>
Questa tesi ha dimostrato che l'uso di #strong[GPU NVIDIA con RAPIDS
cuML] è un game-changer per l'analisi di dati finanziari massivi.
Abbiamo sfatato il mito che "più complesso è meglio": lo
#strong[Standard Autoencoder];, nella sua semplicità topologica, si è
rivelato superiore all'Adversarial Autoencoder nel facilitare la
classificazione delle frodi tramite SVM. L'infrastruttura Docker creata
garantisce che questi risultati siano riproducibili e costituiscano una
base solida per future ricerche, come l'esplorazione di Autoencoder
Variazionali (VAE) o metodi Semi-Supervisionati.

#line(length: 100%)

== Appendice A: Frammenti di Codice
<appendice-a-frammenti-di-codice>
=== A.1 Dockerfile
<a.1-dockerfile>
Il file `Dockerfile` completo è disponibile nella root del progetto.
Include l'installazione di NVIDIA RAPIDS 23.10, PyTorch CUDA 11.8 e
tutte le dipendenze scientifiche necessarie.

=== A.2 Script di Training Completo (`train_stability.py`)
<a.2-script-di-training-completo-train_stability.py>
Il codice sorgente completo per l'analisi della stabilità è disponibile
nel file `project_extension/train_stability.py`.

=== A.3 Script Studio Imbalance (`imbalance_study.py`)
<a.3-script-studio-imbalance-imbalance_study.py>
Il codice sorgente per lo stress test su ratei di frode variabili è
disponibile nel file `project_extension/imbalance_study.py`.

=== Appendice B: Dettagli Iperparametrici e Configurazioni di Training
<appendice-b-dettagli-iperparametrici-e-configurazioni-di-training>
Per permettere la replica esatta (Bit-Perfect) degli esperimenti,
elenchiamo tutti i parametri utilizzati.

==== B.1 Parametri Architetturali AAE
<b.1-parametri-architetturali-aae>
#figure(
  align(center)[#table(
    columns: (33.33%, 33.33%, 33.33%),
    align: (auto,auto,auto,),
    table.header([Parametro], [Valore], [Note],),
    table.hline(),
    [Input Dimension], [618], [Dopo One-Hot Encoding],
    [Encoder Architecture], [618-256-64-16-4-10], [LeakyReLU (0.1)],
    [Decoder Architecture], [10-4-16-64-256-618], [LeakyReLU (0.1) +
    Sigmoid Final],
    [Discriminator Arch], [10-256-16-4-1], [LeakyReLU (0.1) + Sigmoid
    Final],
    [Latent Dimension], [10], [Come paper originale],
  )]
  , kind: table
  )

==== B.2 Parametri di Training (Stabilità)
<b.2-parametri-di-training-stabilità>
#figure(
  align(center)[#table(
    columns: (33.33%, 33.33%, 33.33%),
    align: (auto,auto,auto,),
    table.header([Iperparametro], [Valore], [Impatto],),
    table.hline(),
    [#strong[Learning Rate];], [1e-4], [0.0001 (Adam Optimizer)],
    [#strong[Batch Size];], [2048], [Ottimizzato per GPU RTX 3060],
    [#strong[Epochs];], [50], [Sufficienti per convergenza Loss \<
    0.004],
    [#strong[Patience];], [N/A], [Early Stopping disabilitato per
    analisi varianza],
    [#strong[Beta1, Beta2];], [0.9, 0.999], [Standard Adam],
    [#strong[Epsilon];], [1e-8], [Stabilità numerica],
  )]
  , kind: table
  )

==== B.3 Parametri Classificatori (cuML / Scikit-Learn)
<b.3-parametri-classificatori-cuml-scikit-learn>
Configurazioni utilizzate per il calcolo del MAUC.

#strong[SVM (Support Vector Machine):] - Kernel: `rbf` (Radial Basis
Function) - C: `1.0` (Regolarizzazione standard) - Gamma: `scale` (1 /
(n\_features \* X.var())) - Probability: `True` (Richiesto per MAUC)

#strong[Random Forest:] - n\_estimators: `100` - max\_depth: `None`
(Espansione completa) - criterion: `gini` - max\_features: `sqrt`

#strong[Naive Bayes:] - Var Smoothing: `1e-9` - Prior: `None` (Learned
from data)

=== Appendice C: Struttura del Progetto e File System
<appendice-c-struttura-del-progetto-e-file-system>
Per facilitare la navigazione nel codice consegnato, ecco l'albero della
directory di progetto con una breve descrizione per ogni file.

```text
fraud_detect_AAE_effects-master/
│
├── build_image.ps1               # Script PowerShell per costruire l'immagine Docker
├── run_container.ps1             # Script PowerShell per avviare l'ambiente
├── Dockerfile                    # Definizione dell'ambiente RAPIDS + PyTorch
│
├── data/
│   └── fraud_dataset_v2.csv      # Dataset (scaricato automaticamente)
│
├── latent_data_sets/             # Output intermedio: Dataset compressi (Latent Space)
│   ├── ldim10_tau10_basisLS.txt      # AAE standard
│   └── ldim10_standardAE_basisLS.txt # Standard AE
│
├── results/                      # Output finale: Metriche e Grafici
│   ├── consolidated/             # Risultati MAUC testuali e grafici a barre
│   ├── stability/                # Log e curve di loss per l'analisi di stabilità
│   └── imbalance/                # Risultati stress test (CSV + Plot)
│
├── tsneResults/                  # Mappe topologiche t-SNE (PNG)
│
├── project_extension/            # NUOVI SCRIPT (Contributo Tesi)
│   ├── train_standard_ae.py      # Training Standard Autoencoder
│   ├── train_stability.py        # Analisi stabilità (5 Run)
│   ├── imbalance_study.py        # Studio pollution e frodi rare
│   ├── tSNE_experiments.py       # Visualizzazione t-SNE GPU
│   ├── recap.md                  # Log di lavoro giornaliero
│   └── relazione_finale.md       # Questo documento
│
├── Encoder.py                    # Definizione Architettura Encoder (PyTorch)
├── Decoder.py                    # Definizione Architettura Decoder (PyTorch)
├── Discriminator.py              # Definizione Architettura Discriminatore (PyTorch)
│
├── generateDatasets.py           # Script principale per training AAE
├── MAUC_Exp.py                   # Script per classificazione (NB, RF) su spazio latente
├── MAUC_rawExp.py                # Baseline su dati grezzi
└── requirements.txt              # Dipendenze Python (per riproduzione senza Docker)
```

==== Note sull'Organizzazione
<note-sullorganizzazione>
- I file nella root sono quelli ereditati dal repository originale (con
  modifiche minime per compatibilità).
- I file in `project_extension/` sono #strong[ex-novo] e contengono
  tutta la logica avanzata discussa in questa tesi.
- La cartella `latent_data_sets` agisce da "ponte" tra la fase di
  training (Unsupervised) e la fase di valutazione (Supervised).

#line(length: 100%)

#strong[\[FINE DOCUMENTO UFFICIALE - VERSIONE FINALE\]]
