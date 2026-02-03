= Relazione Tecnica Dettagliata - Estensione Progetto AAE Fraud Detection
<relazione-tecnica-dettagliata---estensione-progetto-aae-fraud-detection>
== 1. Introduzione
<introduzione>
Il presente documento descrive in dettaglio le specifiche tecniche,
l'architettura matematica e i protocolli di validazione adottati
nell'estensione del progetto "Fraud Detection with Adversarial
Autoencoders". L'obiettivo è fornire una granularità tale da garantire
la totale riproducibilità scientifica degli esperimenti.

#line(length: 100%)

== 2. Analisi Architetturale (Teoria)
<analisi-architetturale-teoria>
#strong[Obiettivo];: Implementazione rigorosa della dinamica avversaria
Min-Max.

L'architettura (definita in `Encoder.py`, `Decoder.py`,
`Discriminator.py` e orchestrata in `train_aae_stability.py`) si basa su
tre reti neurali.

=== 2.1 Specifiche dei Modelli
<specifiche-dei-modelli>
Tutti i layer intermedi utilizzano l'attivazione #strong[LeakyReLU] con
`negative_slope=0.4` per prevenire il problema del "dying ReLU".
L'inizializzazione dei pesi segue il metodo #strong[Xavier Uniform];.

==== A. Encoder (Generatore) $Q \( z \| x \)$
<a.-encoder-generatore-qzx>
Mappa lo spazio delle feature $x$ nello spazio latente $z$. \*
#strong[Input];: Dimensione variabile (data dalla somma delle feature
categoriche one-hot e numeriche log-scaled). \* #strong[Hidden Layer 1];:
Dense (256 neuroni) + LeakyReLU(0.4) \* #strong[Hidden Layer 2];: Dense
(64 neuroni) + LeakyReLU(0.4) \* #strong[Hidden Layer 3];: Dense (16
neuroni) + LeakyReLU(0.4) \* #strong[Hidden Layer 4];: Dense (4 neuroni)
\+ LeakyReLU(0.4) \* #strong[Output Layer];: Dense (Latent Dim = 10
neuroni) + LeakyReLU(0.4) \* #emph[Nota];: L'output dell'encoder passa
attraverso un'ulteriore non-linearità, differendo da implementazioni VAE
classiche che usano output lineari per $mu$ e $sigma$.

==== B. Decoder $P \( x \| z \)$
<b.-decoder-pxz>
Ricostruisce $x$ partendo da $z$. \* #strong[Input];: Latent Dim (10).
\* #strong[Struttura Simmetrica];: Dense 10 $arrow.r$ 4 $arrow.r$ 16
$arrow.r$ 64 $arrow.r$ 256. \* #strong[Output Layer];: Dense (Original
Dim) + #strong[Sigmoid];. \* #emph[Nota];: L'uso della Sigmoide finale
implica che i dati in input debbano essere normalizzati nel range \[0,
1\] (come garantito dal preprocessing).

==== C. Discriminatore $D \( z \)$
<c.-discriminatore-dz>
Classificatore binario che stima la probabilità che un campione $z$
provenga dalla distribuzione prior $P \( z \)$. \* #strong[Input];:
Latent Dim (10). \* #strong[Hidden Layer 1];: Dense(256) +
LeakyReLU(0.4). \* #strong[Hidden Layer 2];: Dense(16) + LeakyReLU(0.4).
\* #strong[Hidden Layer 3];: Dense(4) + LeakyReLU(0.4). \*
#strong[Output Layer];: Dense(1) + #strong[Sigmoid] (Probabilità
scalare).

=== 2.2 Formulazione Matematica delle Loss
<formulazione-matematica-delle-loss>
Il training procede per epoche e batch (minibatch SGD), ottimizzando due
obiettivi in sequenza:

+ #strong[Reconstruction Loss ($L_(r e c o n)$)];:
  $ L_(r e c o n) = 1 / N sum_(i = 1)^N \| \| x_i - D \( E \( x_i \) \) \| \|^2 $
  Implementata come `nn.MSELoss()`. #emph[Optimizer];: Adam
  ($alpha = 1 e - 3$).

+ #strong[Adversarial Regularization];: Giochi Min-Max tra Encoder ($E$)
  e Discriminatore ($D i s c$).

  - #strong[Discriminator Loss ($L_(d i s c)$)];: Massimizza la capacità
    di distinguere True ($z tilde.op P \( z \)$) da Fake ($E \( x \)$).
    $ L_(d i s c) = - \[ bb(E)_(z tilde.op P \( z \)) log \( D i s c \( z \) \) + bb(E)_(x tilde.op P \( d a t a \)) log \( 1 - D i s c \( E \( x \) \) \) \] $
    Implementata come somma di due `nn.BCELoss`. #emph[Optimizer];: Adam
    ($alpha = 1 e - 5$, ridotto per stabilità).
  - #strong[Generator Loss ($L_(g e n)$)];: L'Encoder cerca di ingannare
    il Discriminatore.
    $ L_(g e n) = - bb(E)_(x tilde.op P \( d a t a \)) log \( D i s c \( E \( x \) \) \) $
    Implementata come `nn.BCELoss` con target=1. #emph[Optimizer];: Adam
    ($alpha = 1 e - 3$).

#line(length: 100%)

== 3. Riproducibilità (Baseline)
<riproducibilità-baseline>
#strong[Script];: `reproduce_baseline.py`

Il processo di riproducibilità è stato ingegnerizzato per garantire
l'indipendenza dagli ambienti locali: 1. #strong[Download];: Recupero
automatico di `fraud_dataset_v2.csv` (GitHub Raw). 2. #strong[Wrapper];:
Esecuzione sandbox dei file originali (`MAUC_Exp.py`) tramite
`subprocess`, intercettando errori di importazione. 3.
#strong[Verifica];: Conferma che la struttura delle directory
`results/consolidated` corrisponda all'output atteso dal paper
originale.

#line(length: 100%)

== 4. Sperimentazione Critica (Extension)
<sperimentazione-critica-extension>
=== 4.1 Confronto Architetturale: AAE vs Standard AE
<confronto-architetturale-aae-vs-standard-ae>
#strong[Script];: `train_standard_ae.py` Isolamento della componente
GAN. \* #strong[Setup];: Stessa identica architettura di Encoder/Decoder
(256-64-16-4-10) e stesso preprocessing
($l o g + m i n m a x s c a l i n g$). \* #strong[Differenza];:
Rimozione totale del ramo Discriminatore. \* #strong[Training];: Solo
$L_(r e c o n)$ con Adam ($l r = 1 e - 3$). \* #strong[Metriche];:
Valutazione post-hoc tramite `evaluate_comparative.py` che applica
classificatori (SVM, RF, NB) sulle feature latenti generate.

=== 4.2 Analisi della Stabilità
<analisi-della-stabilità>
#strong[Script];: `train_aae_stability.py` Analisi fine-grained della
convergenza. \* #strong[Logging];: Ad ogni epoca (non solo ogni 10),
vengono registrati i valori medi di $L_(r e c o n)$, $L_(d i s c)$,
$L_(g e n)$. \* #strong[Visualizzazione];: Il plot `loss_curves.png`
sovrappone $L_(d i s c)$ e $L_(g e n)$ per mostrare l'equilibrio di
Nash. \* #strong[Target];: Ci aspettiamo $L_(d i s c) approx 0.69$
($l n \( 2 \)$) e $L_(g e n) approx 0.69$ se l'equilibrio è perfetto (il
discriminatore tira a indovinare con probabilità 0.5).

=== 4.3 Studio dell'Imbalance
<studio-dellimbalance>
#strong[Script];: `train_imbalance.py` Stress test su rarefazione delle
anomalie. \* #strong[Manipolazione Dati];:
`python     df_global_red = df_global.sample(frac=0.5, random_state=1234)     df_local_red = df_local.sample(frac=0.5, random_state=1234)`
\* #strong[Dataset Resultante];: Il training set contiene il 100% dei
dati #emph[Regular] ma solo il 50% dei dati #emph[Fraud];. \*
#strong[Ipotesi];: Dato che l'AAE apprende la normalità
($P \( z \) tilde.op R e g u l a r$), ci aspettiamo che riducendo le
frodi la qualità della mappa normale non degradi, mantenendo alta la
capacità di detection (MAUC stabile).

=== 4.4 Analisi Qualitativa (t-SNE)
<analisi-qualitativa-t-sne>
#strong[Script];: `tSNE_experiments.py` \* #strong[Parametri t-SNE];: -
`n_components=2`: Proiezione planare. - `perplexity=10`: Parametro
critico per la conservazione della struttura locale (scelto basso dato
il numero di cluster attesi). \* #strong[Visualizzazione];: Scatterplot
colorato per label reale (Regular/Global/Local), permettendo di
verificare visivamente se le anomalie (specialmente quelle Locali, più
insidiose) si staccano dal "blob" principale delle transazioni lecite.

#line(length: 100%)

== 5. Riferimento Comandi
<riferimento-comandi>
#figure(
  align(center)[#table(
    columns: (33.33%, 33.33%, 33.33%),
    align: (left,left,left,),
    table.header([Step], [Comando], [Output Atteso],),
    table.hline(),
    [#strong[\1.
    Baseline];], [`python project_extension/reproduce_baseline.py`], [`results/consolidated/*.txt`],
    [#strong[\2. AE
    Standard];], [`python project_extension/train_standard_ae.py`], [`latent_data_sets/*standardAE*.txt`],
    [#strong[\3.
    Confronto];], [`python project_extension/evaluate_comparative.py`], [Log
    a terminale (MAUC scores)],
    [#strong[\4.
    Stabilità];], [`python project_extension/train_aae_stability.py`], [`results_stability/loss_curves.png`],
    [#strong[\5.
    Imbalance];], [`python project_extension/train_imbalance.py`], [`latent_data_sets/*imbalance50*.txt`],
    [#strong[\6.
    t-SNE];], [`python tSNE_experiments.py`], [`tsneResults/*.png`],
  )]
  , kind: table
  )
