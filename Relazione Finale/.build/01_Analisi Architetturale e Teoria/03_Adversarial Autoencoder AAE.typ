= Adversarial Autoencoder (AAE)
<adversarial-autoencoder-aae>
In questo paragrafo si descrive nel dettaglio l'architettura Adversarial
Autoencoder (AAE), illustrando come integra i principi delle Generative
Adversarial Networks (GAN) per imporre una distribuzione regolare allo
spazio latente, e analizzando la dinamica di addestramento Min-Max.

L'Adversarial Autoencoder è un'architettura ibrida che estende il
classico Autoencoder aggiungendo un meccanismo di regolarizzazione
avversaria, proposti originariamente da Makhzani et al.
#cite(<makhzani2015adversarial>, form: "prose");. A differenza dei
Variational Autoencoders (VAE) che usano la divergenza KL analitica per
forzare la distribuzione latente, l'AAE utilizza una terza rete neurale,
il #strong[Discriminatore];, per guidare l'Encoder a produrre codici
latenti che seguono una distribuzione #emph[Prior] desiderata
$P \( z \)$ (tipicamente una mistura di Gaussiane).

Il modello è composto da tre reti neurali distinte: 1. #strong[Encoder
($Q$)];: Oltre a minimizzare l'errore di ricostruzione, agisce come il
#emph[Generatore] della GAN. Cerca di "ingannare" il discriminatore
producendo codici latenti $q \( z \| x \)$ indistinguibili dalla
distribuzione Prior. 2. #strong[Decoder ($P$)];: Funziona come nell'AE
standard, ricostruendo l'input $x$ dal codice latente $z$. 3.
#strong[Discriminatore ($D$)];: Un classificatore binario che prende in
input un vettore latente $z$ e tenta di determinare se è un codice
"falso" generato dall'Encoder o un codice "reale" campionato dalla
distribuzione Prior $P \( z \)$.

=== La Dinamica Min-Max (Teoria dei Giochi)
<la-dinamica-min-max-teoria-dei-giochi>
L'addestramento dell'AAE avviene in due fasi distinte per ogni batch di
dati, seguendo una logica competitiva (Min-Max Game):

#strong[Fase 1: Reconstruction] Si aggiornano solo l'Encoder e il
Decoder per minimizzare la #emph[Reconstruction Loss] classica. In
questa fase, il Discriminatore è congelato. L'obiettivo è garantire che
l'informazione necessaria a ricostruire l'input sia preservata nel
codice latente.
$ min_(Q \, P) L_(r e c o n) = \| \| x - P \( Q \( x \) \) \| \|^2 $

#strong[Fase 2: Regularization (Adversarial Step)] Si instaurata la
competizione tra Encoder e Discriminatore sullo spazio latente. \*
Prima, si aggiorna il #strong[Discriminatore] per massimizzare la sua
capacità di distinguere i campioni #emph[True] (dalla Prior) dai
campioni #emph[Fake] (dall'Encoder).
$ max_D bb(E)_(z tilde.op P \( z \)) \[ log D \( z \) \] + bb(E)_(x tilde.op P \( d a t a \)) \[ log \( 1 - D \( Q \( x \) \) \) \] $
\* Successivamente, si aggiorna il #strong[Generatore (Encoder)] per
minimizzare la probabilità che il Discriminatore riconosca i suoi output
come falsi (o equivalentemente, massimizzare la probabilità che il
Discriminatore sbagli).
$ min_Q bb(E)_(x tilde.op P \( d a t a \)) \[ log \( 1 - D \( Q \( x \) \) \) \] $

=== Implicazioni per l'Anomaly Detection
<implicazioni-per-lanomaly-detection>
Questa regolarizzazione ha un effetto profondo sulla topologia dello
spazio latente. Mentre un AE standard può sparpagliare i dati in cluster
irregolari (manifold learning), l'AAE riempie densamente lo spazio
target (es. una ipersfera gaussiana).

Teoricamente, questo dovrebbe facilitare la separazione delle frodi: se
l'Encoder è costretto a mappare i dati normali all'interno della
distribuzione Prior, le anomalie strutturali, che non può comprimere
efficacemente rispettando contemporaneamente il vincolo di ricostruzione
e quello avversario, dovrebbero essere proiettate in regioni a bassa
densità o ai margini della distribuzione, diventando più facilmente
identificabili da classificatori lineari o modelli basati sulla densità.
