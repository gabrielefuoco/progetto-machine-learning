= Autoencoder (AE)
<autoencoder-ae>
In questo paragrafo si analizza l'architettura base dell'Autoencoder,
definendone il funzionamento matematico come algoritmo di compressione
dati e il suo utilizzo per la generazione di punteggi di anomalia basati
sull'errore di ricostruzione.

L'Autoencoder (AE) è una rete neurale non supervisionata addestrata per
copiare il proprio input $x$ nel proprio output $x'$, passando
attraverso un "collo di bottiglia" (bottleneck) che costringe la rete ad
apprendere una rappresentazione compressa e significativa dei dati.
L'architettura è composta da due funzioni principali:

+ #strong[Encoder] ($E$): Mappa l'input ad alta dimensionalità
  $x in bb(R)^d$ in uno spazio latente a bassa dimensionalità
  $z in bb(R)^p$, dove $p lt.double d$.
  $ z = E \( x \) = sigma \( W_e x + b_e \) $
+ #strong[Decoder] ($D$): Ricostruisce l'approssimazione dell'input $x'$
  partendo dal codice latente $z$.
  $ x' = D \( z \) = sigma \( W_d z + b_d \) $

L'obiettivo dell'addestramento è minimizzare la differenza tra l'input
originale e la sua ricostruzione, definita come #emph[Reconstruction
Loss] ($L$). Nel nostro caso, data la natura mista dei dati, utilizziamo
comunemente l'Errore Quadratico Medio (MSE):

$ L \( x \, x' \) = \| \| x - x' \| \|^2 $

=== L'Autoencoder come Anomaly Detector
<lautoencoder-come-anomaly-detector>
Il principio cardine dell'uso degli AE per la rilevazione frodi risiede
nell'assumption che #strong[le anomalie sono difficili da comprimere];.
Poiché l'AE viene addestrato prevalentemente su dati "normali" (che
costituiscono la stragrande maggioranza del dataset), esso imparerà a
comprimere e ricostruire con alta fedeltà i pattern tipici delle
transazioni lecite.

Al contrario, quando al modello viene presentata una transazione
fraudolenta (che possiede caratteristiche statistiche o correlazioni mai
viste durante il training), l'Encoder non riuscirà a mapparla
correttamente nel codice latente ottimizzato per la normalità, e il
Decoder fallirà nel ricostruirla accuratamente. Di conseguenza, il
valore della #emph[Reconstruction Loss] per quel campione sarà
significativamente più alto rispetto alla media.

Questo valore di Loss viene utilizzato direttamente come #strong[Anomaly
Score];: fissando una soglia $tau$, classifichiamo come anomalo
qualsiasi dato $x$ per cui $L \( x \, D \( E \( x \) \) \) > tau$.

Tuttavia, l'AE standard presenta una limitazione fondamentale: non
impone alcuna struttura regolare allo spazio latente $z$. La
distribuzione dei punti codificati può assumere forme arbitrarie e
irregolari, con ampi "buchi" o discontinuità. Questo rende lo spazio
latente poco adatto per l'interpolazione o per l'utilizzo come input per
classificatori secondari che assumono distribuzioni ben definite (come
le Gaussiane). È per risolvere questo limite topologico che si introduce
l'architettura Adversarial Autoencoder.
