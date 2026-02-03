= Design dello Stress Test (Imbalance)
<design-dello-stress-test-imbalance>
In questo paragrafo si definisce il protocollo sperimentale ideato per
valutare la robustezza dell'approccio non supervisionato in scenari di
estrema rarefazione delle anomalie.

=== Motivazione
<motivazione>
Uno dei limiti dei sistemi di #emph[Anomaly Detection] basati su
Autoencoder è la loro sensibilità alla composizione del training set. La
teoria impone che il modello venga addestrato su dati "solamente
normali" per apprendere a ricostruire bene la normalità e male le
anomalie. Tuttavia, in scenari reali non etichettati, il training set è
spesso "inquinato" da una piccola percentuale di frodi non rilevate
(#emph[Pollution];).

La domanda di ricerca a cui vogliamo rispondere è: #strong[Qual è il
punto di rottura del modello?] Fino a che percentuale di inquinamento
l'Autoencoder continua a distinguere correttamente le frodi, e quando
invece inizia a considerarle "normali"?

=== Protocollo di Data Augmentation/Pruning
<protocollo-di-data-augmentationpruning>
Per rispondere a questa domanda, abbiamo sviluppato un modulo dedicato
(`imbalance_study.py`) che manipola artificialmente la composizione del
dataset di training, creando 6 scenari distinti di #emph[Fraud Ratio];:
\* $0.1 %$ e $1.0 %$: Scenari di #strong[scarsità];, tipici di sistemi
molto sicuri. \* $2.0 %$: Scenario base (simile al dataset originale).
\* $5.0 %$ e $10.0 %$: Scenari di #strong[inquinamento elevato];,
simulando un attacco massiccio o un sistema di controllo fuori uso.

Per ogni scenario: 1. Si mantiene costante il numero di transazioni
#emph[Regular] ($N_(l e g i t) approx 530.000$). 2. Si campiona
casualmente un sottoinsieme di transazioni #emph[Fraud] tale da
raggiungere il target ratio desiderato (o si sovracampiona se
necessario). 3. Si esegue il training completo dell'AAE. 4. Si valuta
l'Area Under Curve (AUC) della Reconstruction Loss sul Test Set (che
rimane fisso per garantire la comparabilità).

Ci aspettiamo di osservare una curva a campana (o decrescente), che ci
permetterà di identificare il "Sweet Spot" operativo del modello e la
soglia critica di "Saturazione", oltre la quale le frodi diventano
indistinguibili.
