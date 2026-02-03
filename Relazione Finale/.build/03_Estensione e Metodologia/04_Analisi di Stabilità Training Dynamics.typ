= Analisi di Stabilità (Training Dynamics)
<analisi-di-stabilità-training-dynamics>
In questo paragrafo si descrive la metodologia adottata per verificare
la robustezza del processo di addestramento dell'AAE e scongiurare
problemi tipici delle reti generative come il #emph[Mode Collapse] o
l'instabilità dei gradienti.

=== Il Problema della Riproducibilità nelle GAN
<il-problema-della-riproducibilità-nelle-gan>
Le architetture avversarie (GAN e AAE) sono notoriamente difficili da
addestrare. L'equilibrio di Nash tra Generatore (Encoder) e
Discriminatore è instabile: spesso una delle due reti prevale sull'altra
troppo presto, portando a gradienti evanescenti (#emph[Vanishing
Gradients];) o a oscillazioni perpetue che impediscono la convergenza.
Di conseguenza, presentare i risultati di una singola sessione di
training ("cherry-picking") è scientificamente poco rigoroso.

=== Protocollo di Validazione "5-Run"
<protocollo-di-validazione-5-run>
Per garantire che i risultati ottenuti non fossero frutto del caso,
abbiamo implementato un protocollo di validazione statistica robusto
(`train_stability.py`):

+ #strong[Reiterazione];: L'intero processo di training
  (dall'inizializzazione casuale dei pesi alla convergenza finale) viene
  ripetuto per #strong[5 volte indipendenti] (`N=5`).
+ #strong[Monitoraggio Fine-Grained];: A differenza del codice
  originale, che loggava le metriche sporadicamente, il nostro sistema
  registra i valori di Loss per ogni singola epoca delle 50 totali. Le
  metriche tracciate sono:
  - $L_(r e c o n)$: Errore di ricostruzione (MSE).
  - $L_(d i s c)$: Capacità del Discriminatore di distinguere True/Fake.
  - $L_(g e n)$: Capacità dell'Encoder di ingannare il Discriminatore.
+ #strong[Tuning degli Iperparametri];: Per stabilizzare la dinamica,
  abbiamo adottato un Learning Rate conservativo uniforme di $1 e - 4$
  per tutte le componenti (Encoder, Decoder, Discriminator). Questa
  scelta, abbinata a un Batch Size elevato (2048), ha permesso di
  ridurre la varianza degli aggiornamenti stocastici, prevenendo
  divergenze premature.

L'analisi della varianza tra le curve di apprendimento di queste 5 run
ci permetterà di quantificare la stabilità del modello. Se le curve
finali di $L_(r e c o n)$ convergono tutte nello stesso range ristretto,
potremo affermare con confidenza che l'architettura è robusta.
