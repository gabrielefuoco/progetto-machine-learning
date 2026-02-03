# Analisi di Stabilità (Training Dynamics)

La metodologia adottata per verificare la robustezza del processo di addestramento dell'AAE mira a scongiurare problemi tipici delle reti generative come il Mode Collapse o l'instabilità dei gradienti.

### Il Problema della Riproducibilità nelle GAN
Le architetture avversarie (GAN e AAE) sono notoriamente difficili da addestrare. L'equilibrio di Nash tra Generatore (Encoder) e Discriminatore è instabile: spesso una delle due reti prevale sull'altra troppo presto, portando a gradienti evanescenti (*Vanishing Gradients*) o a oscillazioni perpetue che impediscono la convergenza. Di conseguenza, presentare i risultati di una singola sessione di training ("cherry-picking") è scientificamente poco rigoroso.

### Protocollo di Validazione "5-Run"
Per garantire che i risultati ottenuti non fossero frutto del caso, abbiamo implementato un protocollo di validazione statistica robusto (`train_stability.py`):

1.  **Reiterazione**: L'intero processo di training (dall'inizializzazione casuale dei pesi alla convergenza finale) viene ripetuto per **5 volte indipendenti** (`N=5`).
2.  **Monitoraggio Fine-Grained**: A differenza del codice originale, che loggava le metriche sporadicamente, il nostro sistema registra i valori di Loss per ogni singola epoca delle 50 totali. Le metriche tracciate sono:
    *   $L_{recon}$: Errore di ricostruzione (MSE).
    *   $L_{disc}$: Capacità del Discriminatore di distinguere True/Fake.
    *   $L_{gen}$: Capacità dell'Encoder di ingannare il Discriminatore.
3.  **Tuning degli Iperparametri**: Per stabilizzare la dinamica, abbiamo adottato un Learning Rate conservativo uniforme di $1e-4$ per tutte le componenti (Encoder, Decoder, Discriminator). Questa scelta, abbinata a un Batch Size elevato (2048), ha permesso di ridurre la varianza degli aggiornamenti stocastici, prevenendo divergenze premature.

L'analisi della varianza tra le curve di apprendimento di queste 5 run ci permetterà di quantificare la stabilità del modello. Se le curve finali di $L_{recon}$ convergono tutte nello stesso range ristretto, potremo affermare con confidenza che l'architettura è robusta.
