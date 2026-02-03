# Ottimizzazione Data Pipeline (Memory Engineering)

Le sfide critiche legate alla gestione della memoria RAM emerse durante il preprocessing del dataset hanno richiesto soluzioni ingegneristiche specifiche per essere mitigate.

### Il Problema dell'Esplosione Dimensionale
Come anticipato nell'analisi del dataset, la presenza di feature categoriali ad alta cardinalità (in particolare `HKONT`, il codice conto generale) comporta un drastico aumento delle dimensioni dei dati quando viene applicato il One-Hot Encoding.
La trasformazione di 6 colonne categoriali in vettori binari porta il numero totale di feature da 8 a **618**.
Matematicamente, la dimensione della matrice dati $X$ diventa:
$$ 532.909 \text{ righe} \times 618 \text{ colonne} \times 4 \text{ byte (float32)} \approx 1.32 \text{ GB} $$

Sebbene 1.3 GB sembrino gestibili su macchine moderne, il sovraccarico introdotto dalle strutture dati di Pandas e la necessità di PyTorch di creare copie dei tensori per il calcolo dei gradienti saturano rapidamente la memoria RAM disponibile (>16 GB), portando a crash sistematici per *Out Of Memory* (OOM).

### Soluzione: Sparse Encoding & Batch Processing
Per rendere fattibile il training su hardware consumer, è stata implementata una strategia a due livelli:

1.  **Format Sparse in Memoria**:
    Invece di rappresentare gli zeri esplicitamente (che costituiscono >99% della matrice One-Hot), abbiamo utilizzato il formato sparso. In Pandas, questo si ottiene specificando `sparse=True` durante la creazione delle dummy variables.
    ```python
    pd.get_dummies(df, columns=[...], sparse=True)
    ```
    Questa tecnica riduce l'occupazione di memoria di un fattore $\sim 18x$ (da 1.32 GB a $\sim 75$ MB), memorizzando solo le coordinate e i valori degli elementi non nulli.

2.  **Conversione Densificata Just-in-Time (JIT)**:
    Le reti neurali dense (Fully Connected) richiedono input densi per eseguire le moltiplicazioni matriciali efficienti su GPU. Non potendo convertire l'intero dataset in denso (pena il ritorno al problema OOM), abbiamo implementato un caricatore dati custom che effettua la conversione **batch-wise**:

    *   Il dataset risiede in RAM in formato sparso compresso.
    *   Ad ogni iterazione del training loop, viene estratto un mini-batch (es. 2048 campioni).
    *   Solo questo piccolo sottoinsieme viene convertito in matrice densa e trasferito alla GPU (`.cuda()`).
    *   Dopo il passo di ottimizzazione, la memoria GPU viene liberata immediatamente.

Questo approccio ibrido ha permesso di addestrare modelli su milioni di parametri utilizzando un laptop standard con 16GB di RAM, senza alcun degrado nelle prestazioni del modello.
