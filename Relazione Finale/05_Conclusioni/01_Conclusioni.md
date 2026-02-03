# Conclusioni

Il presente lavoro di tesi ha esplorato l'efficacia degli Adversarial Autoencoders (AAE) nel contesto della rilevazione di frodi finanziarie, confrontandoli sistematicamente con architetture più semplici (Autoencoder Standard) e con le baseline supervisionate classiche.

### Sintesi dei Risultati
La nostra indagine sperimentale ha portato a conclusioni inaspettate ma tecnicamente solide:

1.  **L'AAE non è la Panacea**: Nonostante la sofisticazione teorica e la capacità di imporre una struttura gaussiana allo spazio latente, l'AAE ha mostrato performance di classificazione inferiori ($MAUC \approx 0.96$) rispetto a un Autoencoder Standard non regolarizzato ($MAUC \approx 0.98$) su questo specifico dataset.
2.  **Topologia vs Regolarità**: L'analisi qualitativa tramite t-SNE ha rivelato che la "forma sferica" imposta dall'AAE tende a comprimere eccessivamente le informazioni, sovrapponendo le frodi marginali ai dati leciti. Al contrario, lo Standard AE preserva meglio la struttura naturale delle anomalie.
3.  **Robustezza Operativa**: Abbiamo dimostrato che l'approccio non supervisionato è robusto fino a un tasso di inquinamento (frodi nel training) del 2%, oltre il quale subentra un rapido degrado (Pollution).

### Contributi Implementativi
Al di là dei risultati algoritmici, il progetto lascia in eredità un'infrastruttura moderna e scalabile per la ricerca futura:

*   Miglioramento dell'efficienza computazionale (**>100x speedup**) grazie alla migrazione su **RAPIDS cuML** e GPU.
*   Risoluzione dei problemi di memoria (OOM) tramite l'uso di matrici sparse e caricamento batch-wise ottimizzato.
*   Un ambiente **Docker** riproducibile che astrae la complessità della configurazione CUDA.

In conclusione, sebbene l'AAE si confermi un'architettura affascinante per compiti generativi, nel dominio dell'Anomaly Detection finanziaria la semplicità di un Autoencoder ben ingegnerizzato, unita a pipeline di dati efficienti, si è dimostrata la strategia vincente.
