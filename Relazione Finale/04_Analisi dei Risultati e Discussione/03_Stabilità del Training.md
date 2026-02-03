# Stabilità del Training

La validazione della robustezza del processo di apprendimento è necessaria per verificare se la complessità dell'AAE introduce instabilità o varianza eccessiva nelle performance.

### Convergenza delle Loss
![Curve di Convergenza Loss su 5 Run](../assets/stability_loss.png)

L'analisi delle curve di training sulle 5 run indipendenti (`results/stability/loss_curves.png`) mostra un comportamento sorprendentemente stabile, nonostante la natura avversaria del modello.

*   **Reconstruction Loss ($L_{recon}$)**: Tutte le 5 esecuzioni convergono verso lo stesso asintoto finale ($MSE \approx 0.003$) con una varianza trascurabile ($\sigma < 1e-4$). Questo indica che la componente Autoencoder è robusta.
*   **Adversarial Losses**: Le curve di $L_{disc}$ e $L_{gen}$ mostrano le tipiche oscillazioni del gioco Min-Max, ma rimangono limitate (bounded) senza divergere o collassare a zero. Non si osservano segni di *Vanishing Gradient*, confermando che i learning rate differenziati ($1e-3$ vs $1e-4$) sono stati calibrati correttamente.

### Assenza di Mode Collapse
Un rischio frequente nelle GAN è il *Mode Collapse*, dove il generatore impara a produrre solo un limitato sottoinsieme di output che ingannano il discriminatore, ignorando la varietà dei dati reali. Nel nostro caso, la metrica MAUC stabile e alta su tutte le run (deviazione standard $\approx 0.002$) suggerisce che l'Encoder non ha subito collasso, mantenendo la capacità di mappare l'intera varietà delle transazioni in ingresso nello spazio latente.

In conclusione, sebbene l'AAE sia architettonicamente complesso, l'implementazione si è dimostrata **tecnicamente solida e riproducibile**. I limiti di performance discussi precedentemente non sono dovuti a fallimenti del training, ma a limiti intrinseci del *bias induttivo* del modello (la forzatura Gaussiana).
