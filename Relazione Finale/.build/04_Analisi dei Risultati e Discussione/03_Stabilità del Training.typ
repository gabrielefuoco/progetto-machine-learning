= Stabilità del Training
<stabilità-del-training>
In questo paragrafo si analizza la robustezza del processo di
apprendimento, verificando se la complessità dell'AAE introduce
instabilità o varianza eccessiva nelle performance.

=== Convergenza delle Loss
<convergenza-delle-loss>
#figure(image("../assets/stability_loss.png"),
  caption: [
    Curve di Convergenza Loss su 5 Run
  ]
)

L'analisi delle curve di training sulle 5 run indipendenti
(`results/stability/loss_curves.png`) mostra un comportamento
sorprendentemente stabile, nonostante la natura avversaria del modello.
\* #strong[Reconstruction Loss ($L_(r e c o n)$)];: Tutte le 5
esecuzioni convergono verso lo stesso asintoto finale
($M S E approx 0.003$) con una varianza trascurabile
($sigma < 1 e - 4$). Questo indica che la componente Autoencoder è
robusta. \* #strong[Adversarial Losses];: Le curve di $L_(d i s c)$ e
$L_(g e n)$ mostrano le tipiche oscillazioni del gioco Min-Max, ma
rimangono limitate (bounded) senza divergere o collassare a zero. Non si
osservano segni di #emph[Vanishing Gradient];, confermando che i
learning rate differenziati ($1 e - 3$ vs $1 e - 4$) sono stati
calibrati correttamente.

=== Assenza di Mode Collapse
<assenza-di-mode-collapse>
Un rischio frequente nelle GAN è il #emph[Mode Collapse];, dove il
generatore impara a produrre solo un limitato sottoinsieme di output che
ingannano il discriminatore, ignorando la varietà dei dati reali. Nel
nostro caso, la metrica MAUC stabile e alta su tutte le run (deviazione
standard $approx 0.002$) suggerisce che l'Encoder non ha subito
collasso, mantenendo la capacità di mappare l'intera varietà delle
transazioni in ingresso nello spazio latente.

In conclusione, sebbene l'AAE sia architettonicamente complesso,
l'implementazione si è dimostrata #strong[tecnicamente solida e
riproducibile];. I limiti di performance discussi precedentemente non
sono dovuti a fallimenti del training, ma a limiti intrinseci del
#emph[bias induttivo] del modello (la forzatura Gaussiana).
