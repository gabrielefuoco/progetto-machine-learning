= Appendice B: Iperparametri
<appendice-b-iperparametri>
Si riportano di seguito le tabelle dei parametri utilizzati per
garantire la riproducibilità degli esperimenti finali.

=== Adversarial Autoencoder (AAE) Architecture
<adversarial-autoencoder-aae-architecture>
#figure(
  align(center)[#table(
    columns: (16.67%, 20.83%, 20.83%, 20.83%, 20.83%),
    align: (left,center,center,center,center,),
    table.header([Componente], [Layer], [Neuroni (Input -\>
      Output)], [Attivazione], [Parametri],),
    table.hline(),
    [#strong[Encoder];], [Dense 1], [618 -\> 1000], [LeakyReLU
    (0.2)], [619,000],
    [], [Dense 2], [1000 -\> 1000], [LeakyReLU (0.2)], [1,001,000],
    [], [Linear], [1000 -\> 10 ($z$)], [None (Linear)], [10,010],
    [#strong[Decoder];], [Dense 1], [10 -\> 1000], [LeakyReLU
    (0.2)], [11,000],
    [], [Dense 2], [1000 -\> 1000], [LeakyReLU (0.2)], [1,001,000],
    [], [Output], [1000 -\> 618], [Sigmoid (per valori
    \[0,1\])], [618,618],
    [#strong[Discriminator];], [Dense 1], [10 -\> 1000], [LeakyReLU
    (0.2)], [11,000],
    [], [Dense 2], [1000 -\> 1000], [LeakyReLU (0.2)], [1,001,000],
    [], [Output], [1000 -\> 1], [Sigmoid], [1,001],
  )]
  , kind: table
  )

=== Protocollo di Training (Stability Analysis)
<protocollo-di-training-stability-analysis>
#figure(
  align(center)[#table(
    columns: (33.33%, 33.33%, 33.33%),
    align: (left,left,left,),
    table.header([Parametro], [Valore], [Note],),
    table.hline(),
    [#strong[Optimizer];], [Adam], [Standard per GAN],
    [#strong[Learning Rate (All)];], [$1 times 10^(- 4)$], [Uniforme per
    stabilità garantita],
    [#strong[Batch Size];], [2048], [Ottimizzato per GPU e stima
    gradienti],
    [#strong[Epoche];], [50], [Convergenza tipica \~30 epoche],
    [#strong[Loss Function];], [MSE (Recon) + BCELoss (Disc)], [\-],
    [#strong[Prior Dist];], [Mistura di Gaussiane
    ($tau = 5 \, 10 \, 20$)], [$sigma = 1$, $mu$ equidistanti],
  )]
  , kind: table
  )

=== Classificatori (Fine-Tuning)
<classificatori-fine-tuning>
#figure(
  align(center)[#table(
    columns: (50%, 50%),
    align: (left,left,),
    table.header([Modello], [Iperparametri Chiave],),
    table.hline(),
    [#strong[Naive Bayes];], [`GaussianNB(var_smoothing=1e-9)`],
    [#strong[Random Forest];], [`n_estimators=100`, `max_depth=None`,
    `criterion='gini'`],
    [#strong[SVM];], [`C=1.0`, `kernel='rbf'`, `gamma='scale'`],
    [#strong[t-SNE];], [`perplexity=30`, `n_iter=1000`,
    `method='barnes_hut'`],
  )]
  , kind: table
  )
