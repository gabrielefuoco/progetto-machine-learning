= Robustezza all'Imbalance
<robustezza-allimbalance>
In questo paragrafo finale si analizzano i risultati dello #emph[stress
test] condotto variando la percentuale di frodi presenti nel training
set (Fraud Ratio), per comprendere i limiti operativi del modello.

=== Il Fenomeno della "Pollution"
<il-fenomeno-della-pollution>
#figure(image("../assets/imbalance_pollution.png"),
  caption: [
    Effetto Pollution: MAUC vs Fraud Ratio
  ]
)

Il grafico risultante dall'`imbalance_study.py` mostra un andamento non
lineare delle performance (AUC) al crescere della percentuale di frodi:

+ #strong[Zona di Scarsità (\< 1%)];: Con pochissime frodi (0.1% -
  0.5%), il modello performa bene ma con una varianza leggermente più
  alta. L'AAE riesce comunque a modellare la normalità dominata dai dati
  leciti.
+ #strong[Sweet Spot (\~2%)];: Si osserva un picco di performance
  attorno al 2% di frodi. Controintuitivamente, una piccola
  contaminazione sembra aiutare il Discriminatore a definire meglio i
  confini della distribuzione Prior, fungendo da regolarizzatore
  implicito.
+ #strong[Zona di Saturazione (\> 5%)];: Oltre il 5%, le performance
  crollano verticalmente. Al 10% di Fraud Ratio, il MAUC scende sotto
  0.85 per i classificatori lineari. Questo fenomeno, noto come
  #strong[Pollution];, indica che le anomalie sono diventate troppo
  frequenti per essere considerate tali. L'Autoencoder inizia ad
  apprendere anche i pattern delle frodi come "normali", riuscendo a
  ricostruirle con basso errore. Di conseguenza, l'Anomaly Score
  ($L_(r e c o n)$) perde potere discriminante.

=== Implicazioni Pratiche
<implicazioni-pratiche>
Questo risultato ha un'importante implicazione operativa: l'approccio
non supervisionato AAE è valido solo finché le frodi rimangono eventi
rari (\< 5%). In scenari di attacco massiccio (#emph[Fraud Spike];),
dove il volume di transazioni illecite esplode improvvisamente, il
sistema potrebbe, paradossalmente, diventare cieco, incorporando
l'attacco nella sua definizione di normalità. Per sistemi di produzione
robusti, sarebbe quindi necessario affiancare all'AAE un modulo di
monitoraggio del #emph[Drift] della distribuzione, per rilevare
cambiamenti repentini nella composizione dei dati.
