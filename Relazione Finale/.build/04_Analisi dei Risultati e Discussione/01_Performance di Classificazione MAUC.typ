= Performance di Classificazione (MAUC)
<performance-di-classificazione-mauc>
In questo paragrafo si presentano i risultati quantitativi principali
dell'analisi comparativa, valutando l'efficacia delle diverse
rappresentazioni latenti nel supportare la classificazione delle frodi.

=== Metrica di Valutazione
<metrica-di-valutazione>
La metrica adottata è l'#strong[Area Under the ROC Curve (MAUC)]
multi-classe. A differenza della semplice #emph[Accuracy];, che in un
dataset sbilanciato (99% normali) risulterebbe ingannevolmente alta
(\>0.99) anche per un classificatore "stupido" che predice sempre "tutto
normale", l'AUC misura la capacità del modello di #emph[ordinare] (rank)
correttamente le frodi rispetto alle transazioni lecite,
indipendentemente dalla loro frequenza relativa.

=== Confronto: AAE vs Standard AE
<confronto-aae-vs-standard-ae>
La tabella sottostante riassume i MAUC score ottenuti addestrando i
classificatori (SVM, RF) sia sull'output dell'Adversarial Autoencoder
($z_(A A E)$) sia su quello dell'Autoencoder Standard ($z_(A E)$),
mantenendo fissa la dimensionalità latente ($d = 10$).

#figure(
  align(center)[#table(
    columns: (21.05%, 26.32%, 26.32%, 26.32%),
    align: (left,center,center,center,),
    table.header([Classificatore], [AAE (Best $tau = 20$)], [Standard AE
      (Baseline)], [Delta],),
    table.hline(),
    [#strong[Random Forest];], [0.9687], [#strong[0.9841];], [+1.54%],
    [#strong[SVM (RBF Kernel)];], [0.9950], [#strong[0.9962];], [+0.12%],
  )]
  , kind: table
  )

=== Analisi dei Risultati
<analisi-dei-risultati>
I dati evidenziano un risultato controintuitivo rispetto alle premesse
teoriche del paper originale:

+ #strong[Superiorità dello Standard AE];: In entrambe le configurazioni
  di classificazione, l'Autoencoder Standard, privo di qualsiasi
  regolarizzazione avversaria, produce rappresentazioni latenti che
  portano a performance di classificazione superiori. Sebbene il delta
  sulla SVM sia marginale (+0.12%), il distacco sulla Random Forest è
  netto.
+ #strong[Inefficacia della Gaussianizzazione];: L'ipotesi che forzare i
  dati in una distribuzione Gaussiana (AAE) ne migliorasse la
  separabilità si è rivelata non ottimale per questo specifico dataset.
  Al contrario, lasciare che la rete apprenda la propria topologia
  "naturale" (Standard AE) sembra preservare meglio le sfumature
  necessarie a distinguere le frodi più sottili.

Questo risultato suggerisce che la rigida struttura imposta dal
Discriminatore dell'AAE potrebbe, in effetti, "schiacciare"
eccessivamente le informazioni, sovrapponendo parzialmente le frodi ai
dati leciti nelle regioni ad alta densità della Gaussiana, un fenomeno
che esploreremo visivamente nel prossimo paragrafo tramite t-SNE.
