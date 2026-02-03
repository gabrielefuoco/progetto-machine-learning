# Performance di Classificazione (MAUC)

I risultati quantitativi principali dell'analisi comparativa valutano l'efficacia delle diverse rappresentazioni latenti nel supportare la classificazione delle frodi.

### Metrica di Valutazione
La metrica adottata è l'**Area Under the ROC Curve (MAUC)** multi-classe. A differenza della semplice *Accuracy*, che in un dataset sbilanciato (99% normali) risulterebbe ingannevolmente alta (>0.99) anche per un classificatore "stupido" che predice sempre "tutto normale", l'AUC misura la capacità del modello di *ordinare* (rank) correttamente le frodi rispetto alle transazioni lecite, indipendentemente dalla loro frequenza relativa.

### Confronto: AAE vs Standard AE
La tabella sottostante riassume i MAUC score ottenuti addestrando i classificatori (SVM, RF) sia sull'output dell'Adversarial Autoencoder ($z_{AAE}$) sia su quello dell'Autoencoder Standard ($z_{AE}$), mantenendo fissa la dimensionalità latente ($d=10$).

| Classificatore | AAE (Best $\tau=20$) | Standard AE (Baseline) | Delta |
| :--- | :---: | :---: | :---: |
| **Random Forest** | 0.9687 | **0.9841** | +1.54% |
| **SVM (RBF Kernel)** | 0.9950 | **0.9962** | +0.12% |

### Analisi dei Risultati
I dati evidenziano un risultato controintuitivo rispetto alle premesse teoriche del paper originale:

1.  **Superiorità dello Standard AE**: In entrambe le configurazioni di classificazione, l'Autoencoder Standard, privo di qualsiasi regolarizzazione avversaria, produce rappresentazioni latenti che portano a performance di classificazione superiori. Sebbene il delta sulla SVM sia marginale (+0.12%), il distacco sulla Random Forest è netto.
2.  **Inefficacia della Gaussianizzazione**: L'ipotesi che forzare i dati in una distribuzione Gaussiana (AAE) ne migliorasse la separabilità si è rivelata non ottimale per questo specifico dataset. Al contrario, lasciare che la rete apprenda la propria topologia "naturale" (Standard AE) sembra preservare meglio le sfumature necessarie a distinguere le frodi più sottili.

Questo risultato suggerisce che la rigida struttura imposta dal Discriminatore dell'AAE potrebbe, in effetti, "schiacciare" eccessivamente le informazioni, sovrapponendo parzialmente le frodi ai dati leciti nelle regioni ad alta densità della Gaussiana, un fenomeno che esploreremo visivamente nel prossimo paragrafo tramite t-SNE.
