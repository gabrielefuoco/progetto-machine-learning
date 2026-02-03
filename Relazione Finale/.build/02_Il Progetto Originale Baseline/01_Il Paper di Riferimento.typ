= Il Paper di Riferimento
<il-paper-di-riferimento>
In questo paragrafo si analizza il lavoro scientifico originale che
funge da base per questo progetto, descrivendone gli obiettivi, la
metodologia proposta e i risultati dichiarati dagli autori.

Il presente lavoro di tesi prende le mosse dal paper #emph["Multi-Class
Mobile Money Service Financial Fraud Detection by Integrating Supervised
Learning with Adversarial Autoencoders"];. Questo studio propone un
framework ibrido per affrontare la rilevazione di frodi in dataset
finanziari fortemente sbilanciati, combinando la capacità di
apprendimento di rappresentazioni (representation learning) non
supervisionata degli Adversarial Autoencoders (AAE) con classificatori
supervisionati tradizionali.

La premessa fondamentale degli autori è che le tecniche di
classificazione standard (come Random Forest o SVM) falliscono quando
applicate direttamente su dati grezzi sbilanciati e ad alta
dimensionalità, a causa della difficoltà nel tracciare confini
decisionali efficaci. La soluzione proposta è una pipeline a due stadi:
1. #strong[Stage Non Supervisionato (AAE)];: Un AAE viene addestrato
sulle transazioni (prevalentemente lecite) per comprimerle in uno spazio
latente di dimensione ridotta ($d = 10$) che segue una distribuzione
Gaussiana (o mistura di Gaussiane). L'obiettivo è distillare le
caratteristiche salienti delle transazioni in una forma compatta e
regolarizzata. 2. #strong[Stage Supervisionato (Classificazione)];: Le
rappresentazioni latenti generate dall'Encoder vengono utilizzate come
feature di input per addestrare classificatori supervisionati.

Gli autori sostengono che la regolarizzazione imposta dall'AAE (che
forza i dati normali in cluster compatti) migliora significativamente la
separabilità delle classi rispetto all'uso dei dati grezzi o di tecniche
di riduzione dimensionale classiche come la PCA. Il paper riporta
risultati promettenti, misurati attraverso la metrica #strong[MAUC]
(Multi-class Area Under the Curve), evidenziando performance superiori
in particolare quando la distribuzione #emph[Prior] dello spazio latente
è modellata come una mistura di Gaussiane (Semi-Supervised AAE).
