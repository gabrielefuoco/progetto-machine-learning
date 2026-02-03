= Modernizzazione dello Stack Tecnologico
<modernizzazione-dello-stack-tecnologico>
Questo capitolo documenta le sostanziali modifiche infrastrutturali
apportate al progetto originale per superare i limiti computazionali
delle implementazioni basate su CPU e garantire la riproducibilità
scientifica in ambienti eterogenei.

=== Limitazioni dell'Approccio Originale
<limitazioni-dellapproccio-originale>
L'implementazione di base faceva affidamento sulla libreria
`scikit-learn` per l'esecuzione di algoritmi classici come SVM e t-SNE.
Sebbene sia lo standard industriale per il Machine Learning su CPU,
questa scelta si è rivelata inadeguata per la scala del dataset in esame
(oltre 500.000 campioni ad alta dimensionalità). In particolare: \*
#strong[Tempi di Calcolo];: Il training di una SVM con kernel non
lineare (RBF) su CPU richiedeva ore, rendendo impossibile un tuning fine
degli iperparametri. \* #strong[Visualizzazione];: L'algoritmo t-SNE,
avente complessità $O \( N log N \)$, risultava di fatto ineseguibile
sul dataset completo, costringendo a campionamenti aggressivi che
perdevano dettagli topologici critici.

=== Migrazione a RAPIDS cuML
<migrazione-a-rapids-cuml>
Per risolvere questi colli di bottiglia, l'intero stack di elaborazione
è stato migrato su #strong[NVIDIA RAPIDS];, una suite di librerie
open-source che esegue algoritmi di Data Science interamente su GPU. \*
#strong[cuml (CUDA Machine Learning)];: Sostituto plug-and-play per
`scikit-learn`. L'adozione di `cuml.svm.SVC` e `cuml.manifold.TSNE` ha
ridotto i tempi di esecuzione da ore a secondi (speedup \> 100x),
permettendo di analizzare l'intero dataset senza downsampling. \*
#strong[Integrazione];: È stato sviluppato un layer di compatibilità
(wrapper) per permettere l'uso dei nuovi estimatori GPU all'interno
delle pipeline di valutazione esistenti (es. `OneVsRestClassifier`),
mantenendo inalterata la logica di calcolo del MAUC.

=== Containerizzazione (Docker)
<containerizzazione-docker>
Per garantire che i risultati fossero indipendenti dalla configurazione
hardware/driver della macchina ospite (Windows), è stato ingegnerizzato
un ambiente #strong[Docker] dedicato (`fraud-detection-rapids`).
L'immagine è costruita a strati: 1. #strong[Base System];: Ubuntu 22.04
LTS con driver CUDA 11.8. 2. #strong[Environment];: Conda/Mamba per la
gestione delle dipendenze Python conflittuali. 3. #strong[Scientific
Stack];: PyTorch (compilato per CUDA 11.8) affiancato alle librerie
RAPIDS 23.10.

Questa architettura "Portable Lab" elimina il problema del "funziona
sulla mia macchina", assicurando che ogni esperimento sia bit-perfect
riproducibile su qualsiasi workstation dotata di GPU NVIDIA.
