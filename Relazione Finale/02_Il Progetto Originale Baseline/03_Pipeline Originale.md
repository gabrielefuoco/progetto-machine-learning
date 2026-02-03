# Pipeline Originale

Il flusso di elaborazione dati implementato nel progetto originale si focalizza sulle strategie di encoding e sulla metodologia di training semi-supervisionata.

La pipeline originale, concepita per validare l'efficacia degli AAE, si articola in tre fasi sequenziali: Preprocessing, Unsupervised Pre-training e Supervised Fine-tuning.

### 1. Data Preprocessing
Data la natura eterogenea delle feature, è stato applicato un trattamento differenziato:

*   **One-Hot Encoding**: Le 6 variabili categoriche (`HKONT`, `WAERS`, ecc.) vengono trasformate in vettori binari ortogonali. Questa operazione espande drasticamente la dimensionalità del dataset, portandola da 8 colonne originali a **618 feature totali**.
*   **Log-Scaling**: Le variabili numeriche (`DMBTR`, `WRBTR`), che presentano una distribuzione "a coda lunga" (long-tail) tipica dei dati monetari (pochi importi enormi, molti piccoli), vengono trasformate logaritmicamente ($\log(x+1)$) e successivamente normalizzate nel range [0, 1] tramite Min-Max Scaling. Questo passaggio è cruciale per stabilizzare i gradienti della rete neurale e facilitare la convergenza.

### 2. Unsupervised Learning (AAE)
Il cuore della pipeline è l'addestramento dell'Adversarial Autoencoder su tutto il dataset (senza etichette di frode).

*   **Encoder Setup**: Mappa le 618 feature di input in un vettore latente $z$ di dimensione $d=10$.
*   **Prior Distribution**: Viene imposta una distribuzione *Mixture of Gaussians* (MoG) sullo spazio latente. Nel paper, vengono testati diversi numeri di componenti per la mistura ($\tau=5, 10, 20$). L'idea è che ogni componente iper-sferica della mistura possa catturare un "prototipo" o cluster naturale di transazioni lecite.

### 3. Supervised Classification
Una volta addestrato l'AAE, l'intero dataset viene "passato" attraverso l'Encoder congelato per ottenere le rappresentazioni latenti $z_i \in \mathbb{R}^{10}$. Questi vettori compressi sostituiscono i dati grezzi come input per i classificatori finali:

*   **Support Vector Machine (SVM)**
*   **Random Forest (RF)**
*   **Naive Bayes (NB)**

Questa strategia trasforma il problema da un task ad alta dimensionalità (618 feature sparse) a uno a bassa dimensionalità (10 feature dense), con l'aspettativa che la regolarizzazione avversaria abbia già raggruppato le transazioni simili, facilitando il compito dei classificatori nel separare il "rumore" (frodi) dal segnale (legittimo).
