# Relazione Tecnica Dettagliata - Estensione Progetto AAE Fraud Detection

## 1. Introduzione
Il presente documento descrive in dettaglio le specifiche tecniche, l'architettura matematica e i protocolli di validazione adottati nell'estensione del progetto "Fraud Detection with Adversarial Autoencoders". L'obiettivo è fornire una granularità tale da garantire la totale riproducibilità scientifica degli esperimenti.

---

## 2. Analisi Architetturale (Teoria)
**Obiettivo**: Implementazione rigorosa della dinamica avversaria Min-Max.

L'architettura (definita in `Encoder.py`, `Decoder.py`, `Discriminator.py` e orchestrata in `train_aae_stability.py`) si basa su tre reti neurali.

### 2.1 Specifiche dei Modelli
Tutti i layer intermedi utilizzano l'attivazione **LeakyReLU** con `negative_slope=0.4` per prevenire il problema del "dying ReLU". L'inizializzazione dei pesi segue il metodo **Xavier Uniform**.

#### A. Encoder (Generatore) $Q(z|x)$
Mappa lo spazio delle feature $x$ nello spazio latente $z$.
*   **Input**: Dimensione variabile (data dalla somma delle feature categoriche one-hot e numeriche log-scaled).
*   **Hidden Layer 1**: Dense (256 neuroni) + LeakyReLU(0.4)
*   **Hidden Layer 2**: Dense (64 neuroni) + LeakyReLU(0.4)
*   **Hidden Layer 3**: Dense (16 neuroni) + LeakyReLU(0.4)
*   **Hidden Layer 4**: Dense (4 neuroni) + LeakyReLU(0.4)
*   **Output Layer**: Dense (Latent Dim = 10 neuroni) + LeakyReLU(0.4)
*   *Nota*: L'output dell'encoder passa attraverso un'ulteriore non-linearità, differendo da implementazioni VAE classiche che usano output lineari per $\mu$ e $\sigma$.

#### B. Decoder $P(x|z)$
Ricostruisce $x$ partendo da $z$.
*   **Input**: Latent Dim (10).
*   **Struttura Simmetrica**: Dense 10 $\to$ 4 $\to$ 16 $\to$ 64 $\to$ 256.
*   **Output Layer**: Dense (Original Dim) + **Sigmoid**.
*   *Nota*: L'uso della Sigmoide finale implica che i dati in input debbano essere normalizzati nel range [0, 1] (come garantito dal preprocessing).

#### C. Discriminatore $D(z)$
Classificatore binario che stima la probabilità che un campione $z$ provenga dalla distribuzione prior $P(z)$.
*   **Input**: Latent Dim (10).
*   **Hidden Layer 1**: Dense(256) + LeakyReLU(0.4).
*   **Hidden Layer 2**: Dense(16) + LeakyReLU(0.4).
*   **Hidden Layer 3**: Dense(4) + LeakyReLU(0.4).
*   **Output Layer**: Dense(1) + **Sigmoid** (Probabilità scalare).

### 2.2 Formulazione Matematica delle Loss
Il training procede per epoche e batch (minibatch SGD), ottimizzando due obiettivi in sequenza:

1.  **Reconstruction Loss ($L_{recon}$)**:
    $$ L_{recon} = \frac{1}{N} \sum_{i=1}^{N} || x_i - D(E(x_i)) ||^2 $$
    Implementata come `nn.MSELoss()`.
    *Optimizer*: Adam ($\alpha=1e-3$).

2.  **Adversarial Regularization**:
    Giochi Min-Max tra Encoder ($E$) e Discriminatore ($Disc$).
    *   **Discriminator Loss ($L_{disc}$)**: Massimizza la capacità di distinguere True ($z \sim P(z)$) da Fake ($E(x)$).
        $$ L_{disc} = - [ \mathbb{E}_{z \sim P(z)} \log(Disc(z)) + \mathbb{E}_{x \sim P(data)} \log(1 - Disc(E(x))) ] $$
        Implementata come somma di due `nn.BCELoss`.
        *Optimizer*: Adam ($\alpha=1e-5$, ridotto per stabilità).
    *   **Generator Loss ($L_{gen}$)**: L'Encoder cerca di ingannare il Discriminatore.
        $$ L_{gen} = - \mathbb{E}_{x \sim P(data)} \log(Disc(E(x))) $$
        Implementata come `nn.BCELoss` con target=1.
        *Optimizer*: Adam ($\alpha=1e-3$).

---

## 3. Riproducibilità (Baseline)
**Script**: `reproduce_baseline.py`

Il processo di riproducibilità è stato ingegnerizzato per garantire l'indipendenza dagli ambienti locali:
1.  **Download**: Recupero automatico di `fraud_dataset_v2.csv` (GitHub Raw).
2.  **Wrapper**: Esecuzione sandbox dei file originali (`MAUC_Exp.py`) tramite `subprocess`, intercettando errori di importazione.
3.  **Verifica**: Conferma che la struttura delle directory `results/consolidated` corrisponda all'output atteso dal paper originale.

---

## 4. Sperimentazione Critica (Extension)

### 4.1 Confronto Architetturale: AAE vs Standard AE
**Script**: `train_standard_ae.py`
Isolamento della componente GAN.
*   **Setup**: Stessa identica architettura di Encoder/Decoder (256-64-16-4-10) e stesso preprocessing ($log + minmax scaling$).
*   **Differenza**: Rimozione totale del ramo Discriminatore.
*   **Training**: Solo $L_{recon}$ con Adam ($lr=1e-3$).
*   **Metriche**: Valutazione post-hoc tramite `evaluate_comparative.py` che applica classificatori (SVM, RF, NB) sulle feature latenti generate.

### 4.2 Analisi della Stabilità
**Script**: `train_aae_stability.py`
Analisi fine-grained della convergenza.
*   **Logging**: Ad ogni epoca (non solo ogni 10), vengono registrati i valori medi di $L_{recon}$, $L_{disc}$, $L_{gen}$.
*   **Visualizzazione**: Il plot `loss_curves.png` sovrappone $L_{disc}$ e $L_{gen}$ per mostrare l'equilibrio di Nash.
*   **Target**: Ci aspettiamo $L_{disc} \approx 0.69$ ($ln(2)$) e $L_{gen} \approx 0.69$ se l'equilibrio è perfetto (il discriminatore tira a indovinare con probabilità 0.5).

### 4.3 Studio dell’Imbalance
**Script**: `train_imbalance.py`
Stress test su rarefazione delle anomalie.
*   **Manipolazione Dati**:
    ```python
    df_global_red = df_global.sample(frac=0.5, random_state=1234)
    df_local_red = df_local.sample(frac=0.5, random_state=1234)
    ```
*   **Dataset Resultante**: Il training set contiene il 100% dei dati *Regular* ma solo il 50% dei dati *Fraud*.
*   **Ipotesi**: Dato che l'AAE apprende la normalità ($P(z) \sim Regular$), ci aspettiamo che riducendo le frodi la qualità della mappa normale non degradi, mantenendo alta la capacità di detection (MAUC stabile).

### 4.4 Analisi Qualitativa (t-SNE)
**Script**: `tSNE_experiments.py`
*   **Parametri t-SNE**:
    -   `n_components=2`: Proiezione planare.
    -   `perplexity=10`: Parametro critico per la conservazione della struttura locale (scelto basso dato il numero di cluster attesi).
*   **Visualizzazione**: Scatterplot colorato per label reale (Regular/Global/Local), permettendo di verificare visivamente se le anomalie (specialmente quelle Locali, più insidiose) si staccano dal "blob" principale delle transazioni lecite.

---

## 5. Riferimento Comandi

| Step | Comando | Output Atteso |
| :--- | :--- | :--- |
| **1. Baseline** | `python project_extension/reproduce_baseline.py` | `results/consolidated/*.txt` |
| **2. AE Standard** | `python project_extension/train_standard_ae.py` | `latent_data_sets/*standardAE*.txt` |
| **3. Confronto** | `python project_extension/evaluate_comparative.py` | Log a terminale (MAUC scores) |
| **4. Stabilità** | `python project_extension/train_aae_stability.py` | `results_stability/loss_curves.png` |
| **5. Imbalance** | `python project_extension/train_imbalance.py` | `latent_data_sets/*imbalance50*.txt` |
| **6. t-SNE** | `python tSNE_experiments.py` | `tsneResults/*.png` |
