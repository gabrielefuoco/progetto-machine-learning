# Appendice B: Iperparametri

Si riportano di seguito le tabelle dei parametri utilizzati per garantire la riproducibilità degli esperimenti finali.

### Adversarial Autoencoder (AAE) Architecture
| Componente | Layer | Neuroni (Input -> Output) | Attivazione | Parametri |
| :--- | :---: | :---: | :---: | :---: |
| **Encoder** | Dense 1 | 618 -> 1000 | LeakyReLU (0.2) | 619,000 |
| | Dense 2 | 1000 -> 1000 | LeakyReLU (0.2) | 1,001,000 |
| | Linear | 1000 -> 10 ($z$) | None (Linear) | 10,010 |
| **Decoder** | Dense 1 | 10 -> 1000 | LeakyReLU (0.2) | 11,000 |
| | Dense 2 | 1000 -> 1000 | LeakyReLU (0.2) | 1,001,000 |
| | Output | 1000 -> 618 | Sigmoid (per valori [0,1]) | 618,618 |
| **Discriminator** | Dense 1 | 10 -> 1000 | LeakyReLU (0.2) | 11,000 |
| | Dense 2 | 1000 -> 1000 | LeakyReLU (0.2) | 1,001,000 |
| | Output | 1000 -> 1 | Sigmoid | 1,001 |

### Protocollo di Training (Stability Analysis)
| Parametro | Valore | Note |
| :--- | :--- | :--- |
| **Optimizer** | Adam | Standard per GAN |
| **Learning Rate (All)** | $1 \times 10^{-4}$ | Uniforme per stabilità garantita |
| **Batch Size** | 2048 | Ottimizzato per GPU e stima gradienti |
| **Epoche** | 50 | Convergenza tipica ~30 epoche |
| **Loss Function** | MSE (Recon) + BCELoss (Disc) | - |
| **Prior Dist** | Mistura di Gaussiane ($\tau=5,10,20$) | $\sigma=1$, $\mu$ equidistanti |

### Classificatori (Fine-Tuning)
| Modello | Iperparametri Chiave |
| :--- | :--- |
| **Naive Bayes** | `GaussianNB(var_smoothing=1e-9)` |
| **Random Forest** | `n_estimators=100`, `max_depth=None`, `criterion='gini'` |
| **SVM** | `C=1.0`, `kernel='rbf'`, `gamma='scale'` |
| **t-SNE** | `perplexity=30`, `n_iter=1000`, `method='barnes_hut'` |
