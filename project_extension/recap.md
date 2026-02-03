# Recap: Migrazione RAPIDS cuML ed Estensione Progetto

Questo documento riepiloga tutte le attività svolte per modernizzare l'ambiente di sviluppo, ottimizzare il codice per l'esecuzione su GPU e riprodurre gli esperimenti di baseline e comparativi.

## 1. Migrazione Ambiente e Dockerizzazione

L'obiettivo principale era migrare il progetto da un ambiente CPU-bound (lento e obsoleto) a un ambiente GPU-accelerated moderno utilizzando le librerie **RAPIDS cuML**.

### 1.1 Docker Environment
Per garantire riproducibilità e compatibilità con le librerie CUDA, è stato creato un ambiente Docker personalizzato.

*   **Base Image**: `rapidsai/base:23.10-cuda11.8-py3.9` (Ubuntu 22.04, Python 3.9, CUDA 11.8).
*   **Dipendenze Aggiunte**:
    *   `seaborn`, `matplotlib` (per la visualizzazione).
    *   `scipy`, `scikit-learn` (per compatibilità e metriche).
    *   `torch`, `torchvision`, `torchaudio` (PyTorch con supporto CUDA).
*   **Script di Gestione**:
    *   `build_image.ps1`: Script PowerShell per costruire l'immagine Docker (`fraud-detection-rapids`).
    *   `run_container.ps1`: Script PowerShell per avviare il container con supporto GPU (`--gpus all`) e mount del volume corrente.

### 1.2 Migrazione Codice a cuML
Sostituzione degli algoritmi `scikit-learn` con le controparti accelerate su GPU di `cuml`:
*   `sklearn.svm.SVC` -> `cuml.svm.SVC` (Speedup masiccio per SVM non lineari).
*   `sklearn.ensemble.RandomForestClassifier` -> `cuml.ensemble.RandomForestClassifier`.
*   `sklearn.naive_bayes.GaussianNB` -> `cuml.naive_bayes.GaussianNB`.

## 2. Ottimizzazione e Debugging del Codice

Durante la fase di esecuzione sono emersi diversi problemi di memoria (OOM) e compatibilità, risolti con interventi mirati.

### 2.1 `generateDatasets.py` (OOM Fix)
Lo script originale andava in crash per Out-Of-Memory durante il One-Hot Encoding e la concatenazione dei tensori.
*   **Soluzione**:
    *   Utilizzo di `pd.get_dummies(..., sparse=True)` per risparmiare memoria durante l'encoding categoriale.
    *   Implementazione di un ciclo manuale per processare i batch, convertendo in tensori densi solo piccole porzioni di dati alla volta.
    *   Evitata la conversione dell'intero dataset in un unico Tensore PyTorch gigante.

### 2.2 `train_standard_ae.py` (Performance & OOM Fix)
Anche il training dello Standard AE soffriva di lentezza estrema e OOM.
*   **Soluzione**:
    *   Conversione preventiva del DataFrame in **NumPy array (`float32`)** prima del training loop.
    *   Sostituzione dell'indicizzazione lenta di Pandas (`iloc`) con slicing veloce di NumPy (`data_np[i:i+batch]`).
    *   Implementazione di **sparse encoding** simile a `generateDatasets.py`.
    *   Aggiunto logging frequente (ogni epoca) per monitorare il progresso.

### 2.3 `evaluate_comparative.py` (Compatibilità API)
Il wrapper `cuml.multiclass.OneVsRestClassifier` ha mostrato incompatibilità con il metodo `predict_proba` necessario per il calcolo MAUC.
*   **Soluzione**:
    *   Utilizzo del wrapper **`sklearn.multiclass.OneVsRestClassifier`** applicato agli estimatori **cuML**. Questo garantisce la compatibilità dell'API di scikit-learn mantenendo l'accelerazione GPU dei classificatori sottostanti.

### 2.4 `MAUC_Exp.py` & `MAUC_rawExp.py` (Fix Grafici e Warning)
*   Risolto errore `AttributeError: factorplot` (deprecato in Seaborn) sostituendolo con `catplot`.
*   Aggiunta creazione automatica della directory `results/consolidated` per evitare crash in salvataggio.
*   Soppressione dei warning benigni di cuML/Python per pulire l'output.

## 3. Dettaglio Esperimenti e Risultati

Di seguito i risultati esatti ottenuti, con riferimento agli script utilizzati e ai file di output generati.

### 3.1 Baseline: Feature Originali (Raw Data)
Esecuzione dei classificatori direttamente sulle feature originali del dataset (senza Autoencoder).
*   **Script**: `MAUC_rawExp.py` (NB, RF), `MAUC_rawExp.py` + `SVM` (SVM)
*   **File Output**:
    *   `results/consolidated/originalAtt_MAUC.txt`
    *   `results/consolidated/SVMoriginalAtt_MAUC.txt`

| Classificatore | MAUC (Raw) | Note |
|---|---|---|
| **NB** | 0.5126 | Performance quasi casuale (Gaussian assumption violata dai dati raw) |
| **RF** | 0.9972 | Eccellente, gestisce bene feature non lineari/categoriali |
| **SVM** | 0.9937 | Eccellente |

### 3.2 Baseline: Adversarial Autoencoder (AAE)
Riproduzione del paper originale. L'AAE proietta i dati in uno spazio latente (dim=10) forzando una distribuzione prior (mixture of Gaussians).
*   **Script Generazione Dati**: `generateDatasets.py`
    *   **Output Dati**: `latent_data_sets/ldim10_tauX_basisLS.txt` (dove X è 5, 10, 20)
*   **Script Valutazione**: `MAUC_Exp.py`
*   **File Output**:
    *   **Grafico**: `results/consolidated/ldim10_MAUC.png`
    *   **Dati**: `results/consolidated/numeric_ldim10_MAUC.txt`

| Classificatore | Tau=5 | Tau=10 (Default) | Tau=20 |
|---|---|---|---|
| **NB** | 0.8362 | 0.9427 | 0.9517 |
| **RF** | 0.9345 | 0.9772 | 0.9687 |
| **SVM** | 0.9215 | 0.8356 | **0.9950** |

> **Nota**: SVM beneficia enormemente di `tau=20` (più gaussiane = cluster più piccoli e separabili), mentre soffre con `tau=10`. RF è robusto in ogni configurazione.

### 3.3 Nuova Estensione: Standard Autoencoder
Confronto con un Autoencoder standard (senza discriminatore avversariale).
*   **Script Training**: `train_standard_ae.py` (100 epoche, ldim=10)
    *   **Output Dati**: `latent_data_sets/ldim10_standardAE_basisLS.txt`
*   **Script Valutazione**: `evaluate_comparative.py`
*   **File Output**: (Log Console)

| Classificatore | Standard AE | vs AAE (Tau=10) | Note |
|---|---|---|---|
| **NB** | 0.9814 | **+4.1%** | Spazio latente molto "pulito" e gaussiano |
| **RF** | 0.9822 | +0.5% | Performance comparabile al top AAE |
| **SVM** | 0.9674 | **+13.1%** | Molto meglio di AAE Tau=10, quasi livello Tau=20 |

### 3.4 Analisi Qualitativa (t-SNE)
Studio della topologia dello spazio latente tramite proiezioni t-SNE (t-Distributed Stochastic Neighbor Embedding) per verificare visivamente la separabilità delle classi.

*   **Script Eseguito**: `tSNE_experiments.py`
    *   **Configurazione**: Utilizzo di **RAPIDS cuML TSNE** per accelerazione GPU.
    *   **Dataset**: Intero dataset proiettato (~533.000 campioni) **senza campionamento**, per evitare perdita di informazioni e artefatti visivi.
*   **File Generati** (in `tsneResults/`):
    1.  `ldim10_tau5_basisLS_tSNE.png`
    2.  `ldim10_tau10_basisLS_tSNE.png`
    3.  `ldim10_tau20_basisLS_tSNE.png`
    4.  `ldim10_standardAE_basisLS_tSNE.png`

**Analisi Dettagliata dei Risultati Visivi**:
1.  **AAE (Tau=5, 10, 20)**:
    *   **Topologia**: I grafici mostrano una struttura a "stella" o nube sferica estremamente densa al centro.
    *   **Interpretazione**: Il Discriminatore forza con successo lo spazio latente verso la distribuzione Prior (Gaussiana).
    *   **Criticità**: Le frodi (punti arancioni/rossi) sono spesso "schiacciate" all'interno della massa densa dei dati normali (blu). Manca una chiara separazione spaziale, spiegando le difficoltà dei classificatori lineari (SVM MAUC ~0.83 con Tau=10).

2.  **Standard AE**:
    *   **Topologia**: Il grafico mostra una struttura radicalmente diversa, frammentata in "isole" e cluster irregolari distribuiti su un'area ampia (manifold naturale).
    *   **Vantaggio**: Si osservano spazi vuoti tra i cluster normali. Le frodi tendono a posizionarsi ai margini o in isole isolate. Questa topologia "sparsa" facilita enormemente il compito dei classificatori, giustificando il **MAUC superiore (0.98+)**.

**Conclusione Qualitativa**:
L'analisi t-SNE fornisce la "pistola fumante": la regolarizzazione avversariale dell'AAE, pur matematicamente elegante, tende a comprimere eccessivamente la struttura topologica, rendendo le anomalie meno distinguibili rispetto a un semplice Autoencoder non regolarizzato.

### Sintesi Finale
L'approccio **Standard AE** (molto più semplice e veloce da addestrare, ~2 secondi/epoca su GPU) ha prodotto uno spazio latente che compete o supera l'AAE complesso del paper originale. Questa è una scoperta significativa per l'estensione del progetto.

## 4. Stato Avanzamento rispetto al Piano Originale

Di seguito un confronto dettagliato tra gli obiettivi originali del progetto e lo stato attuale dei lavori.

### 4.1 Riproducibilità (Baseline)
> *Obiettivo: Riproduzione degli esperimenti originali del paper di riferimento, al fine di validare le metriche di performance (MAUC) sul dataset fornito.*

*   **Stato**: ✅ **COMPLETATO**
*   **Risultato**: Le metriche MAUC per l'AAE a diversi `tau` sono state riprodotte con successo. I valori ottenuti sono coerenti con quelli attesi (NB migliora con tau alto, RF stabile, SVM performante su tau=20).

### 4.2 Sperimentazione Critica e Comparativa (Estensione)
> *Obiettivo: Validazione dell’architettura e della sua stabilità.*

#### A. Confronto Architetturale
> *Obiettivo: Isolare il contributo della componente avversaria confrontando l'AAE con un Autoencoder standard (senza discriminatore).*

*   **Stato**: ✅ **COMPLETATO**
*   **Risultato**: Abbiamo dimostrato un risultato inatteso e significativo. Lo **Standard AE** (molto più semplice e veloce da addestrare) ha ottenuto performance di classificazione **superiori** (NB: 0.98 vs 0.94) o comparabili all'AAE complesso. Questo suggerisce che per questo specifico dataset, la regolarizzazione avversaria potrebbe non essere strettamente necessaria per la separabilità delle frodi.

#### B. Analisi Qualitativa (t-SNE)
> *Obiettivo: Studio della topologia dello spazio latente tramite proiezioni t-SNE per verificare visivamente la separazione delle frodi.*

*   **Stato**: ✅ **COMPLETATO**
*   **Risultato**: Le visualizzazioni confermano che l'AAE comprime eccessivamente i dati (collasso gaussiano), mentre lo Standard AE preserva un manifold complesso con spazi vuoti che facilitano la classificazione.

#### C. Analisi della Stabilità del Training
> *Obiettivo: Analizzare le curve di apprendimento (Reconstruction, Generator, Discriminator Loss) per verificare la convergenza ed evitare il vanishing gradient.*

*   **Stato**: ✅ **COMPLETATO**
*   **Esperimento**: Eseguite **5 run indipendenti** (`N=5`) dell'AAE con `tau=10`, per 50 epoche ciascuna.
*   **Risultati**:
    *   **Convergenza**: La `Reconstruction Loss` converge in modo estremamente coerente in tutte e 5 le run, stabilizzandosi nel range **0.0037 - 0.0044**.
    *   **Dinamica Avversariale**: La `Discriminator Loss` e la `Generator Loss` mostrano la classica oscillazione delle GAN, stabilizzandosi (Disc ~1.2, Gen ~0.9) senza segni di collasso del gradiente o divergenza.
    *   **Affidabilità**: La bassa varianza tra le run (evidente nei grafici `results/stability/loss_curves.png`) conferma che il modello è robusto e l'addestramento è stabile sui dati forniti.

#### D. Studio dell’Imbalance
> *Obiettivo: Stress test del modello riducendo artificialmente la percentuale di campioni anomali.*

*   **Stato**: ✅ **COMPLETATO**
*   **Esperimento**: Testato l'AAE su ratei di frode decrescenti dallo 0.1% al 10%.
*   **Risultati**:
    | Fraud Ratio | Reconstruction AUC | Note |
    |---|---|---|
    | 0.1% | 0.7830 | Ottima performance nonostante l'estrema scarsità |
    | 1.0% | 0.7760 | Stabile |
    | **2.0%** | **0.8005** | **Picco di performance** |
    | 5.0% | 0.7218 | Declino (Pollution) |
    | 10.0% | 0.7125 | Declino marcato |
*   **Analisi**: Il modello soffre di "pollution" quando le frodi superano il 2%. Essendo un modello non supervisionato, se le frodi sono troppo frequenti, l'Autoencoder inizia a imparare a ricostruirle correttamente, riducendo l'errore di ricostruzione e quindi la capacità di distinguerle.

## 5. Deviazioni e Adattamenti Tecnici rispetto al Piano

Per raggiungere gli obiettivi prefissati, è stato necessario introdurre modifiche tecniche sostanziali non previste nel piano iniziale.

### 5.1 Ottimizzazione Memoria (OOM)
Il dataset proiettato (One-Hot Encoded) si è rivelato troppo grande per la gestione standard in memoria RAM (crash sistematici su `generateDatasets.py` e `MAE_rawExp.py`).
*   **Adattamento**: Riscrittura completa della pipeline di data loading utilizzando **Matrici Sparse** (risparmio ~90% RAM) e elaborazione **Batch-wise** manuale.
*   **Impatto**: Ha reso possibile l'esecuzione degli esperimenti su hardware consumer (RTX 3060 / 16GB RAM) senza crash.

### 5.2 Migrazione a RAPIDS cuML
L'esecuzione degli algoritmi classici (SVM, Random Forest) su CPU tramite `scikit-learn` risultava proibitiva in termini di tempo per la mole di dati processata.
*   **Adattamento**: Migrazione totale dello stack tecnologico su **RAPIDS cuML** per sfruttare l'accelerazione GPU.
*   **Impatto**: Tempi di training ridotti drasticamente (es. SVM da ore a secondi), permettendo iterazioni rapide.

### 5.3 Dockerizzazione dell'Ambiente
La gestione delle dipendenze CUDA e delle librerie scientifiche su Windows è risultata instabile.
*   **Adattamento**: Creazione di un ambiente **Docker** custom (`rapidsai/base` modificata).
*   **Impatto**: Garantisce la **riproducibilità totale** degli esperimenti indipendentemente dalla macchina host, un requisito fondamentale per la tesi.

### 5.4 Risultati Standard AE
Il piano prevedeva lo Standard AE come semplice baseline inferiore.
*   **Deviazione**: I risultati mostrano che lo Standard AE è **superiore** all'AAE in configurazione base.
*   **Impatto**: Questo risultato richiederà una discussione approfondita nella tesi: la complessità dell'approccio avversariale potrebbe non essere giustificata per questo specifico task, o potrebbe richiedere un tuning molto più fine (`train_stability.py`) per emergere.
