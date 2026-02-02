# Relazione Modifica Ambiente: Migrazione a RAPIDS cuML

## Riepilogo
È stata effettuata la migrazione dei modelli di classificazione da `scikit-learn` a `RAPIDS cuML` per abilitare l'accelerazione GPU. I file principali del progetto sono stati aggiornati per utilizzare le implementazioni ottimizzate CUDA di SVM, Random Forest e t-SNE.

## Modifiche Apportate

### 1. Verifica Ambiente
- Creato script `verify_cuml.py` nella root del progetto.
- **Funzione**: Verifica che le librerie `cuml` e `cudf` siano importabili ed esegue un test rapido di training SVM.
- **Uso**: `python verify_cuml.py`

### 2. Comparative Evaluation (`project_extension/evaluate_comparative.py`)
- Sostituiti import:
    - `sklearn.svm.SVC` -> `cuml.svm.SVC`
    - `sklearn.ensemble.RandomForestClassifier` -> `cuml.ensemble.RandomForestClassifier`
    - `sklearn.multiclass.OneVsRestClassifier` -> `cuml.multiclass.OneVsRestClassifier`
    - `sklearn.naive_bayes.GaussianNB` -> `cuml.naive_bayes.GaussianNB` (se non disponibile in versioni vecchie, fallback su sklearn, ma qui abbiamo usato cuml assumendo disponibilità nella 23.10+)

### 3. t-SNE (`tSNE_experiments.py`)
- Sostituito `sklearn.manifold.TSNE` con `cuml.manifold.TSNE`.
- Atteso miglioramento significativo nei tempi di calcolo per la riduzione dimensionale e visualizzazione.

### 4. Esperimenti Legacy (`MAUC_Exp.py` e `MAUC_rawExp.py`)
- Aggiornati i classificatori SVM e Random Forest per usare le versioni `cuml`.
- Mantenuta la logica `OneVsRestClassifier`.

## Istruzioni per l'Esecuzione (Metodo Consigliato: Docker)
È stato predisposto un ambiente Docker containerizzato per garantire la massima compatibilità e facilità d'uso.

### 1. Setup Iniziale (Una Tantum)
Apri PowerShell nella cartella del progetto ed esegui lo script di costruzione:
```powershell
.\build_image.ps1
```
Questo scaricherà l'immagine RAPIDS e installerà le dipendenze extra (`matplotlib`, `seaborn`) definendo l'immagine `fraud-detection-rapids`.

### 2. Avvio Ambiente
Ogni volta che vuoi lavorare:
```powershell
.\run_container.ps1
```
Ti ritroverai in una shell Linux (`root@...:/workspace#`) con GPU abilitata.

### 3. Esecuzione Esperimenti
Dentro il container:
```bash
# Verifica rapida
python verify_cuml.py

# Esegui esperimento comparativo
python project_extension/evaluate_comparative.py
```

---

## Istruzioni Alternative (Conda Manuale)
Se preferisci non usare Docker, puoi creare l'ambiente manualmente:
```bash
conda create -n rapids -c rapidsai -c conda-forge -c nvidia cuml=23.10 python=3.9 cuda-version=11.8 scikit-learn seaborn matplotlib
conda activate rapids
```
*Nota: Questa procedura può richiedere molto tempo e dipende dalla configurazione dei driver locali.*
