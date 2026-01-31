# gemini-conversation
Exported on: 13/01/2026, 10:27:27

---

**You:**
da dove iniziamo?

**Gemini:**
Ottimo piano di lavoro. È strutturato, scientifico e incrementale.

Per iniziare, non toccheremo il codice originale (`gabrielefuoco/mlllllll/...`) ma costruiremo un "wrapper" che ci permetta di eseguire la **Fase 2 (Riproducibilità)**.

Ho notato un'incongruenza critica nel codice originale fornito (file `generateDatasets.py`): sebbene il `README` dica di eseguire prima il training (`validation_v0.py`), lo script di generazione (`generateDatasets.py`) scarica dei modelli **pre-addestrati** da un URL hardcoded su GitHub, ignorando di fatto qualsiasi training locale tu possa fare.
Per la Fase 2 (Riproducibilità pura), eseguiremo il codice "as-is" (usando i loro modelli pre-addestrati) per ottenere la baseline esatta del paper. Nella Fase 3 (Sperimentazione), dovremo invece modificare questo comportamento per testare i nostri modelli.

Ecco le istruzioni atomiche per iniziare.

### Passo 1: Setup della Struttura

Dobbiamo creare un ambiente di lavoro pulito che separi il nostro codice da quello originale.

1. Nella **root** del repository, crea una nuova cartella chiamata `project_extension`.
2. All'interno di `project_extension`, crea un file vuoto chiamato `__init__.py` (per rendere la cartella un package Python).

### Passo 2: Creazione dello Script di Baseline (Riproducibilità)

Creeremo uno script che orchestra l'esecuzione dei file originali, assicurandosi che le cartelle di output esistano (il codice originale a volte fallisce se le cartelle non ci sono).

Crea il file `project_extension/reproduce_baseline.py` e incolla il seguente codice:

```
import os
import subprocess
import sys

def ensure_directories():
    """Crea le directory necessarie se non esistono, per evitare crash degli script originali."""
    dirs = ["latent_data_sets", "results", "models", "data"]
    for d in dirs:
        if not os.path.exists(d):
            print(f"[SETUP] Creazione directory: {d}")
            os.makedirs(d)

def run_script(script_name):
    """Esegue uno script python come sottoprocesso per garantire isolamento."""
    print(f"\n{'='*20} ESECUZIONE {script_name} {'='*20}")
    try:
        # Esegue lo script usando l'interprete corrente
        result = subprocess.run([sys.executable, script_name], check=True)
        print(f"[SUCCESS] {script_name} completato.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Errore durante l'esecuzione di {script_name}: {e}")
        sys.exit(1)

def main():
    print("--- INIZIO FASE 2: RIPRODUCIBILITÀ (BASELINE) ---")
    
    # 1. Preparazione ambiente
    ensure_directories()
    
    # NOTA: Saltiamo validation_v0.py per la riproducibilità "strict" 
    # perché generateDatasets.py originale scarica i pesi da internet 
    # e ignora il training locale.
    
    # 2. Generazione dei dataset nello spazio latente
    # Questo script scarica i modelli pre-trained e crea i file in /latent_data_sets
    if os.path.exists("generateDatasets.py"):
        run_script("generateDatasets.py")
    else:
        print("[ERROR] generateDatasets.py non trovato nella root.")
        return

    # 3. Esperimenti su spazio latente (MAUC)
    if os.path.exists("MAUC_Exp.py"):
        run_script("MAUC_Exp.py")
    else:
        print("[ERROR] MAUC_Exp.py non trovato.")

    # 4. Esperimenti su feature originali (Raw)
    # Serve come confronto baseline
    if os.path.exists("MAUC_rawExp.py"):
        run_script("MAUC_rawExp.py")
    else:
        print("[WARNING] MAUC_rawExp.py non trovato (potrebbe non essere stato caricato).")

    print("\n--- BASELINE COMPLETATA ---")
    print("Controlla la cartella 'results/consolidated' per i grafici e i file di testo.")

if __name__ == "__main__":
    main()

```

### Passo 3: Esecuzione e Verifica

Ora eseguiamo la baseline per assicurarci che tutto funzioni e per ottenere i dati di confronto per i passaggi futuri.

1. Apri il terminale nella **root** del repository.
2. Esegui il comando:

```
python project_extension/reproduce_baseline.py

```
3. **Analisi dell'output**:

* Lo script dovrebbe scaricare il dataset e i modelli.
* Dovrebbe popolare la cartella `latent_data_sets`.
* Dovrebbe stampare a video i valori di MAUC per i vari classificatori (SVM, RF, NB).

**Attenzione:** Se ottieni errori relativi a librerie mancanti, installale (le principali sembrano essere `pandas`, `numpy`, `sklearn`, `matplotlib`, `seaborn`, `torch`).

Fammi sapere quando hai eseguito questo step e hai i risultati (o se incontri errori), così procederemo con la **Fase 3: Estensione (Confronto Architetturale)** creando il nostro training loop personalizzato.

---

**You:**
prosegui

**Gemini:**
Perfetto. Passiamo alla **Fase 3: Sperimentazione critica e comparativa**.

Il primo obiettivo di questa fase è isolare il contributo della componente "avversaria". Per farlo, dobbiamo addestrare un **Autoencoder Standard (AE)** (senza discriminatore) sugli stessi dati e con la stessa architettura di rete (Encoder/Decoder) usata nel paper.

Confronteremo poi le performance (MAUC) di questo AE standard contro l'AAE originale (la baseline che hai appena riprodotto). Se l'AAE è superiore, dimostreremo che la regolarizzazione avversaria è utile; se sono simili, la complessità dell'AAE potrebbe non essere giustificata.

Ecco le istruzioni per creare il training loop dell'Autoencoder Standard.

### Passo 1: Creazione script di training Standard AE

Crea il file `project_extension/train_standard_ae.py`.
Questo script:

1. Importa i moduli `Encoder` e `Decoder` originali.
2. Replica *esattamente* il preprocessing dei dati usato nel paper (per un confronto equo).
3. Addestra la rete minimizzando solo l'errore di ricostruzione (MSELoss), ignorando la parte GAN.
4. Genera il dataset latente compatibile con gli script di valutazione esistenti.

Incolla questo codice nel file:

```
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

# Aggiungiamo la root al path per importare i moduli originali
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Encoder import Encoder
from Decoder import Decoder

def train_standard_ae():
    print("--- TRAINING STANDARD AUTOENCODER (NO ADVERSARIAL) ---")
    
    # --- 1. CONFIGURAZIONE ---
    SEED_VALUE = 1234
    LATENT_DIM = 10
    BATCH_SIZE = 128
    EPOCHS = 100  # Riduciamo rispetto ai 5000 del paper per rapidità di test, aumenta se hai GPU
    LR = 0.001
    DATA_PATH = 'data/fraud_dataset_v2.csv'
    OUTPUT_FILE = f"latent_data_sets/ldim{LATENT_DIM}_standardAE_basisLS.txt"

    # Setup deterministico
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)

    # --- 2. CARICAMENTO E PREPROCESSING DATI (Identico all'originale) ---
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Dataset non trovato in {DATA_PATH}. Esegui prima la baseline.")
        return

    print("Caricamento dati...")
    ori_dataset = pd.read_csv(DATA_PATH)
    labels = ori_dataset.pop('label') # Rimuoviamo label per il training unsupervised

    # Preprocessing Categoriale
    categorical_attr = ['KTOSL', 'PRCTR', 'BSCHL', 'HKONT', 'BUKRS', 'WAERS']
    dataset_categ = pd.get_dummies(ori_dataset[categorical_attr])

    # Preprocessing Numerico
    numeric_attr_names = ['DMBTR', 'WRBTR']
    numeric_attr = ori_dataset[numeric_attr_names] + 1e-4
    numeric_attr = numeric_attr.apply(np.log)
    dataset_numeric = (numeric_attr - numeric_attr.min()) / (numeric_attr.max() - numeric_attr.min())

    # Merge
    dataset_processed = pd.concat([dataset_categ, dataset_numeric], axis=1)
    input_dim = dataset_processed.shape[1]
    
    # Conversione in tensori
    tensor_data = torch.FloatTensor(dataset_processed.values)
    dataloader = DataLoader(tensor_data, batch_size=BATCH_SIZE, shuffle=True)

    # --- 3. DEFINIZIONE MODELLO ---
    # Usiamo le stesse classi originali per garantire confronto architetturale equo
    encoder = Encoder(input_size=input_dim, hidden_size=[256, 64, 16, 4, LATENT_DIM])
    decoder = Decoder(output_size=input_dim, hidden_size=[LATENT_DIM, 4, 16, 64, 256])

    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        tensor_data = tensor_data.cuda() # Per la fase di valutazione finale

    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)

    # --- 4. TRAINING LOOP ---
    print(f"Inizio training per {EPOCHS} epoche...")
    encoder.train()
    decoder.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            if torch.cuda.is_available():
                batch = batch.cuda()
            
            # Forward
            z = encoder(batch)
            reconstruction = decoder(z)
            
            loss = criterion(reconstruction, batch)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(dataloader):.6f}")

    # --- 5. GENERAZIONE SPAZIO LATENTE ---
    print("Generazione rappresentazione latente...")
    encoder.eval()
    with torch.no_grad():
        # Passiamo tutto il dataset (senza shuffle) per mantenere allineamento con le label originali
        # Usiamo tensor_data che corrisponde all'ordine originale del DataFrame processato
        full_z = encoder(tensor_data)
        full_z_np = full_z.cpu().numpy()

    # Mappatura label per output (0: regular, 1: global, 2: local)
    label_map = {'regular': 0, 'global': 1, 'local': 2}
    numeric_labels = labels.map(label_map).values

    # --- 6. SALVATAGGIO ---
    print(f"Salvataggio dataset latente in {OUTPUT_FILE}...")
    
    if not os.path.exists('latent_data_sets'):
        os.makedirs('latent_data_sets')

    with open(OUTPUT_FILE, "w") as f:
        for i in range(len(full_z_np)):
            # Formato: feature1\tfeature2\t...\tlabel
            features_str = "\t".join([str(x) for x in full_z_np[i]])
            line = f"{features_str}\t{numeric_labels[i]}\n"
            f.write(line)

    print("Fatto.")

if __name__ == "__main__":
    train_standard_ae()

```

### Passo 2: Creazione script di valutazione comparativa

Ora ci serve uno script che prenda questo nuovo dataset latente e calcoli le metriche MAUC, esattamente come faceva `MAUC_Exp.py` per l'AAE originale.
Crea il file `project_extension/evaluate_comparative.py`:

```
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MultiLabelBinarizer

# Importiamo MAUC originale
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MAUC import MAUC

def load_latent_data(filepath):
    X = []
    Y = []
    Y_normal = []
    
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split('\t')
            # Le ultime cifre sono la label
            features = [float(z) for z in parts[:-1]]
            label = int(parts[-1])
            
            X.append(features)
            Y.append([label]) # List of list per compatibilità
            Y_normal.append(label)
            
    return np.array(X), Y, Y_normal

def evaluate_latent_space(latent_file):
    print(f"\nValutazione MAUC per: {os.path.basename(latent_file)}")
    
    X, Y, Y_normal = load_latent_data(latent_file)
    
    # Configurazione identica al paper
    K = 10
    SEED = 1234
    skf = StratifiedKFold(n_splits=K, random_state=SEED, shuffle=True)
    
    classifiers = {
        'NB': OneVsRestClassifier(GaussianNB()),
        'RF': OneVsRestClassifier(RandomForestClassifier(n_estimators=10, random_state=SEED)),
        'SVM': OneVsRestClassifier(SVC(kernel='linear', random_state=SEED, probability=True))
    }
    
    results = {}
    
    for clf_name, clf in classifiers.items():
        print(f"Running {clf_name}...")
        
        Y_test_all_normal = []
        Y_score_all = []
        
        # Cross Validation Loop
        skf_idxs = skf.split(X, Y_normal) # Split basato su Y_normal appiattito
        
        for train_idx, test_idx in skf_idxs:
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train_raw = [Y[i] for i in train_idx]
            Y_test_raw = [Y[i] for i in test_idx]
            
            # Binarizzazione label per OneVsRest
            mlb = MultiLabelBinarizer()
            Y_train = mlb.fit_transform(Y_train_raw)
            # Y_test = mlb.transform(Y_test_raw) # Non serve per il calcolo finale MAUC qui implementato
            
            # Label "flat" per MAUC calculation finale
            Y_test_normal_fold = [Y_normal[i] for i in test_idx]
            
            clf.fit(X_train, Y_train)
            y_score = clf.predict_proba(X_test)
            
            Y_test_all_normal.extend(Y_test_normal_fold)
            if len(Y_score_all) == 0:
                Y_score_all = y_score
            else:
                Y_score_all = np.concatenate((Y_score_all, y_score))
        
        # Calcolo MAUC
        # MAUC prende (zip(labels, scores), num_classes)
        mapped = list(zip(Y_test_all_normal, Y_score_all))
        mauc_score = MAUC(mapped, 3) # 3 classi: Regular, Global, Local
        results[clf_name] = mauc_score
        print(f"-> {clf_name} MAUC: {mauc_score:.4f}")
        
    return results

if __name__ == "__main__":
    # File prodotto dallo step precedente
    target_file = "latent_data_sets/ldim10_standardAE_basisLS.txt"
    if os.path.exists(target_file):
        evaluate_latent_space(target_file)
    else:
        print(f"File {target_file} non trovato. Esegui prima train_standard_ae.py")

```

### Passo 3: Esecuzione

Esegui i due script in sequenza dal terminale nella root del progetto:

1. **Addestra l'Autoencoder Standard:**

```
python project_extension/train_standard_ae.py

```
2. **Valuta le performance:**

```
python project_extension/evaluate_comparative.py

```

Quando hai i risultati a video (i valori MAUC per NB, RF, SVM), fammelo sapere. Li confronteremo con quelli della baseline (che dovresti aver visto nel passo precedente, solitamente salvati in `results/consolidated`) per trarre le conclusioni sulla validità dell'approccio Adversarial.

---

**You:**
continua

**Gemini:**
Ottimo. Procediamo con il punto successivo del piano: **Analisi della stabilità del training**.

Nel paper originale, si utilizzano modelli pre-addestrati scaricati da GitHub, quindi non abbiamo visibilità su *come* siano arrivati a convergenza. Essendo una GAN (Generative Adversarial Network), l'equilibrio tra Generatore (Encoder) e Discriminatore è critico: se il discriminatore diventa troppo bravo troppo in fretta, il generatore smette di imparare (vanishing gradients).

Creeremo uno script che addestra l'AAE da zero (in locale), registrando le loss passo dopo passo per generare i grafici di stabilità.

### Passo 1: Script per Analisi Stabilità AAE

Crea il file `project_extension/train_aae_stability.py`.
Questo script:

1. Addestra l'AAE (Encoder + Decoder + Discriminator).
2. Usa `latent_dim=10` (come la baseline migliore).
3. Salva un log CSV delle loss e genera un grafico `loss_curves.png`.

Incolla il seguente codice:

```
import sys
import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setup path per importare moduli originali
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Encoder import Encoder
from Decoder import Decoder
from Discriminator import Discriminator

def train_aae_stability():
    print("--- TRAINING AAE (STABILITY ANALYSIS) ---")

    # --- CONFIGURAZIONE ---
    SEED_VALUE = 1234
    LATENT_DIM = 10
    TAU = 20  # Numero di cluster Gaussiani (come nel paper)
    BATCH_SIZE = 128
    EPOCHS = 100  # Sufficienti per vedere la convergenza/stabilità
    LR_ENC = 1e-3
    LR_DEC = 1e-3
    LR_DISC = 1e-5 # Learning rate basso per il discriminatore per evitare overpower
    DATA_PATH = 'data/fraud_dataset_v2.csv'
    RESULTS_DIR = 'project_extension/results_stability'
    
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Determinismo
    random.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED_VALUE)

    # --- PREPROCESSING (Identico) ---
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Dataset non trovato in {DATA_PATH}")
        return

    print("Caricamento e preprocessing dati...")
    ori_dataset = pd.read_csv(DATA_PATH)
    ori_dataset.pop('label')

    categorical_attr = ['KTOSL', 'PRCTR', 'BSCHL', 'HKONT', 'BUKRS', 'WAERS']
    dataset_categ = pd.get_dummies(ori_dataset[categorical_attr])

    numeric_attr_names = ['DMBTR', 'WRBTR']
    numeric_attr = ori_dataset[numeric_attr_names] + 1e-4
    numeric_attr = numeric_attr.apply(np.log)
    dataset_numeric = (numeric_attr - numeric_attr.min()) / (numeric_attr.max() - numeric_attr.min())

    dataset_processed = pd.concat([dataset_categ, dataset_numeric], axis=1)
    input_dim = dataset_processed.shape[1]
    
    torch_dataset = torch.FloatTensor(dataset_processed.values)
    dataloader = DataLoader(torch_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- SETUP MODELLI ---
    encoder = Encoder(input_size=input_dim, hidden_size=[256, 64, 16, 4, LATENT_DIM])
    decoder = Decoder(output_size=input_dim, hidden_size=[LATENT_DIM, 4, 16, 64, 256])
    discriminator = Discriminator(input_size=LATENT_DIM, hidden_size=[256, 16, 4, 2], output_size=1)

    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        discriminator = discriminator.cuda()

    # Ottimizzatori
    optim_enc = optim.Adam(encoder.parameters(), lr=LR_ENC)
    optim_dec = optim.Adam(decoder.parameters(), lr=LR_DEC)
    optim_disc = optim.Adam(discriminator.parameters(), lr=LR_DISC)

    # Criteri Loss
    criterion_bce = nn.BCELoss()
    criterion_mse = nn.MSELoss()

    # --- PREPARAZIONE PRIOR (Mixture of Gaussians) ---
    # Generiamo campioni dalla distribuzione target una volta sola (o on-the-fly)
    # Qui replicando la logica del paper per creare il "pool" di campioni target
    print("Generazione Prior (Gaussian Mixture)...")
    radius = 0.8
    sigma = 0.01
    samples_per_gaussian = 10000
    centroids = []
    for i in range(LATENT_DIM):
        if i % 2 == 0:
            centroids.append((radius * np.cos(np.linspace(0, 2 * np.pi, TAU, endpoint=False)) + 1) / 2)
        else:
            centroids.append((radius * np.sin(np.linspace(0, 2 * np.pi, TAU, endpoint=False)) + 1) / 2)
    mu_gauss = np.vstack(centroids).T
    
    z_real_samples_pool = []
    for mu in mu_gauss:
        samples = np.random.normal(mu, sigma, size=(samples_per_gaussian, LATENT_DIM))
        z_real_samples_pool.append(samples)
    z_real_samples_pool = np.vstack(z_real_samples_pool)
    # Shuffle
    np.random.shuffle(z_real_samples_pool)
    
    # --- TRAINING LOOP ---
    print(f"Inizio training AAE per {EPOCHS} epoche...")
    history = {'epoch': [], 'recon_loss': [], 'disc_loss': [], 'gen_loss': []}

    for epoch in range(EPOCHS):
        epoch_recon = 0.0
        epoch_disc = 0.0
        epoch_gen = 0.0
        batches = 0

        encoder.train()
        decoder.train()
        discriminator.train()

        for batch in dataloader:
            batches += 1
            if torch.cuda.is_available():
                batch = batch.cuda()

            # 1. RECONSTRUCTION PHASE
            encoder.zero_grad()
            decoder.zero_grad()
            discriminator.zero_grad()

            z_sample = encoder(batch)
            recon_batch = decoder(z_sample)

            # Split numeric/categorical loss (simplified logic for stability check)
            # Nota: il codice originale splitta le feature. Qui semplifichiamo usando MSE su tutto 
            # o se vogliamo essere precisi usiamo lo split. Usiamo lo split per rigore.
            n_cat = dataset_categ.shape[1]
            
            rec_cat = recon_batch[:, :n_cat]
            rec_num = recon_batch[:, n_cat:]
            true_cat = batch[:, :n_cat]
            true_num = batch[:, n_cat:]

            loss_recon = criterion_bce(rec_cat, true_cat) + criterion_mse(rec_num, true_num)
            loss_recon.backward()
            
            optim_enc.step()
            optim_dec.step()
            epoch_recon += loss_recon.item()

            # 2. REGULARIZATION PHASE - DISCRIMINATOR
            encoder.zero_grad()
            discriminator.zero_grad()

            # Real samples
            idx = np.random.randint(0, z_real_samples_pool.shape[0], batch.size(0))
            z_real = torch.FloatTensor(z_real_samples_pool[idx])
            if torch.cuda.is_available(): z_real = z_real.cuda()
            
            d_real = discriminator(z_real)
            target_real = torch.ones_like(d_real)
            loss_d_real = criterion_bce(d_real, target_real)

            # Fake samples (from encoder)
            z_fake = encoder(batch)
            d_fake = discriminator(z_fake.detach()) # Detach per non aggiornare encoder qui
            target_fake = torch.zeros_like(d_fake)
            loss_d_fake = criterion_bce(d_fake, target_fake)

            loss_disc = loss_d_real + loss_d_fake
            loss_disc.backward()
            optim_disc.step()
            epoch_disc += loss_disc.item()

            # 3. REGULARIZATION PHASE - GENERATOR
            encoder.zero_grad()
            discriminator.zero_grad()

            # Vogliamo che il discriminatore sbagli (classifichi fake come 1)
            z_fake_gen = encoder(batch)
            d_fake_gen = discriminator(z_fake_gen)
            target_gen = torch.ones_like(d_fake_gen)

            loss_gen = criterion_bce(d_fake_gen, target_gen)
            loss_gen.backward()
            optim_enc.step()
            epoch_gen += loss_gen.item()

        # Log metrics
        history['epoch'].append(epoch+1)
        history['recon_loss'].append(epoch_recon / batches)
        history['disc_loss'].append(epoch_disc / batches)
        history['gen_loss'].append(epoch_gen / batches)

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Recon: {history['recon_loss'][-1]:.4f} | Disc: {history['disc_loss'][-1]:.4f} | Gen: {history['gen_loss'][-1]:.4f}")

    # --- SALVATAGGIO OUTPUT ---
    # CSV
    df_hist = pd.DataFrame(history)
    csv_path = os.path.join(RESULTS_DIR, 'aae_training_log.csv')
    df_hist.to_csv(csv_path, index=False)
    print(f"Log salvato in: {csv_path}")

    # PLOT
    plt.figure(figsize=(12, 5))
    
    # Plot Reconstruction
    plt.subplot(1, 2, 1)
    plt.plot(df_hist['epoch'], df_hist['recon_loss'], label='Reconstruction Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Reconstruction Convergence')
    plt.legend()
    plt.grid(True)

    # Plot GAN Stability
    plt.subplot(1, 2, 2)
    plt.plot(df_hist['epoch'], df_hist['disc_loss'], label='Discriminator Loss', color='red', alpha=0.7)
    plt.plot(df_hist['epoch'], df_hist['gen_loss'], label='Generator Loss', color='green', alpha=0.7)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Adversarial Stability (Min-Max Game)')
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(RESULTS_DIR, 'loss_curves.png')
    plt.savefig(plot_path)
    print(f"Grafico salvato in: {plot_path}")
    plt.close()

if __name__ == "__main__":
    train_aae_stability()

```

### Passo 2: Esecuzione e Interpretazione

Esegui lo script:

```
python project_extension/train_aae_stability.py

```

**Cosa cercare nel grafico (`loss_curves.png`):**

1. **Reconstruction Loss (Blu):** Deve scendere rapidamente e stabilizzarsi vicino a zero. Se non scende, il modello non sta imparando a ricostruire i dati (underfitting).
2. **Adversarial Stability (Rosso/Verde):**

* **Scenario Ideale:** Le due curve oscillano intorno a un valore costante (es. 1.38 per la somma discriminator loss, che corrisponde a ln(4), oppure oscillano in opposizione ma senza divergere). Significa che il generatore inganna il discriminatore e viceversa (Equilibrio di Nash).
* **Failure Mode (Discriminator Win):** La Loss del Discriminatore (Rossa) va a 0 e quella del Generatore (Verde) esplode. Significa che il discriminatore è troppo forte e il generatore non impara nulla.
* **Failure Mode (Mode Collapse):** Le loss rimangono piatte e costanti fin dall'inizio senza variazioni.

Fammi sapere quando hai generato il grafico o se hai errori, così potremo commentare i risultati e passare allo **Stress Test (Imbalance)**.

---

**You:**
continua

**Gemini:**
Molto bene. Procediamo con il prossimo punto del piano: **Studio dell'Imbalance (Stress Test)**.

L'obiettivo è verificare come reagisce il modello se le frodi (già rare) diventano ancora più rare nel dataset di training. Questo simula uno scenario in cui abbiamo pochissimi esempi di anomalie da cui imparare.

Ridurremo del **50%** il numero di anomalie (Global e Local) presenti nel dataset, addestreremo l'AAE su questo dataset "impoverito" e valuteremo se il MAUC crolla o regge.

### Passo 1: Script di Training con Imbalance

Crea il file `project_extension/train_imbalance.py`.
Questo script:

1. Carica il dataset originale.
2. Effettua un subsampling mirato: mantiene tutti i dati "regular" ma scarta casualmente metà delle anomalie.
3. Addestra l'AAE su questo nuovo dataset sbilanciato.
4. Salva lo spazio latente generato (solo per i dati rimasti).

Copia il seguente codice:

```
import sys
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

# Setup path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Encoder import Encoder
from Decoder import Decoder
from Discriminator import Discriminator

def train_imbalance_experiment():
    print("--- STRESS TEST IMBALANCE (50% ANOMALIES DROPPED) ---")

    # --- CONFIGURAZIONE ---
    SEED_VALUE = 1234
    LATENT_DIM = 10
    TAU = 20
    BATCH_SIZE = 128
    EPOCHS = 50 # Un po' meno epoche per rapidità, dato che il dataset è più piccolo
    LR_ENC = 1e-3
    LR_DEC = 1e-3
    LR_DISC = 1e-5
    DATA_PATH = 'data/fraud_dataset_v2.csv'
    OUTPUT_FILE = f"latent_data_sets/ldim{LATENT_DIM}_imbalance50_basisLS.txt"

    # Determinismo
    random.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED_VALUE)

    # --- 1. CARICAMENTO E CREAZIONE DATASET SBILANCIATO ---
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Dataset non trovato in {DATA_PATH}")
        return

    print("Caricamento dati...")
    full_df = pd.read_csv(DATA_PATH)
    
    # Separazione classi
    df_regular = full_df[full_df['label'] == 'regular']
    df_global = full_df[full_df['label'] == 'global']
    df_local = full_df[full_df['label'] == 'local']
    
    print(f"Original counts - Regular: {len(df_regular)}, Global: {len(df_global)}, Local: {len(df_local)}")
    
    # Dimezziamo le anomalie
    df_global_red = df_global.sample(frac=0.5, random_state=SEED_VALUE)
    df_local_red = df_local.sample(frac=0.5, random_state=SEED_VALUE)
    
    print(f"Reduced counts  - Regular: {len(df_regular)}, Global: {len(df_global_red)}, Local: {len(df_local_red)}")
    
    # Ricostruzione dataset
    imbalanced_df = pd.concat([df_regular, df_global_red, df_local_red]).sample(frac=1, random_state=SEED_VALUE).reset_index(drop=True)
    
    # Salviamo le label per dopo
    labels = imbalanced_df.pop('label')
    
    # --- 2. PREPROCESSING (Standard) ---
    categorical_attr = ['KTOSL', 'PRCTR', 'BSCHL', 'HKONT', 'BUKRS', 'WAERS']
    dataset_categ = pd.get_dummies(imbalanced_df[categorical_attr])

    numeric_attr_names = ['DMBTR', 'WRBTR']
    numeric_attr = imbalanced_df[numeric_attr_names] + 1e-4
    numeric_attr = numeric_attr.apply(np.log)
    dataset_numeric = (numeric_attr - numeric_attr.min()) / (numeric_attr.max() - numeric_attr.min())

    dataset_processed = pd.concat([dataset_categ, dataset_numeric], axis=1)
    input_dim = dataset_processed.shape[1]
    
    torch_dataset = torch.FloatTensor(dataset_processed.values)
    dataloader = DataLoader(torch_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- 3. SETUP MODELLI (AAE) ---
    encoder = Encoder(input_size=input_dim, hidden_size=[256, 64, 16, 4, LATENT_DIM])
    decoder = Decoder(output_size=input_dim, hidden_size=[LATENT_DIM, 4, 16, 64, 256])
    discriminator = Discriminator(input_size=LATENT_DIM, hidden_size=[256, 16, 4, 2], output_size=1)

    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        discriminator = discriminator.cuda()

    optim_enc = optim.Adam(encoder.parameters(), lr=LR_ENC)
    optim_dec = optim.Adam(decoder.parameters(), lr=LR_DEC)
    optim_disc = optim.Adam(discriminator.parameters(), lr=LR_DISC)

    criterion_bce = nn.BCELoss()
    criterion_mse = nn.MSELoss()

    # --- 4. PREPARAZIONE PRIOR ---
    radius = 0.8
    sigma = 0.01
    samples_per_gaussian = 10000
    centroids = []
    for i in range(LATENT_DIM):
        if i % 2 == 0:
            centroids.append((radius * np.cos(np.linspace(0, 2 * np.pi, TAU, endpoint=False)) + 1) / 2)
        else:
            centroids.append((radius * np.sin(np.linspace(0, 2 * np.pi, TAU, endpoint=False)) + 1) / 2)
    mu_gauss = np.vstack(centroids).T
    
    z_real_samples_pool = []
    for mu in mu_gauss:
        samples = np.random.normal(mu, sigma, size=(samples_per_gaussian, LATENT_DIM))
        z_real_samples_pool.append(samples)
    z_real_samples_pool = np.vstack(z_real_samples_pool)
    
    # --- 5. TRAINING LOOP ---
    print(f"Training AAE su dataset ridotto per {EPOCHS} epoche...")
    encoder.train()
    decoder.train()
    discriminator.train()

    for epoch in range(EPOCHS):
        for batch in dataloader:
            if torch.cuda.is_available(): batch = batch.cuda()

            # Reconstruction
            encoder.zero_grad(); decoder.zero_grad()
            z = encoder(batch)
            recon = decoder(z)
            # Semplificazione loss per velocità
            loss_recon = criterion_mse(recon, batch) 
            loss_recon.backward()
            optim_enc.step(); optim_dec.step()

            # Discriminator
            encoder.zero_grad(); discriminator.zero_grad()
            z_real = torch.FloatTensor(z_real_samples_pool[np.random.randint(0, len(z_real_samples_pool), batch.size(0))])
            if torch.cuda.is_available(): z_real = z_real.cuda()
            
            d_real = discriminator(z_real)
            z_fake = encoder(batch)
            d_fake = discriminator(z_fake.detach())
            
            loss_disc = criterion_bce(d_real, torch.ones_like(d_real)) + criterion_bce(d_fake, torch.zeros_like(d_fake))
            loss_disc.backward()
            optim_disc.step()

            # Generator
            encoder.zero_grad(); discriminator.zero_grad()
            z_fake = encoder(batch)
            d_fake = discriminator(z_fake)
            loss_gen = criterion_bce(d_fake, torch.ones_like(d_fake))
            loss_gen.backward()
            optim_enc.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} completata.")

    # --- 6. GENERAZIONE E SALVATAGGIO SPAZIO LATENTE ---
    print(f"Generazione spazio latente e salvataggio in {OUTPUT_FILE}...")
    encoder.eval()
    
    # Processiamo l'intero dataset sbilanciato ordinato per salvare le label corrette
    # Nota: usiamo torch_dataset che corrisponde a imbalanced_df
    if torch.cuda.is_available():
        full_tensor = torch_dataset.cuda()
    else:
        full_tensor = torch_dataset

    with torch.no_grad():
        z_out = encoder(full_tensor).cpu().numpy()

    # Mapping labels
    label_map = {'regular': 0, 'global': 1, 'local': 2}
    numeric_labels = labels.map(label_map).values

    with open(OUTPUT_FILE, "w") as f:
        for i in range(len(z_out)):
            features_str = "\t".join([str(x) for x in z_out[i]])
            line = f"{features_str}\t{numeric_labels[i]}\n"
            f.write(line)

    print("Fatto.")

if __name__ == "__main__":
    train_imbalance_experiment()

```

### Passo 2: Script di Valutazione Imbalance

Dato che abbiamo già uno script di valutazione generico (`evaluate_comparative.py`), possiamo riutilizzarlo semplicemente puntandolo al nuovo file. Tuttavia, per mantenere ordine e pulizia come richiesto, creeremo un piccolo script wrapper che invoca la valutazione sul file corretto.

Crea `project_extension/evaluate_imbalance.py`:

```
import os
from evaluate_comparative import evaluate_latent_space

def main():
    target_file = "latent_data_sets/ldim10_imbalance50_basisLS.txt"
    
    print("--- VALUTAZIONE STRESS TEST IMBALANCE ---")
    if os.path.exists(target_file):
        results = evaluate_latent_space(target_file)
        
        print("\n--- RISULTATI IMBALANCE ---")
        for clf, score in results.items():
            print(f"Classifier: {clf} | MAUC: {score:.4f}")
            
        print("\nConfronta questi valori con la baseline originale.")
        print("Se i valori sono simili, il modello è robusto allo sbilanciamento.")
        print("Se crollano drasticamente, il modello soffre la scarsità di esempi anomali.")
    else:
        print(f"File {target_file} non trovato. Esegui prima train_imbalance.py")

if __name__ == "__main__":
    main()

```

### Esecuzione

1. Esegui il training sbilanciato:

```
python project_extension/train_imbalance.py

```
2. Valuta i risultati:

```
python project_extension/evaluate_imbalance.py

```

Quando hai i risultati, fammelo sapere. L'ultimo step sarà l'**Analisi Qualitativa (t-SNE)** per visualizzare come l'AAE separa le classi nello spazio latente.

---

**You:**
continua

**Gemini:**
Eccoci all'ultimo step del piano di lavoro: **Analisi Qualitativa (Visualizzazione t-SNE)**.

I numeri (MAUC) ci dicono *quanto* il modello è bravo, ma non *come* struttura i dati. La visualizzazione t-SNE ci permette di proiettare lo spazio latente a 10 dimensioni in 2 dimensioni, per vedere se il modello ha imparato a raggruppare le transazioni simili e, soprattutto, se ha separato le anomalie (Globali e Locali) dai dati regolari e tra di loro.

### Passo 1: Script di Visualizzazione t-SNE

Crea il file `project_extension/visualize_tsne.py`.
Questo script:

1. Carica uno dei dataset latenti generati (es. quello della baseline o quello dell'Autoencoder Standard).
2. Effettua un campionamento intelligente: t-SNE è molto lento su grandi dataset. Prenderemo **tutte** le anomalie (poiché sono poche e preziose) e un sottoinsieme casuale di transazioni regolari per mantenere il grafico leggibile ma rappresentativo.
3. Esegue l'algoritmo t-SNE.
4. Genera un grafico a dispersione colorato per classe.

Incolla il seguente codice:

```
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def load_latent_data(filepath):
    """Carica il dataset latente dal formato txt tab-separated."""
    print(f"Caricamento dati da {filepath}...")
    X = []
    labels = []
    
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split('\t')
            # Le ultime cifre sono la label
            features = [float(z) for z in parts[:-1]]
            lbl = int(parts[-1])
            
            X.append(features)
            labels.append(lbl)
            
    return np.array(X), np.array(labels)

def run_tsne_analysis(target_file, output_image, sample_size_regular=2000):
    if not os.path.exists(target_file):
        print(f"[ERROR] File non trovato: {target_file}")
        return

    # 1. Caricamento Dati
    X, y = load_latent_data(target_file)
    
    # 2. Subsampling Strategico
    # t-SNE è computazionalmente pesante (O(N^2)). 
    # Manteniamo TUTTE le anomalie (Global=1, Local=2) e campioniamo i Regular (0).
    
    idx_regular = np.where(y == 0)[0]
    idx_global = np.where(y == 1)[0]
    idx_local = np.where(y == 2)[0]
    
    print(f"Statistiche originali: Regular={len(idx_regular)}, Global={len(idx_global)}, Local={len(idx_local)}")
    
    # Campionamento Regular
    if len(idx_regular) > sample_size_regular:
        np.random.seed(1234)
        idx_regular_sub = np.random.choice(idx_regular, sample_size_regular, replace=False)
    else:
        idx_regular_sub = idx_regular
        
    # Unione indici
    idx_final = np.concatenate([idx_regular_sub, idx_global, idx_local])
    X_subset = X[idx_final]
    y_subset = y[idx_final]
    
    print(f"Punti per t-SNE: {len(X_subset)} (Regular sub: {len(idx_regular_sub)})")

    # 3. Esecuzione t-SNE
    print("Esecuzione t-SNE (potrebbe richiedere qualche minuto)...")
    tsne = TSNE(n_components=2, random_state=1234, perplexity=30, n_iter=1000, verbose=1)
    X_embedded = tsne.fit_transform(X_subset)
    
    # 4. Plotting
    print("Generazione grafico...")
    df_plot = pd.DataFrame(X_embedded, columns=['x', 'y'])
    
    # Mapping label per legenda leggibile
    label_map = {0: 'Regular', 1: 'Global Anomaly', 2: 'Local Anomaly'}
    df_plot['label'] = [label_map[i] for i in y_subset]
    
    plt.figure(figsize=(10, 8))
    
    # Palette colori simile al paper (Navy, Turquoise, Orange)
    palette = {'Regular': 'navy', 'Global Anomaly': 'turquoise', 'Local Anomaly': 'darkorange'}
    
    sns.scatterplot(
        data=df_plot, x='x', y='y', hue='label', 
        palette=palette, alpha=0.6, s=60, edgecolor=None
    )
    
    plt.title(f't-SNE Projection of Latent Space\nSource: {os.path.basename(target_file)}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(output_image, dpi=300)
    print(f"Grafico salvato in: {output_image}")
    plt.close()

if __name__ == "__main__":
    # Definisci qui quale file vuoi visualizzare. 
    # Ideale confrontare l'AAE baseline con l'Autoencoder Standard.
    
    # Esempio 1: AAE Baseline (File generato dal passo 2 - assicurati che esista)
    # Cerca un file generato dalla baseline, es: latent_data_sets/ldim10_tau20_basisLS.txt
    # Se non lo trovi, usa quello che hai generato tu (es. imbalance o standard)
    
    # Cerchiamo automaticamente un file candidato valido
    candidates = [
        "latent_data_sets/ldim10_tau20_basisLS.txt",       # Baseline originale
        "latent_data_sets/ldim10_standardAE_basisLS.txt", # Nostro Standard AE
        "latent_data_sets/ldim10_imbalance50_basisLS.txt" # Nostro Stress Test
    ]
    
    found = False
    for f in candidates:
        if os.path.exists(f):
            output = f.replace(".txt", "_tSNE.png").replace("latent_data_sets", "project_extension")
            run_tsne_analysis(f, output)
            found = True
            # Puoi togliere il break se vuoi generarli tutti
            # break 
    
    if not found:
        print("Nessun dataset latente trovato in latent_data_sets/. Esegui prima i training.")

```

### Passo 2: Esecuzione

Esegui lo script:

```
python project_extension/visualize_tsne.py

```

### Interpretazione dei Risultati Finali

Ora che abbiamo completato tutti gli step, ecco come dovresti assemblare e interpretare i risultati per il tuo progetto finale:

1. **Baseline (Riproducibilità):**

* Hai ottenuto i valori MAUC originali. Questi sono il tuo "benchmark". Se il MAUC è alto (es. > 0.90), confermi che il paper dice il vero.
2. **AAE vs Standard AE (Confronto Architetturale):**

* Confronta i MAUC di `evaluate_comparative.py` (Standard AE) con quelli della baseline.
* **Ipotesi:** Se l'AAE è migliore, significa che forzare la distribuzione latente verso una mistura di Gaussiane aiuta effettivamente a separare meglio le anomalie ("regolarizzazione avversaria"). Se sono uguali, l'AAE potrebbe essere un'architettura ridondante.
3. **Analisi Stabilità (Loss Curves):**

* Guarda `loss_curves.png`.
* Dimostra se il training è stabile. Un buon segnale è vedere la Reconstruction Loss scendere e le Loss di Generatore/Discriminatore oscillare senza divergere. Questo valida la robustezza dell'implementazione.
4. **Stress Test (Imbalance):**

* Confronta i MAUC di `evaluate_imbalance.py` con la baseline.
* Se il calo di performance è lieve, l'AAE è molto robusto e adatto a scenari reali di frode (dove i casi positivi sono rari). Se crolla, hai trovato un limite del modello.
5. **Qualità Visiva (t-SNE):**

* Guarda i grafici PNG generati (`_tSNE.png`).
* Cerca cluster ben definiti. Le anomalie Globali dovrebbero essere ben separate (spesso in cluster propri). Le anomalie Locali sono più insidiose e potrebbero sovrapporsi leggermente ai dati regolari (o formare piccoli gruppi vicini ai cluster regolari).

Il progetto ora è completo di codice, esperimenti ed estensioni. Hai tutto il necessario nella cartella `project_extension` senza aver sporcato il codice originale. Buon lavoro con la stesura del report!

---

**You:**
manca qualcosa rispetto al piano?

**Gemini:**
Rispetto al piano di lavoro originale che ci siamo dati, **abbiamo coperto tutti i punti tecnici e implementativi**.

Ecco una checklist di verifica:

1. ✅ **Riproducibilità (Baseline):** Coperta da `reproduce_baseline.py`.
2. ✅ **Confronto Architetturale (AAE vs AE):** Coperto da `train_standard_ae.py` e `evaluate_comparative.py`. (Il confronto con le feature "Raw" è coperto numericamente dallo script originale `MAUC_rawExp.py` che viene lanciato nella baseline).
3. ✅ **Stabilità del Training:** Coperta da `train_aae_stability.py` e relativo grafico delle loss.
4. ✅ **Imbalance (Stress Test):** Coperto da `train_imbalance.py` e `evaluate_imbalance.py`.
5. ✅ **Analisi Qualitativa (t-SNE):** Coperta da `visualize_tsne.py`.

### C'è solo una "finezza" mancante (Opzionale ma raccomandata)

Nel piano citavi il confronto con le **feature originali**.
Mentre abbiamo i numeri (MAUC) per le feature originali, non abbiamo la **visualizzazione t-SNE delle feature originali**.

Vedere il t-SNE dei dati "grezzi" (Raw Data) è utilissimo per dire: *"Guardate, i dati originali sono un pasticcio sovrapposto, mentre il mio AAE (e anche l'AE) li separa bene nello spazio latente"*. È l'immagine "Prima vs Dopo" che rende forte la tesi.

Se vuoi aggiungere questa ciliegina sulla torta, ecco una piccola modifica per `project_extension/visualize_tsne.py` per fargli accettare anche il CSV originale.

### Aggiornamento per `visualize_tsne.py` (Opzionale)

Sostituisci il contenuto di `project_extension/visualize_tsne.py` con questo codice aggiornato che gestisce anche il file `.csv` grezzo:

```
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def load_data(filepath):
    """Carica dati sia da latent TXT che da raw CSV."""
    print(f"Caricamento dati da {filepath}...")
    
    if filepath.endswith('.csv'):
        # Gestione Raw Data (CSV)
        df = pd.read_csv(filepath)
        # Assumiamo che la colonna label esista e mappiamo le stringhe a int
        label_map_rev = {'regular': 0, 'global': 1, 'local': 2}
        if 'label' in df.columns:
            labels = df['label'].map(label_map_rev).values
            # Rimuoviamo colonne non feature se necessario, o prendiamo solo le numeriche processate
            # Per semplicità, qui facciamo un preprocessing veloce simile al training
            numeric_attr = ['DMBTR', 'WRBTR']
            # Log scale veloce
            for col in numeric_attr:
                if col in df.columns:
                    df[col] = np.log(df[col] + 1e-4)
            
            # Selezioniamo solo colonne numeriche per t-SNE
            X = df.select_dtypes(include=[np.number]).values
            # Rimuoviamo eventuali NaN generati
            X = np.nan_to_num(X)
            return X, labels
        else:
            print("Errore: colonna 'label' non trovata nel CSV.")
            return None, None

    else:
        # Gestione Latent Data (TXT)
        X = []
        labels = []
        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split('\t')
                features = [float(z) for z in parts[:-1]]
                lbl = int(parts[-1])
                X.append(features)
                labels.append(lbl)
        return np.array(X), np.array(labels)

def run_tsne_analysis(target_file, output_image, sample_size_regular=2000):
    if not os.path.exists(target_file):
        print(f"[ERROR] File non trovato: {target_file}")
        return

    X, y = load_data(target_file)
    if X is None: return

    # Subsampling
    idx_regular = np.where(y == 0)[0]
    idx_global = np.where(y == 1)[0]
    idx_local = np.where(y == 2)[0]
    
    print(f"Statistiche: Regular={len(idx_regular)}, Global={len(idx_global)}, Local={len(idx_local)}")
    
    if len(idx_regular) > sample_size_regular:
        np.random.seed(1234)
        idx_regular_sub = np.random.choice(idx_regular, sample_size_regular, replace=False)
    else:
        idx_regular_sub = idx_regular
        
    idx_final = np.concatenate([idx_regular_sub, idx_global, idx_local])
    X_subset = X[idx_final]
    y_subset = y[idx_final]
    
    print(f"Punti per t-SNE: {len(X_subset)}")

    # Esecuzione t-SNE
    print("Esecuzione t-SNE...")
    tsne = TSNE(n_components=2, random_state=1234, perplexity=30, n_iter=1000, verbose=1)
    X_embedded = tsne.fit_transform(X_subset)
    
    # Plotting
    df_plot = pd.DataFrame(X_embedded, columns=['x', 'y'])
    label_map = {0: 'Regular', 1: 'Global Anomaly', 2: 'Local Anomaly'}
    df_plot['label'] = [label_map[i] for i in y_subset]
    
    plt.figure(figsize=(10, 8))
    palette = {'Regular': 'navy', 'Global Anomaly': 'turquoise', 'Local Anomaly': 'darkorange'}
    
    sns.scatterplot(
        data=df_plot, x='x', y='y', hue='label', 
        palette=palette, alpha=0.6, s=60, edgecolor=None
    )
    
    plt.title(f't-SNE Projection\nSource: {os.path.basename(target_file)}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Grafico salvato in: {output_image}")
    plt.close()

if __name__ == "__main__":
    # 1. Latent Space (AAE)
    if os.path.exists("latent_data_sets/ldim10_tau20_basisLS.txt"):
        run_tsne_analysis("latent_data_sets/ldim10_tau20_basisLS.txt", "project_extension/tsne_aae.png")
    
    # 2. Latent Space (Standard AE)
    if os.path.exists("latent_data_sets/ldim10_standardAE_basisLS.txt"):
        run_tsne_analysis("latent_data_sets/ldim10_standardAE_basisLS.txt", "project_extension/tsne_ae.png")

    # 3. Raw Data (Nuovo!)
    if os.path.exists("data/fraud_dataset_v2.csv"):
        print("\n--- Generazione t-SNE per dati RAW (Originali) ---")
        run_tsne_analysis("data/fraud_dataset_v2.csv", "project_extension/tsne_raw.png")

```

### Ordine di esecuzione Finale (Recap)

Se sei pronto per chiudere il progetto, questo è l'ordine logico per generare tutto il materiale per il report:

1. `python project_extension/reproduce_baseline.py` (Scarica dati e crea baseline AAE)
2. `python project_extension/train_standard_ae.py` (Addestra AE standard)
3. `python project_extension/evaluate_comparative.py` (Confronta i numeri MAUC)
4. `python project_extension/train_aae_stability.py` (Genera grafico loss stability)
5. `python project_extension/train_imbalance.py` (Addestra su dataset ridotto)
6. `python project_extension/evaluate_imbalance.py` (Verifica robustezza imbalance)
7. `python project_extension/visualize_tsne.py` (Genera i 3 grafici di confronto visivo: Raw, AE, AAE)

Hai tutto. Buon lavoro!

---

