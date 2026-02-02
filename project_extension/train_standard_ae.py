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
    EPOCHS = 100
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

    # Preprocessing Categoriale - USE SPARSE to save memory
    categorical_attr = ['KTOSL', 'PRCTR', 'BSCHL', 'HKONT', 'BUKRS', 'WAERS']
    dataset_categ = pd.get_dummies(ori_dataset[categorical_attr], sparse=True, dtype=int)

    # Preprocessing Numerico
    numeric_attr_names = ['DMBTR', 'WRBTR']
    # + 1e-4 come nell'originale per evitare log(0)
    numeric_attr = ori_dataset[numeric_attr_names] + 1e-4
    numeric_attr = numeric_attr.apply(np.log)
    dataset_numeric = (numeric_attr - numeric_attr.min()) / (numeric_attr.max() - numeric_attr.min())

    # Merge - remains sparse
    dataset_processed = pd.concat([dataset_categ, dataset_numeric], axis=1)
    input_dim = dataset_processed.shape[1]
    
    # Pre-conversion to numpy float32 to speed up batch slicing
    # This fits in memory (~500MB) unlike the full Tensor (~2.5GB due to overhead)
    print("Conversione set dati in NumPy...")
    data_np = dataset_processed.values.astype(np.float32)
    total_rows = len(data_np)
    
    # --- 3. DEFINIZIONE MODELLO ---
    # Usiamo le stesse classi originali per garantire confronto architetturale equo
    encoder = Encoder(input_size=input_dim, hidden_size=[256, 64, 16, 4, LATENT_DIM])
    decoder = Decoder(output_size=input_dim, hidden_size=[LATENT_DIM, 4, 16, 64, 256])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)

    # --- 4. TRAINING LOOP (Memory-efficient batch processing) ---
    print(f"Inizio training per {EPOCHS} epoche... (con ottimizzazione NumPy)")
    encoder.train()
    decoder.train()
    
    # Shuffle indices for each epoch
    indices = np.arange(total_rows)

    for epoch in range(EPOCHS):
        np.random.shuffle(indices)
        total_loss = 0
        num_batches = 0
        
        for i in range(0, total_rows, BATCH_SIZE):
            batch_indices = indices[i:i+BATCH_SIZE]
            
            # SLICING NUMPY IS FAST
            batch_np = data_np[batch_indices] 
            
            # Transfer to GPU
            batch = torch.from_numpy(batch_np).to(device)
            
            # Forward
            z = encoder(batch)
            reconstruction = decoder(z)
            
            loss = criterion(reconstruction, batch)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Log every epoch
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/num_batches:.6f}")

    # --- 5. GENERAZIONE SPAZIO LATENTE (batch-wise to avoid OOM) ---
    print("Generazione rappresentazione latente...")
    encoder.eval()
    
    latent_vectors = []
    with torch.no_grad():
        for i in range(0, total_rows, BATCH_SIZE):
            # SLICING NUMPY IS FAST
            batch_np = data_np[i:i+BATCH_SIZE]
            
            batch = torch.from_numpy(batch_np).to(device)
            z = encoder(batch)
            latent_vectors.append(z.cpu().numpy())
            
    full_z_np = np.vstack(latent_vectors)

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
