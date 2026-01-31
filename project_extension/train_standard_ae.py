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

    # Preprocessing Categoriale
    categorical_attr = ['KTOSL', 'PRCTR', 'BSCHL', 'HKONT', 'BUKRS', 'WAERS']
    dataset_categ = pd.get_dummies(ori_dataset[categorical_attr], dtype=int)

    # Preprocessing Numerico
    numeric_attr_names = ['DMBTR', 'WRBTR']
    # + 1e-4 come nell'originale per evitare log(0)
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
