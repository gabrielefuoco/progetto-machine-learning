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
    EPOCHS = 50 
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
    dataset_categ = pd.get_dummies(imbalanced_df[categorical_attr], dtype=int)

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

    # --- 4. DATASET VALIDATION (PRIOR) ---
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
    np.random.shuffle(z_real_samples_pool)

    # --- 5. TRAINING LOOP (Condensed) ---
    print(f"Inizio training su dataset {len(imbalanced_df)} samples per {EPOCHS} epoche...")
    
    encoder.train()
    decoder.train()
    discriminator.train()

    for epoch in range(EPOCHS):
        for batch in dataloader:
            if torch.cuda.is_available(): batch = batch.cuda()

            # Reconstruction
            encoder.zero_grad(); decoder.zero_grad(); discriminator.zero_grad()
            z_sample = encoder(batch)
            recon_batch = decoder(z_sample)
            loss_recon = criterion_mse(recon_batch, batch)
            loss_recon.backward()
            optim_enc.step(); optim_dec.step()

            # Discriminator
            encoder.zero_grad(); discriminator.zero_grad()
            idx = np.random.randint(0, z_real_samples_pool.shape[0], batch.size(0))
            z_real = torch.FloatTensor(z_real_samples_pool[idx])
            if torch.cuda.is_available(): z_real = z_real.cuda()
            d_real = discriminator(z_real)
            loss_d_real = criterion_bce(d_real, torch.ones_like(d_real))
            
            z_fake = encoder(batch)
            d_fake = discriminator(z_fake.detach())
            loss_d_fake = criterion_bce(d_fake, torch.zeros_like(d_fake))
            
            loss_disc = loss_d_real + loss_d_fake
            loss_disc.backward()
            optim_disc.step()

            # Generator
            encoder.zero_grad(); discriminator.zero_grad()
            z_fake_gen = encoder(batch)
            d_fake_gen = discriminator(z_fake_gen)
            loss_gen = criterion_bce(d_fake_gen, torch.ones_like(d_fake_gen))
            loss_gen.backward()
            optim_enc.step()

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}")

    # --- 6. SALVATAGGIO ---
    print(f"Salvataggio dataset latente in {OUTPUT_FILE}...")
    encoder.eval()
    with torch.no_grad():
        full_z = encoder(torch_dataset.cuda() if torch.cuda.is_available() else torch_dataset)
        full_z_np = full_z.cpu().numpy()

    label_map = {'regular': 0, 'global': 1, 'local': 2}
    numeric_labels = labels.map(label_map).values

    if not os.path.exists('latent_data_sets'):
        os.makedirs('latent_data_sets')

    with open(OUTPUT_FILE, "w") as f:
        for i in range(len(full_z_np)):
            features_str = "\t".join([str(x) for x in full_z_np[i]])
            line = f"{features_str}\t{numeric_labels[i]}\n"
            f.write(line)

    print("Fatto.")

if __name__ == "__main__":
    train_imbalance_experiment()
