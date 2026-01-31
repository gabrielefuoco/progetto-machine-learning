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
    EPOCHS = 100
    LR_ENC = 1e-3
    LR_DEC = 1e-3
    LR_DISC = 1e-5 # Leaning rate basso per il discriminatore
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
    dataset_categ = pd.get_dummies(ori_dataset[categorical_attr], dtype=int)

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

            # Split numeric/categorical loss simplified
            # Se vogliamo matchare esattamente l'originale, il MSE è applicato a tutto nel train originale
            # validation_v0.py:108 -> loss = criterion(reconstruction, batch_features)
            
            loss_recon = criterion_mse(recon_batch, batch)
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
