import sys
import os
import time
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

# Add root to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Encoder import Encoder
from Decoder import Decoder
from Discriminator import Discriminator

# Configuration
LATENT_DIM = 10
TAU = 10  # Number of Gaussians for Mixture Prior
N_RUNS = 5
EPOCHS = 50 
BATCH_SIZE = 2048 # Optimized for GPU saturation (was 128)
LR = 0.0001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample_prior(batch_size, z_dim, n_gaussians, radius=0.8, sigma=0.01):
    """
    Sample from a Mixture of Gaussians (Swiss Roll or Circle like distribution).
    Mimics the logic suggested by the 'tau', 'radius', 'sigma' parameters.
    """
    # Sample indices of gaussians (0 to n_gaussians-1)
    indices = np.random.randint(0, n_gaussians, size=batch_size)
    
    # Calculate centers of gaussians (distributed on a 2D circle for first 2 dims, 0 elsewhere? 
    # Or hyper-sphere? The paper usually projects 2D structure. 
    # Let's assume a 2D circle layout for the first 2 dimensions and Gaussian(0, sigma) for others 
    # OR the mixture is in the full latent space.
    # Given the visualization shows a star/cluster shape, let's assume simple Gaussian(0,1) 
    # if we want to test "std AE vs AAE" baseline where AAE imposes Gaussian.
    # BUT, the parameters 'radius=0.8' imply a geometric structure.
    # Let's try to match the likely implementation:
    # 2D Ring for dims 0,1 and Normal(0, sigma) for rest?
    
    if z_dim != LATENT_DIM:
        raise ValueError("Latent dim mismatch")

    # If the paper uses a "semi-supervised" style with categorical labels, 
    # the structure might be different. Here we are doing Unsupervised/Semi (Label is hidden?).
    # Let's implement a standard Gaussian(0,1) for simplicity first OR 
    # a proper mixture if we want to reproduce the "star" shape constraints.
    # Given the t-SNE result "star shape", it looks like a centralized distribution.
    
    # However, to be robust, let's implement the Mixture:
    # Centers on a circle in 2D plane (dim 0, 1)
    
    # Vectorized implementation
    # Sample indices
    indices = np.random.randint(0, n_gaussians, size=batch_size)
    
    # Calculate angles
    angles = 2 * np.pi * indices / n_gaussians
    
    # Calculate centers
    cx = radius * np.cos(angles)
    cy = radius * np.sin(angles)
    
    # Sample noise for all at once
    z = np.random.normal(0, sigma, size=(batch_size, z_dim)).astype(np.float32)
    
    # Add centers to first two components
    z[:, 0] += cx
    z[:, 1] += cy
        
    return torch.from_numpy(z).to(DEVICE)

def load_data():
    print("Loading data...")
    # Logic copied from train_standard_ae.py
    local_csv = 'data/fraud_dataset_v2.csv'
    if not os.path.exists(local_csv):
        url = 'https://raw.githubusercontent.com/GitiHubi/deepAI/master/data/fraud_dataset_v2.csv'
        print(f"Downloading {url}...")
        df = pd.read_csv(url)
        df.to_csv(local_csv, index=False)
    else:
        df = pd.read_csv(local_csv)

    categorical_attr = ['KTOSL', 'PRCTR', 'BSCHL', 'HKONT', 'BUKRS', 'WAERS']
    data_categ = pd.get_dummies(df[categorical_attr], sparse=True)

    numeric_attr = ['DMBTR', 'WRBTR']
    data_num = df[numeric_attr] + 1e-4
    data_num = data_num.apply(np.log)
    data_num = (data_num - data_num.min()) / (data_num.max() - data_num.min())

    # Sparse concat
    data_all = pd.concat([data_categ, data_num], axis=1)
    
    # Convert to dense numpy array once
    print("Converting to dense numpy array (float32)...")
    data_np = data_all.values.astype(np.float32)
    return data_np, data_all.shape[1]

def train_run(run_id, data_np, input_dim):
    print(f"\n--- Run {run_id+1}/{N_RUNS} ---")
    
    # Init Models
    # Hidden sizes from generateDatasets.py: Enc=[256, 64, 16, 4, latent], Dec=[latent, 4, 16, 64, 256]
    encoder = Encoder(input_size=input_dim, hidden_size=[256, 64, 16, 4, LATENT_DIM]).to(DEVICE)
    decoder = Decoder(output_size=input_dim, hidden_size=[LATENT_DIM, 4, 16, 64, 256]).to(DEVICE)
    discriminator = Discriminator(input_size=LATENT_DIM, hidden_size=[256, 16, 4], output_size=1).to(DEVICE)

    # Optimizers
    # Using separate optimizers
    optim_enc = optim.Adam(encoder.parameters(), lr=LR)
    optim_dec = optim.Adam(decoder.parameters(), lr=LR)
    optim_disc = optim.Adam(discriminator.parameters(), lr=LR)
    # Generator optimizer (updates encoder to fool discriminator)
    # Often Encoder has two optimizers or we just use one. 
    # Let's use a separate optimizer for the generator phase to be safe/clear.
    optim_gen = optim.Adam(encoder.parameters(), lr=LR) 

    criterion_recon = nn.MSELoss()
    criterion_adv = nn.BCELoss() # Binary Cross Entropy

    history = {'epoch': [], 'recon_loss': [], 'disc_loss': [], 'gen_loss': []}

    n_samples = data_np.shape[0]
    n_batches = n_samples // BATCH_SIZE

    start_time = time.time()
    
    # Use DataLoader for speed
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(data_np))
    # num_workers=0 is safer on Windows/Docker for now, pin_memory=True helps GPU transfer
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True) 
    
    n_batches = len(dataloader)

    for epoch in range(EPOCHS):
        epoch_recon = 0
        epoch_disc = 0
        epoch_gen = 0
        
        for (batch_data,) in dataloader:
            batch_data = batch_data.to(DEVICE)
            curr_batch_size = batch_data.size(0)
            
            # --- Phase 1: Reconstruction ---
            optim_enc.zero_grad()
            optim_dec.zero_grad()
            
            z = encoder(batch_data)
            recon_batch = decoder(z)
            
            loss_recon = criterion_recon(recon_batch, batch_data)
            loss_recon.backward()
            
            optim_enc.step()
            optim_dec.step()
            
            epoch_recon += loss_recon.item()
            
            # --- Phase 2: Adversarial (Regularization) ---
            
            # 2a. Train Discriminator
            optim_disc.zero_grad()
            
            # Real samples (Prior)
            z_real = sample_prior(curr_batch_size, LATENT_DIM, TAU).to(DEVICE)
            d_real = discriminator(z_real)
            # Label 1 for Real
            label_real = torch.ones(curr_batch_size, 1).to(DEVICE)
            loss_d_real = criterion_adv(d_real, label_real)
            
            # Fake samples (Encoder output detached)
            z_fake = encoder(batch_data).detach() 
            d_fake = discriminator(z_fake)
            # Label 0 for Fake
            label_fake = torch.zeros(curr_batch_size, 1).to(DEVICE)
            loss_d_fake = criterion_adv(d_fake, label_fake)
            
            loss_disc = loss_d_real + loss_d_fake
            loss_disc.backward()
            optim_disc.step()
            
            epoch_disc += loss_disc.item()
            
            # 2b. Train Generator (Encoder)
            optim_gen.zero_grad()
            
            # We want Encoder to produce z that Discriminator classifies as Real (1)
            z_gen = encoder(batch_data) # Re-compute with gradients connected
            d_gen = discriminator(z_gen)
            
            loss_gen = criterion_adv(d_gen, label_real) # Trick: Label 1
            loss_gen.backward()
            optim_gen.step()
            
            epoch_gen += loss_gen.item()

        # Average losses
        avg_recon = epoch_recon / n_batches
        avg_disc = epoch_disc / n_batches
        avg_gen = epoch_gen / n_batches
        
        history['epoch'].append(epoch)
        history['recon_loss'].append(avg_recon)
        history['disc_loss'].append(avg_disc)
        history['gen_loss'].append(avg_gen)
        
        if (epoch+1) % 10 == 0:
            print(f"Ep {epoch+1}/{EPOCHS} | Recon: {avg_recon:.6f} | Disc: {avg_disc:.4f} | Gen: {avg_gen:.4f}")

    print(f"Run {run_id+1} finished in {time.time()-start_time:.1f}s")
    return pd.DataFrame(history)

def main():
    if not os.path.exists("results/stability"):
        os.makedirs("results/stability")
        
    data_np, input_dim = load_data()
    
    all_runs = []
    
    for run in range(N_RUNS):
        df_hist = train_run(run, data_np, input_dim)
        df_hist['run'] = run
        all_runs.append(df_hist)
        
    # Aggregate results
    final_df = pd.concat(all_runs)
    final_df.to_csv("results/stability/training_logs.csv", index=False)
    
    # Plotting
    print("Plotting stability curves...")
    
    # Melt for seaborn
    df_melt = final_df.melt(id_vars=['epoch', 'run'], 
                            value_vars=['recon_loss', 'disc_loss', 'gen_loss'],
                            var_name='Loss Type', value_name='Value')
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_melt, x='epoch', y='Value', hue='Loss Type', ci='sd')
    plt.title(f'AAE Training Stability ({N_RUNS} runs, Tau={TAU})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 0.1) # Zoom in for checking convergence if losses are small
    plt.savefig("results/stability/loss_curves.png")
    
    # Specific Zoom on Reconstruction
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=final_df, x='epoch', y='recon_loss', hue='run', palette="tab10", alpha=0.6)
    plt.title('Reconstruction Loss Stability')
    plt.yscale('log')
    plt.savefig("results/stability/recon_loss_zoom.png")

    print("\nanalysis complete. Results saved to results/stability/")

if __name__ == "__main__":
    main()
