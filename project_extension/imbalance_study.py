import sys
import os
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Encoder import Encoder
from Decoder import Decoder
from Discriminator import Discriminator

# Configuration
LATENT_DIM = 10
TAU = 10
EPOCHS = 30 # Reduced for speed, seeking trend
BATCH_SIZE = 2048
LR = 0.0001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FRAUD_RATIOS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10] # 0.1% to 10%

def sample_prior(batch_size, z_dim, n_gaussians, radius=0.8, sigma=0.01):
    # Vectorized implementation
    indices = np.random.randint(0, n_gaussians, size=batch_size)
    angles = 2 * np.pi * indices / n_gaussians
    cx = radius * np.cos(angles)
    cy = radius * np.sin(angles)
    z = np.random.normal(0, sigma, size=(batch_size, z_dim)).astype(np.float32)
    z[:, 0] += cx
    z[:, 1] += cy
    return torch.from_numpy(z).to(DEVICE)

def load_data_and_resample(fraud_ratio):
    """
    Loads data and resamples the *training* set to have specific fraud ratio.
    The Test set remains the same (realistic evaluation).
    Actually, usually anomaly detection assumes UNSUPERVISED training (no labels).
    
    If the dataset is fully unlabeled during training, changing the ratio simulates 
    "how rare the anomalies are in the wild".
    
    We will:
    1. Load full dataset.
    2. Split Train/Test.
    3. In Train set, subsample/oversample Frauds to match `fraud_ratio`.
       WAIT: Standard AAE is trained on EVERYTHING (Unsupervised).
       So we just adjust the ratio of the dataset fed to the model.
    """
    local_csv = 'data/fraud_dataset_v2.csv'
    if not os.path.exists(local_csv):
        url = 'https://raw.githubusercontent.com/GitiHubi/deepAI/master/data/fraud_dataset_v2.csv'
        df = pd.read_csv(url)
    else:
        df = pd.read_csv(local_csv)

    # Preprocessing
    categorical_attr = ['KTOSL', 'PRCTR', 'BSCHL', 'HKONT', 'BUKRS', 'WAERS']
    data_categ = pd.get_dummies(df[categorical_attr], sparse=True)

    numeric_attr = ['DMBTR', 'WRBTR']
    data_num = df[numeric_attr] + 1e-4
    data_num = data_num.apply(np.log)
    data_num = (data_num - data_num.min()) / (data_num.max() - data_num.min())

    # Labels
    labels = df['label'].copy()
    is_fraud = (labels == 'global') | (labels == 'local')
    
    # Sparse concat
    data_all = pd.concat([data_categ, data_num], axis=1)
    data_np = data_all.values.astype(np.float32)
    
    # Split Regular and Fraud
    regulars = data_np[~is_fraud]
    frauds = data_np[is_fraud]
    
    # Target total size (approx same as original to keep epochs comparable)
    total_size = len(data_np)
    n_frauds_target = int(total_size * fraud_ratio)
    n_regulars_target = total_size - n_frauds_target
    
    # Resample Regulars (if needed, usually we keep all regulars)
    # To strictly control ratio, we might need to drop regulars if we can't find enough frauds
    # But real scenario: Regulars are abundant. Frauds are the variable.
    # Let's keep ALL regulars and subsample/oversample Frauds.
    
    # Actually, simpler: Keep all Regulars. Adjust Frauds to be X% of total.
    # n_frauds / (n_regs + n_frauds) = ratio
    # n_frauds = ratio * n_regs + ratio * n_frauds
    # n_frauds * (1 - ratio) = ratio * n_regs
    # n_frauds = (ratio * n_regs) / (1 - ratio)
    
    n_regs = len(regulars)
    required_frauds = int((fraud_ratio * n_regs) / (1 - fraud_ratio))
    
    if required_frauds > len(frauds):
        # Oversample frauds
        indices = np.random.choice(len(frauds), required_frauds, replace=True)
        frauds_resampled = frauds[indices]
    else:
        # Subsample frauds
        indices = np.random.choice(len(frauds), required_frauds, replace=False)
        frauds_resampled = frauds[indices]
        
    data_resampled = np.vstack([regulars, frauds_resampled])
    np.random.shuffle(data_resampled)
    
    print(f"Ratio {fraud_ratio:.1%}: {len(frauds_resampled)} Frauds, {len(regulars)} Regulars")
    
    return data_resampled, data_all.shape[1], frauds, regulars # Return original frauds/regs for testing

def train_and_evaluate(fraud_ratio, dim):
    print(f"\n--- Testing Fraud Ratio: {fraud_ratio:.1%} ---")
    data_train, input_dim, test_frauds, test_regulars = load_data_and_resample(fraud_ratio)
    
    # Model Setup
    encoder = Encoder(input_size=input_dim, hidden_size=[256, 64, 16, 4, LATENT_DIM]).to(DEVICE)
    decoder = Decoder(output_size=input_dim, hidden_size=[LATENT_DIM, 4, 16, 64, 256]).to(DEVICE)
    discriminator = Discriminator(input_size=LATENT_DIM, hidden_size=[256, 16, 4], output_size=1).to(DEVICE)

    optim_enc = optim.Adam(encoder.parameters(), lr=LR)
    optim_dec = optim.Adam(decoder.parameters(), lr=LR)
    optim_disc = optim.Adam(discriminator.parameters(), lr=LR)
    optim_gen = optim.Adam(encoder.parameters(), lr=LR) 

    criterion_recon = nn.MSELoss()
    criterion_adv = nn.BCELoss()

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(data_train))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True) 

    encoder.train()
    decoder.train()
    discriminator.train()

    for epoch in range(EPOCHS):
        for (batch_data,) in dataloader:
            batch_data = batch_data.to(DEVICE)
            curr_bs = batch_data.size(0)
            
            # Reconstruction
            optim_enc.zero_grad()
            optim_dec.zero_grad()
            z = encoder(batch_data)
            recon = decoder(z)
            loss_recon = criterion_recon(recon, batch_data)
            loss_recon.backward()
            optim_enc.step()
            optim_dec.step()
            
            # Discriminator
            optim_disc.zero_grad()
            z_real = sample_prior(curr_bs, LATENT_DIM, TAU).to(DEVICE)
            d_real = discriminator(z_real)
            loss_d_real = criterion_adv(d_real, torch.ones(curr_bs, 1).to(DEVICE))
            
            z_fake = encoder(batch_data).detach()
            d_fake = discriminator(z_fake)
            loss_d_fake = criterion_adv(d_fake, torch.zeros(curr_bs, 1).to(DEVICE))
            
            loss_disc = loss_d_real + loss_d_fake
            loss_disc.backward()
            optim_disc.step()
            
            # Generator
            optim_gen.zero_grad()
            z_gen = encoder(batch_data)
            d_gen = discriminator(z_gen)
            loss_gen = criterion_adv(d_gen, torch.ones(curr_bs, 1).to(DEVICE))
            loss_gen.backward()
            optim_gen.step()
            
    # Evaluation
    print("Evaluating...")
    encoder.eval()
    decoder.eval()
    
    # Prepare Test Set (Original Regulars + Original Frauds)
    # We want to see if it can still detect ALL original frauds
    X_test_fraud = torch.from_numpy(test_frauds).to(DEVICE)
    X_test_reg = torch.from_numpy(test_regulars).to(DEVICE)
    
    # MAUC / Reconstruction Error scoring
    # AAE usually uses Reconstruction Error OR Discriminator Probability as Anomaly Score
    # Standard AE uses Reconstruction Error.
    # AAE Paper often uses Discriminator output? No, usually Reconstruction Error.
    # Let's use Reconstruction Error (MSE)
    
    with torch.no_grad():
        # Fraud Scores
        z_f = encoder(X_test_fraud)
        rec_f = decoder(z_f)
        score_f = torch.mean((X_test_fraud - rec_f)**2, dim=1).cpu().numpy()
        
        # Regular Scores
        # batched for memory
        score_r_list = []
        for i in range(0, len(test_regulars), BATCH_SIZE):
            batch = X_test_reg[i:i+BATCH_SIZE]
            z_r = encoder(batch)
            rec_r = decoder(z_r)
            s_r = torch.mean((batch - rec_r)**2, dim=1).cpu().numpy()
            score_r_list.append(s_r)
        score_r = np.concatenate(score_r_list)
        
    y_true = np.concatenate([np.ones(len(score_f)), np.zeros(len(score_r))])
    y_scores = np.concatenate([score_f, score_r])
    
    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    
    print(f"Result Ratio {fraud_ratio:.1%} -> AUC: {auc:.4f} | AP: {ap:.4f}")
    return auc, ap

def main():
    if not os.path.exists("results/imbalance"):
        os.makedirs("results/imbalance")
        
    results = []
    
    for ratio in FRAUD_RATIOS:
        auc, ap = train_and_evaluate(ratio, LATENT_DIM)
        results.append({'Ratio': ratio, 'AUC': auc, 'AP': ap})
        
    df_res = pd.DataFrame(results)
    df_res.to_csv("results/imbalance/imbalance_results.csv", index=False)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_res, x='Ratio', y='AUC', marker='o', label='AUC')
    sns.lineplot(data=df_res, x='Ratio', y='AP', marker='s', label='Average Precision')
    plt.xscale('log')
    plt.title('AAE Performance vs Fraud Ratio (Imbalance Stress Test)')
    plt.xlabel('Fraud Ratio (Log Scale)')
    plt.ylabel('Score')
    plt.savefig("results/imbalance/imbalance_plot.png")
    
if __name__ == "__main__":
    main()
