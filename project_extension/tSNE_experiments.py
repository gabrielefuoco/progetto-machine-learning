import numpy as np
import pandas as pd
import os
from cuml.manifold import TSNE
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

def run_tsne_experiment(file_path, title_suffix):
    print(f"--- Processing {file_path} ---")
    
    if not os.path.exists(file_path):
        print(f"File {file_path} not found. Skipping.")
        return

    # 1. Fast Loading with Pandas
    print("Reading data...")
    try:
        df = pd.read_csv(file_path, sep='\t', header=None, engine='c')
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Last column is label
    # Use full dataset (approx 530k rows)
    X_final = df.iloc[:, :-1].values.astype(np.float32)
    Y_final = df.iloc[:, -1].values.astype(np.int32)
    
    n_total = len(X_final)
    print(f"Total points for t-SNE: {n_total} (Full Dataset)")

    # 3. Projecting with cuML
    # verbose=True will print optimization progress from the GPU kernel
    print("Projecting with cuML t-SNE (this might take a few minutes)...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, verbose=True)
    X_embedded = tsne.fit_transform(X_final)
    
    # 4. Plotting
    print("Plotting...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    # Masks for plotting
    mask_reg = (Y_final == 0)
    mask_glob = (Y_final == 1)
    mask_loc = (Y_final == 2)
    
    # Plot Regular (Alpha lower to see density)
    ax.scatter(X_embedded[mask_reg, 0], X_embedded[mask_reg, 1], 
               c='C0', marker="o", label='Regular', s=1, alpha=0.05)
    
    # Plot Global Outliers
    ax.scatter(X_embedded[mask_glob, 0], X_embedded[mask_glob, 1], 
               c='C1', marker="x", label='Global Outlier', s=20, alpha=0.8)
               
    # Plot Local Outliers
    ax.scatter(X_embedded[mask_loc, 0], X_embedded[mask_loc, 1], 
               c='C3', marker="x", label='Local Outlier', s=20, alpha=0.8)
    
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_title(f't-SNE Projection: {title_suffix}')
    ax.legend()
    
    # Save
    output_dir = "tsneResults"
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.basename(file_path).replace('.txt', '_tSNE.png')
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved to {output_path}\n")

if __name__ == "__main__":
    latent_dim = 10
    
    # List of datasets to process
    datasets = [
        (f"latent_data_sets/ldim{latent_dim}_tau5_basisLS.txt", "AAE (tau=5)"),
        (f"latent_data_sets/ldim{latent_dim}_tau10_basisLS.txt", "AAE (tau=10)"),
        (f"latent_data_sets/ldim{latent_dim}_tau20_basisLS.txt", "AAE (tau=20)"),
        (f"latent_data_sets/ldim{latent_dim}_standardAE_basisLS.txt", "Standard AE")
    ]
    
    for path, title in datasets:
        run_tsne_experiment(path, title)
