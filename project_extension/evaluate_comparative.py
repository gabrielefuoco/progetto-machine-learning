import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
# We use sklearn's OneVsRestClassifier because cuML's wrapper has issues with predict_proba
from sklearn.multiclass import OneVsRestClassifier
# from cuml.multiclass import OneVsRestClassifier
from cuml.ensemble import RandomForestClassifier
from cuml.svm import SVC
from cuml.naive_bayes import GaussianNB
from sklearn.preprocessing import MultiLabelBinarizer

# Importiamo MAUC originale
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from MAUC import MAUC
except ImportError:
    print("[ERROR] Impossibile importare MAUC.py. Assicurati che lo script sia eseguito dalla root o che sys.path sia corretto.")
    sys.exit(1)


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
    # File prodotto dallo step precedente - verifica esistenza
    target_file = "latent_data_sets/ldim10_standardAE_basisLS.txt"
    if os.path.exists(target_file):
        evaluate_latent_space(target_file)
    else:
        print(f"[WARNING] File {target_file} non trovato.")
        print("Esegui 'python project_extension/train_standard_ae.py' prima di questo script.")
