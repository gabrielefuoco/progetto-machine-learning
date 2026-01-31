import os
import subprocess
import sys

def ensure_directories():
    """Crea le directory necessarie se non esistono, per evitare crash degli script originali."""
    dirs = ["latent_data_sets", "results", "models", "data", "results/consolidated"]
    for d in dirs:
        if not os.path.exists(d):
            print(f"[SETUP] Creazione directory: {d}")
            os.makedirs(d)

def download_dataset():
    """Scarica il dataset se non presente."""
    url = 'https://raw.githubusercontent.com/GitiHubi/deepAI/master/data/fraud_dataset_v2.csv'
    dest = 'data/fraud_dataset_v2.csv'
    if not os.path.exists(dest):
        print(f"[SETUP] Download dataset da {url} a {dest}...")
        import urllib.request
        try:
            urllib.request.urlretrieve(url, dest)
            print("[SETUP] Download completato.")
        except Exception as e:
            print(f"[ERROR] Download fallito: {e}")

def run_script(script_name):
    """Esegue uno script python come sottoprocesso per garantire isolamento."""
    print(f"\n{'='*20} ESECUZIONE {script_name} {'='*20}")
    try:
        # Usa legacy_runner.py per patchare pandas prima di eseguire lo script target
        runner_script = os.path.join("project_extension", "legacy_runner.py")
        cmd = [sys.executable, runner_script, script_name]
        
        result = subprocess.run(cmd, check=True)
        print(f"[SUCCESS] {script_name} completato.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Errore durante l'esecuzione di {script_name}: {e}")
        # Se fallisce generateDatasets, gli altri falliranno comunque.
        if script_name == "generateDatasets.py":
             sys.exit(1)

def main():
    print("--- INIZIO FASE 2: RIPRODUCIBILITÀ (BASELINE) ---")
    
    # 1. Preparazione ambiente
    ensure_directories()
    download_dataset()
    
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
    print("Controlla la cartella 'results' per i grafici e i file di testo.")

if __name__ == "__main__":
    main()
