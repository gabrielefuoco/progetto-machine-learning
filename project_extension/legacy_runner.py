import sys
import os
import pandas as pd

def patch_pandas():
    """
    Monkeypatch pandas.get_dummies to return int instead of bool (default in pandas 2.0+).
    This fixes 'TypeError: can't convert np.ndarray of type numpy.object_' in PyTorch
    when converting mixed bool/float dataframes.
    """
    original_get_dummies = pd.get_dummies

    def patched_get_dummies(*args, **kwargs):
        if 'dtype' not in kwargs:
            kwargs['dtype'] = int
        return original_get_dummies(*args, **kwargs)
    
    pd.get_dummies = patched_get_dummies
    print("[LEGACY RUNNER] Pandas patched: get_dummies defaults to dtype=int.")

def patch_scipy():
    """
    Monkeypatch scipy.interp to alias numpy.interp.
    'from scipy import interp' fails in SciPy 1.10+ as it was removed.
    """
    import scipy
    import numpy as np
    if not hasattr(scipy, 'interp'):
        scipy.interp = np.interp
        print("[LEGACY RUNNER] Scipy patched: scipy.interp aliased to numpy.interp.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python legacy_runner.py <script_to_run>")
        sys.exit(1)
    
    script_path = sys.argv[1]
    
    # Setup environment
    patch_pandas()
    patch_scipy()

    
    # Prepare globals for exec
    script_dir = os.path.dirname(os.path.abspath(script_path))
    # Add script dir to sys.path if not there (though usually we run from root)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        
    print(f"[LEGACY RUNNER] Executing {script_path}...")
    
    with open(script_path, 'r') as f:
        code = f.read()
    
    # Eseguiamo lo script nel contesto globale corrente
    # Simula 'python script.py' ma con l'ambiente patchato
    global_namespace = globals().copy()
    global_namespace['__file__'] = script_path
    global_namespace['__name__'] = '__main__'
    
    try:
        exec(code, global_namespace)
    except Exception as e:
        print(f"[LEGACY RUNNER] Error executing {script_path}: {e}")
        # Re-raise per far fallire il processo chiamante
        raise e

if __name__ == "__main__":
    main()
