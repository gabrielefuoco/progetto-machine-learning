# Design: Project Analysis Extension

## problem
The current repository mixes source code with execution scripts, and the original "generation" script relies on downloading pre-trained models rather than local training, hindering true reproducibility and experimentation. Furthermore, there is no baseline (Standard AE) to justify the added complexity of the AAE.

## Solution Architecture
We will create a separate, isolated package `project_extension/` to house all new logic. **No files in the root directory will be modified.**

### Directory Structure
```
root/
  project_extension/          # <--- ALL NEW CODE HERE
    __init__.py
    reproduce_baseline.py       
    train_standard_ae.py
    evaluate_comparative.py
    train_aae_stability.py
    train_imbalance.py
    results_stability/
  (original files ignored)
```

### Key Decisions
1.  **Wrapper Strategy**: `reproduce_baseline.py` will use `subprocess` to run original scripts. This ensures that the original global state (if any) doesn't interfere with our new scripts and handles directory creation automatically.
2.  **Shared Modules**: New scripts will import `Encoder`, `Decoder`, `Discriminator` from the root to ensure we are testing the *exact* same architectures, not copies. `sys.path.append` will be used to resolve these imports.
3.  **Unified Evaluation**: We will reuse the `MAUC` calculation logic to ensure metrics are directly comparable between the paper's results and our new experiments.
