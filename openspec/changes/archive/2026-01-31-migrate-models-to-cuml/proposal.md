# Change: Migrate Models to RAPIDS cuML

## Why
The current efficiency of CPU-based model training (scikit-learn) is insufficient for the growing dataset sizes and complexity. Migrating to RAPIDS cuML will leverage the available NVIDIA GPU to significantly accelerate training and inference.

## What Changes
- Replace `scikit-learn` imports with `cuml` equivalents in `project_extension/evaluate_comparative.py`.
- Update legacy experiment scripts (`exp_imbMulticlass.py`, `experiment_crossClasf.py`, `MAUC_Exp.py`) to use `cuml` estimators.
- Update `tSNE_experiments.py` to use `cuml.manifold.TSNE` for faster visualization.
- **BREAKING**: Requires an NVIDIA GPU and `cuml` library installed; code will no longer run on CPU-only environments without fallback logic (which is not currently planned for this strict migration).

## Impact
- Affected specs: `acceleration`
- Affected code: Experiment scripts in root and `project_extension` folder.
