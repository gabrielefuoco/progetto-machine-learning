# Design: GPU Acceleration with RAPIDS cuML

## Architecture Decisions

### 1. Library Replacement Strategy
We will perform a direct replacement of `sklearn` imports with `cuml` imports for valid estimators.
- `sklearn.svm.SVC` -> `cuml.svm.SVC`
- `sklearn.ensemble.RandomForestClassifier` -> `cuml.ensemble.RandomForestClassifier`
- `sklearn.naive_bayes.GaussianNB` -> `cuml.naive_bayes.GaussianNB`
- `sklearn.manifold.TSNE` -> `cuml.manifold.TSNE`

### 2. Parameter Compatibility
Most parameters are identical.
- `cuml.svm.SVC`: `probability=True` is supported but might have performance implications. We will keep it enabled as it is used in the current codebase.

### 3. Data Handling
`cuml` accepts NumPy arrays and automatically handles transfer to GPU memory (or accepts CuPy arrays/cudf DataFrames). Since the current data pipeline produces NumPy arrays (from `validation_v0.py` and `generateDatasets.py`), we can pass these directly to `cuml` estimators. `cuml` will handle the host-to-device copy internally.

### 4. Multiclass Support
The current code uses `OneVsRestClassifier`. `cuml` models often support multiclass natively or we can use `cuml.multiclass.OneVsRestClassifier`. We will attempt to wrap `cuml` estimators with `cuml.multiclass.OneVsRestClassifier` if available, or `sklearn.multiclass.OneVsRestClassifier` if `cuml` estimators are compatible (they usually adhere to sklearn API). *Optimization*: Using `cuml`'s OVR is preferred for full GPU pipeline.

## Risks
- **Dependency**: The project will deeply depend on `cuml` being installed. Only users with NVIDIA GPUs can run these specific scripts.
- **Windows Support**: RAPIDS has historically been Linux-only (WSL on Windows). If the user is on native Windows without WSL, `cuml` might not be available or require specific older versions/builds. *Assamption*: user requested this so they have the means to run it.
