# Spec: Reproducibility

## ADDED Requirements

### Requirement: Independent Environment Setup
The system MUST provide a dedicated directory `project_extension/` treated as a python package to isolate new experiments from original code.

#### Scenario: Verify Setup
- Given the `project_extension` directory does not exist
- When the setup is run
- Then the directory `project_extension/` and `project_extension/__init__.py` should be created.

### Requirement: Baseline Reproduction
The system MUST provide a script `reproduce_baseline.py` that orchestrates the execution of the original project scripts (`generateDatasets.py`, `MAUC_Exp.py`, `MAUC_rawExp.py`) while ensuring all necessary output directories exist to prevent crashes.

#### Scenario: Run Baseline
- Given the original scripts exist in the root
- When `python project_extension/reproduce_baseline.py` is executed
- Then it should create `latent_data_sets`, `results`, `models`, `data` directories if missing.
- And it should execute `generateDatasets.py` successfully.
- And it should execute `MAUC_Exp.py` successfully.
- And it should execute `MAUC_rawExp.py` successfully (if present).
