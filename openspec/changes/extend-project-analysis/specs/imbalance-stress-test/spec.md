# Spec: Imbalance Stress Test

## ADDED Requirements

### Requirement: Imbalanced Training
The system MUST implement `train_imbalance.py` to train the AAE on a modified dataset where 50% of `global` and `local` anomaly samples are removed.

#### Scenario: Verify Data Subsampling
- Given the `train_imbalance.py` script
- When executed
- Then it should print the original counts of regular and anomaly classes.
- And it should print the reduced counts showing ~50% reduction in anomalies.
- And it should train the model on this reduced dataset.
- And it should save the latent space to `latent_data_sets/ldim10_imbalance50_basisLS.txt`.
