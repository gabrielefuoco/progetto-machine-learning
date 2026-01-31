# Spec: Comparative Analysis

## ADDED Requirements

### Requirement: Standard Autoencoder Training
The system MUST implement `train_standard_ae.py` to train a non-adversarial Autoencoder using the exact same `Encoder` and `Decoder` architecture and data preprocessing as the original AAE.

#### Scenario: Train Standard AE
- Given the `train_standard_ae.py` script
- When executed
- Then it should train for a defined number of epochs (default 100).
- And it should save the generated latent space to `latent_data_sets/ldim10_standardAE_basisLS.txt`.
- And the output file format must match the tab-separated format required by evaluation scripts.

### Requirement: Comparative Evaluation
The system MUST implement `evaluate_comparative.py` to calculate the MAUC metric for the Standard AE latent space using the same classifiers (SVM, RF, NB) and 10-fold cross-validation strategy as the original paper.

#### Scenario: Evaluate MAUC
- Given the `latent_data_sets/ldim10_standardAE_basisLS.txt` file exists
- When `evaluate_comparative.py` is executed
- Then it should output MAUC scores for NB, RF, and SVM.
- And these scores should be comparable to the baseline results.
