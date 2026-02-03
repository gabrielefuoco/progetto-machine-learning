# Regole di Progetto: ThesisFlow (Relazione Finale)

Questo file contiene le direttive persistenti per la stesura della tesi. **Consultare sempre prima di generare contenuti.**

## 1. Contesto del Progetto
*   **Oggetto**: Tesi sperimentale su Adversarial Autoencoders (AAE) per Fraud Detection.
*   **Obiettivo Estensione**: Dimostrare che un AE Standard (più semplice) performa meglio di un AAE (più complesso) su questo dataset, e validare la stabilità del training.
*   **Stack Tecnologico**: Python 3.9, RAPIDS cuML (GPU), Docker, PyTorch.

## 2. Linee Guida Linguistiche
*   **Lingua**: Italiano.
*   **Registro**: Accademico, Formale, Impersonale (evitare "io ho fatto", usare "si è osservato", "il modello mostra").
*   **Terminologia Tecnica**:
    *   Lasciare in **Inglese** i termini standard dell'AI: *Loss*, *Overfitting*, *One-Hot Encoding*, *Embeddings*, *Latent Space*, *Mode Collapse*, *Vanishing Gradient*, *Epoch*, *Batch*.
    *   Usare l'italiano per la struttura del discorso: *Addestramento* (Training può essere usato ma alternare), *Rete Neurale*, *Insieme di Validazione*.

### 4. Codice
*   **Minimalismo**: Includere SOLO righe di codice singole o blocchi di 2-3 righe se ASSOLUTAMENTE essenziali per spiegare un concetto (es. `pd.get_dummies(sparse=True)`).
*   **Niente copia-incolla**: Non inserire intere funzioni o classi. Descrivere la logica in pseudocodice o testo.

## 3. Fonti di Verità (Source of Truth)
*   **Per i Risultati**: `project_extension/recap.md` e `project_extension/relazione_tecnica_extension.md`.
*   **Per l'Ambiente**: `project_extension/relazione-modifica ambiente.md`.
*   **Per i Grafici**: Cartelle `results/stability/`, `results/imbalance/`, `tsneResults/`.

## 4. Struttura dei Riferimenti
*   Quando si cita codice, fare riferimento ai file specifici (es: `train_stability.py`).
*   Non inventare metriche. Usare solo quelle presenti nei log: MAUC, AUC, AP, Reconstruction Loss.

## 5. Formattazione
*   Usare il formato Markdown standard.
*   Per le immagini, usare la sintassi ThesisFlow: `![Caption](@/path/to/image.png)`.
*   Per le citazioni bibliografiche: `[@citationKey]`.

## 6. Workflow "Extremely Detailed"
*   Prima di scrivere un capitolo, leggere sempre i file sorgente mappati nel piano.
*   Non assumere che il codice faccia cose non documentate nel `recap.md`.
