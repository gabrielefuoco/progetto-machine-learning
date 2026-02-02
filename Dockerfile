# Usa l'immagine ufficiale RAPIDS come base
# Versione specificata: RAPIDS 23.10, CUDA 11.8, Python 3.9 (Ubuntu 22.04 base)
FROM rapidsai/base:23.10-cuda11.8-py3.9

# Metadati maintainer (opzionale)
LABEL description="Environment for Fraud Detection AAE with RAPIDS cuML"

# Installa dipendenze Python aggiuntive non incluse nell'immagine base
# L'immagine base ha già: numpy, pandas, scikit-learn, cuml, cudf
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

RUN pip install --no-cache-dir \
    seaborn \
    matplotlib \
    ipython

# Imposta la directory di lavoro
WORKDIR /workspace

# Comando di default all'avvio (shell interattiva)
CMD ["/bin/bash"]
