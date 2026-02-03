= Appendice A: Configurazione Ambiente
<appendice-a-configurazione-ambiente>
In questa appendice tecnico vengono forniti i dettagli per la
riproduzione dell'ambiente di sviluppo Dockerizzato ottimizzato per GPU.

=== Dockerfile Layering
<dockerfile-layering>
L'immagine è costruita su
`nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`. I layer principali
sono: 1. #strong[System Dependencies];: `build-essential`, `curl`,
`git`, `libgl1`. 2. #strong[Miniforge];: Installazione di Mamba per la
gestione pacchetti. 3. #strong[Environment];: Creazione dell'environment
`rapids-23.10` con: \* `rapids=23.10` \* `python=3.10` \*
`cudatoolkit=11.8` \* `pytorch-cuda=11.8`

=== Script di Avvio
<script-di-avvio>
Lo script PowerShell `run_container.ps1` automatizza il mount dei
volumi: \* Mappa la cartella corrente su `/workspace`. \* Abilita
l'accesso alla GPU tramite `--gpus all`. \* Espone la porta 8888 per
Jupyter Lab.

=== Requisiti Hardware
<requisiti-hardware>
- #strong[GPU];: NVIDIA Pascal o superiore (Compute Capability \>= 6.0).
  Testato su RTX 3060.
- #strong[RAM];: Minimo 16GB (consigliati 32GB per operazioni in memoria
  su dataset completi senza batching).
- #strong[Driver];: NVIDIA Driver 520+ (Windows 11 WSL2 backend
  supportato).
