# Script per avviare il container con supporto GPU
# Monta la cartella corrente in /workspace

Write-Host "Avvio dell'ambiente 'fraud-detection-rapids' con supporto GPU..." -ForegroundColor Cyan

# Controlla se Docker è in esecuzione
docker info > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Errore: Docker non sembra essere in esecuzione." -ForegroundColor Red
    Pause
    exit
}

# Avvia il container
# --gpus all: abilita accesso alla GPU (richiede driver NVIDIA e WSL2 configurato)
# --rm: rimuove il container quando esci
# -it: interattivo (ti da una shell)
# -v: monta la cartella corrente
docker run --gpus all --rm -it `
    -v ${PWD}:/workspace `
    fraud-detection-rapids

