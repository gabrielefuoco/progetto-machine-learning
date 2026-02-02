# Script per costruire l'immagine Docker personalizzata
# Esegui questo script UNA SOLA VOLTA (o quando modifichi il Dockerfile)

Write-Host "Costruzione dell'immagine Docker 'fraud-detection-rapids'..." -ForegroundColor Cyan

docker build -t fraud-detection-rapids .

if ($?) {
    Write-Host "`nImmagine costruita con successo!" -ForegroundColor Green
    Write-Host "Ora puoi avviare l'ambiente con .\run_container.ps1" -ForegroundColor Yellow
} else {
    Write-Host "`nErrore durante la costruzione dell'immagine." -ForegroundColor Red
}

Pause
