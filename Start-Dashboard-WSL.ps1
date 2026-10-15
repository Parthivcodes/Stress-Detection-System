# Use the isolated Linux inference environment. No Windows Python DLLs are loaded.
$ErrorActionPreference = 'Stop'
Write-Host 'Starting the six-model dashboard at http://localhost:8000'
& wsl.exe -d Ubuntu -u nobody --cd $PSScriptRoot --exec /opt/stress-venv/bin/python -u server.py
if ($LASTEXITCODE -ne 0) {
    throw 'The backend could not start. Check Ubuntu and /opt/stress-venv first.'
}
