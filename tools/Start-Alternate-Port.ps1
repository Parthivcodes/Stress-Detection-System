$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
# Separate, local-only preview; leaves existing ports untouched.
$ErrorActionPreference = 'Stop'
Write-Host 'Starting BRFSS dashboard at http://127.0.0.1:8003/#predictions'
& wsl.exe -d Ubuntu -u nobody --cd $projectRoot --exec env PORT=8003 /opt/stress-venv/bin/python -u server.py
if ($LASTEXITCODE -ne 0) { throw 'BRFSS server could not start. Check the trained models and WSL environment.' }
