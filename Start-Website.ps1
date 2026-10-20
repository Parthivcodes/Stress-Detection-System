# Verified expanded version; existing data, models and previews stay available.
$ErrorActionPreference = 'Stop'
Write-Host 'Starting expanded BRFSS dashboard at http://127.0.0.1:8004/#predictions'
& wsl.exe -d Ubuntu -u nobody --cd $PSScriptRoot --exec env PORT=8004 BRFSS_MODEL_VERSION=brfss2022_expanded /opt/stress-venv/bin/python -u server.py
if ($LASTEXITCODE -ne 0) { throw 'Expanded preview could not start. Check model verification gates and the WSL environment.' }
