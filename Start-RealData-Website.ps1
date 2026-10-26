$ErrorActionPreference='Stop'
Write-Host 'Real-data reference website: http://localhost:8000/#predictions'
& wsl.exe -d Ubuntu -u nobody --cd $PSScriptRoot --exec env PORT=8000 BRFSS_MODEL_VERSION=brfss2022_expanded PYTHONDONTWRITEBYTECODE=1 /opt/stress-venv/bin/python -B server.py
