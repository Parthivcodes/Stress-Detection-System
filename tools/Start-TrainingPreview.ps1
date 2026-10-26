$ErrorActionPreference='Stop'
$projectPath=Split-Path $PSScriptRoot -Parent
$linuxPath='/mnt/'+$projectPath.Substring(0,1).ToLowerInvariant()+$projectPath.Substring(2).Replace('\','/')
& wsl.exe -d Ubuntu -u nobody --exec env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 /opt/stress-venv/bin/python -B "$linuxPath/tools/start_training_preview.py" --port 8001
