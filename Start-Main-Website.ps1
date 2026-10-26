# NERVA Stress Intelligence System - Main Website Launcher (Primary System)
$ErrorActionPreference = 'Stop'
$projectPath = $PSScriptRoot
$linuxPath = '/mnt/' + $projectPath.Substring(0,1).ToLowerInvariant() + $projectPath.Substring(2).Replace('\','/')
$port = if ($env:PORT) { $env:PORT } else { '8001' }

Write-Host "=========================================================="
Write-Host "  NERVA Stress Detection System — MAIN WEBSITE (REAL)"
Write-Host "  Dataset: 10,000 AI-Generated Profile Dataset"
Write-Host "  Architecture: Six High-Accuracy Models (~87% - 91%)"
Write-Host "  URL: http://localhost:$port/#predictions"
Write-Host "=========================================================="

& wsl.exe -d Ubuntu -u nobody --exec env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 /opt/stress-venv/bin/python -B "$linuxPath/tools/start_training_preview.py" --port $port
if ($LASTEXITCODE -ne 0) { 
    throw "Main website could not start. Please verify WSL environment." 
}
