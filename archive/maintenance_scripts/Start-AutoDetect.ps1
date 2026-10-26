$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
# Launch from any working directory. Prefer a regular project Python environment.
Set-Location -LiteralPath $projectRoot
# Prefer the repaired WSL inference environment when it is provisioned.
if (Get-Command wsl.exe -ErrorAction SilentlyContinue) {
    & wsl.exe -d Ubuntu -u root --exec test -x /opt/stress-venv/bin/python 2>$null
    if ($LASTEXITCODE -eq 0) {
        & (Join-Path $PSScriptRoot 'Start-WSL.ps1')
        exit $LASTEXITCODE
    }
}
$projectPython = Join-Path $projectRoot '.venv/Scripts/python.exe'
$embeddedPython = 'C:/Program Files/PostgreSQL/18/pgAdmin 4/python/python.exe'
if (Test-Path -LiteralPath $projectPython) {
    & $projectPython server.py
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    python server.py
} elseif ((Test-Path -LiteralPath $embeddedPython) -and (Test-Path -LiteralPath '.runtime/compat/sklearn')) {
    & $embeddedPython tools/run_windows_fallback.py server.py
} else {
    Write-Error 'Install Python and requirements.txt as described in README.md.'
}
