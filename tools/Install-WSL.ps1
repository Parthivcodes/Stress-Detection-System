$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
# Installs only the supported Linux runtime; does not restart Windows.
$ErrorActionPreference = 'Stop'
$logPath = Join-Path $projectRoot 'output/wsl-install.log'
Start-Transcript -Path $logPath -Force
try {
    & "$env:WINDIR\System32\wsl.exe" --install -d Ubuntu --no-launch
    Write-Output "WSL installer exit code: $LASTEXITCODE"
    & "$env:WINDIR\System32\wsl.exe" --status
} finally {
    Stop-Transcript
}
