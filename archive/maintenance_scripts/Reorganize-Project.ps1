# One-time, lossless relocation with preflight checks and a SHA-256 inventory.
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$pairs = @(
    @('data/Sleep_health_and_lifestyle_dataset.csv','data/original_sleep_dataset/sleep_lifestyle.csv'),
    @('data/brfss2022','data/baseline_survey_dataset'),
    @('data/brfss2022_expanded','data/expanded_survey_dataset'),
    @('models/brfss2022_expanded','models/current_models'),
    @('models/brfss_active.json','models/active_model.json'),
    @('backups/models/legacy','backups/original_models'),
    @('backups/models/brfss2022','backups/baseline_models'),
    @('backups/models/brfss2022_tuned','backups/tuned_models'),
    @('backups/models/brfss_active.before-expanded.json','backups/previous_active_model.json'),
    @('backups/models/README.md','backups/README.md'),
    @('sechenov_ui','frontend'),
    @('wellness-dashboard','frontend/dashboard'),
    @('brfss_pipeline.py','training/train_baseline_models.py'),
    @('expand_brfss.py','training/train_expanded_models.py'),
    @('tune_brfss.py','training/tune_baseline_models.py'),
    @('train_pipeline.py','training/train_original_models.py'),
    @('load_data.py','training/download_original_dataset.py'),
    @('app.py','legacy/original_dashboard.py'),
    @('streamlit_app.py','legacy/streamlit_app.py'),
    @('Start-BRFSS-Expanded.ps1','Start-Website.ps1'),
    @('Start-BRFSS.ps1','tools/Start-Alternate-Port.ps1'),
    @('Start-Dashboard.ps1','tools/Start-AutoDetect.ps1'),
    @('Start-Dashboard-WSL.ps1','tools/Start-WSL.ps1'),
    @('Install-ProjectWSL.ps1','tools/Install-WSL.ps1'),
    @('run_local.py','tools/run_windows_fallback.py'),
    @('verify_project.py','tests/verify_original_models.py'),
    @('validate_submission.py','tests/validate_original_dataset.py'),
    @('PROJECT_REPORT.md','docs/ORIGINAL_PROJECT_REPORT.md')
)
$manifestPath = Join-Path $root 'reports/project_cleanup/reorganization.json'
if (Test-Path -LiteralPath $manifestPath) { throw 'Relocation manifest already exists; do not rerun.' }
$inventory = @()
foreach ($pair in $pairs) {
    $source = [IO.Path]::GetFullPath((Join-Path $root $pair[0]))
    $dest = [IO.Path]::GetFullPath((Join-Path $root $pair[1]))
    if (-not $source.StartsWith($root+'\') -or -not $dest.StartsWith($root+'\')) { throw 'Out-of-project path' }
    if (-not (Test-Path -LiteralPath $source) -or (Test-Path -LiteralPath $dest)) { throw "Missing source or occupied target: $source" }
    $item = Get-Item -LiteralPath $source
    $files = if ($item.PSIsContainer) { Get-ChildItem -LiteralPath $source -Recurse -File } else { @($item) }
    foreach ($file in $files) {
        if ($file.FullName -match '\\(node_modules|\.npm-cache|__pycache__)\\') { continue }
        $suffix = $file.FullName.Substring($source.Length).Replace('\','/')
        $inventory += [pscustomobject]@{before=$pair[0]+$suffix; after=$pair[1]+$suffix; sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash; bytes=$file.Length}
    }
}
# Persist the source inventory before the first move.
$inventory | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $manifestPath -Encoding utf8
foreach ($pair in $pairs) {
    $source = [IO.Path]::GetFullPath((Join-Path $root $pair[0]))
    $dest = [IO.Path]::GetFullPath((Join-Path $root $pair[1]))
    New-Item -ItemType Directory -Path (Split-Path -Parent $dest) -Force | Out-Null
    Move-Item -LiteralPath $source -Destination $dest
    Write-Output "$($pair[0]) -> $($pair[1])"
}
foreach ($entry in $inventory) {
    if ((Get-FileHash -LiteralPath (Join-Path $root $entry.after) -Algorithm SHA256).Hash -ne $entry.sha256) {
        throw "Relocation checksum mismatch: $($entry.after)"
    }
}
Write-Output "All $($inventory.Count) inventoried files relocated byte-for-byte. Dependency directories moved intact, but not individually hashed."
