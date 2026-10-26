# Explicit, recoverable cleanup. Does not touch datasets, models or dependencies.
[CmdletBinding()]
param([switch]$Restore)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$archiveRoot = Join-Path $projectRoot '.cleanup/2026-10-20'
$targets = @(
    'sechenov_ui/drafts',
    'sechenov_ui/app.js',
    'sechenov_ui/style.css',
    'sechenov_ui/experience.js',
    'sechenov_ui/experience.css',
    'sechenov_ui/strain.js',
    'sechenov_ui/strain.css',
    'sechenov_ui/nerva.js',
    'sechenov_ui/nerva.css',
    '__pycache__',
    'src/__pycache__',
    'wellness-dashboard/tsconfig.tsbuildinfo'
)
$moves = @()
foreach ($relative in $targets) {
    $currentRelative = $relative.Replace('sechenov_ui/', 'frontend/').Replace('wellness-dashboard/', 'frontend/dashboard/')
    $original = [IO.Path]::GetFullPath((Join-Path $projectRoot $currentRelative))
    $archived = [IO.Path]::GetFullPath((Join-Path $archiveRoot $relative))
    if (-not $original.StartsWith($projectRoot + [IO.Path]::DirectorySeparatorChar) -or
        -not $archived.StartsWith($archiveRoot + [IO.Path]::DirectorySeparatorChar)) {
        throw "Unsafe path: $relative"
    }
    $source = if ($Restore) { $archived } else { $original }
    $destination = if ($Restore) { $original } else { $archived }
    if (-not (Test-Path -LiteralPath $source)) { continue }
    if (Test-Path -LiteralPath $destination) { throw "Will not overwrite $destination" }
    $item = Get-Item -LiteralPath $source
    if ($item.LinkType) { throw "Will not move a filesystem link: $source" }
    $moves += [pscustomobject]@{Source=$source; Destination=$destination; Relative=$relative}
}
# Verify all destinations before making the first move.
foreach ($move in $moves) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $move.Destination) -Force | Out-Null
    Move-Item -LiteralPath $move.Source -Destination $move.Destination
    Write-Output "$($move.Source) -> $($move.Destination)"
}
