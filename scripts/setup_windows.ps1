#Requires -Version 5.1
param(
    [switch]$DownloadModels,
    [switch]$Help
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if ($Help) {
    Write-Host @"
Usage: .\scripts\setup_windows.ps1 [-DownloadModels]

  Syncs the uv project (including Windows loopback extra), checks ffmpeg,
  and optionally downloads the pyannote model (needs HF_TOKEN once).
"@
    exit 0
}

Write-Host "==> Checking uv"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv not found. Install: https://docs.astral.sh/uv/getting-started/installation/"
}

Write-Host "==> Checking ffmpeg"
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Error "ffmpeg not found in PATH. Install ffmpeg and re-run."
} else {
    Write-Host "ffmpeg OK: $((Get-Command ffmpeg).Source)"
}

Write-Host "==> uv sync --extra win"
uv sync --extra win

Write-Host ""
Write-Host "==> Checking WASAPI loopback devices"
uv run python -c @"
from diarizer.audio import list_input_devices
devs = [d for d in list_input_devices() if d.is_loopback]
if not devs:
    raise SystemExit('No WASAPI loopback devices found')
print('Found:')
for d in devs:
    print(' ', d.label())
"@

if ($DownloadModels) {
    Write-Host ""
    Write-Host "==> Downloading pyannote model (one-time, needs HF_TOKEN)"
    if (-not $env:HF_TOKEN -and -not $env:HUGGINGFACE_HUB_TOKEN) {
        Write-Error "Set HF_TOKEN before -DownloadModels"
    }
    uv run diarizer-download-models
}

Write-Host ""
Write-Host "Setup finished."
Write-Host "Record a meeting:  uv run diarizer"
Write-Host "List devices:      uv run diarizer --list-devices"
