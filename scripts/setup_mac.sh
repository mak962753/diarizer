#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DOWNLOAD_MODELS=0
OPEN_MIDI=0

usage() {
  cat <<'EOF'
Usage: ./scripts/setup_mac.sh [--download-models] [--open-midi]

  Installs/syncs the uv project, checks ffmpeg + BlackHole, and
  optionally downloads the pyannote model (needs HF_TOKEN once).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --download-models) DOWNLOAD_MODELS=1; shift ;;
    --open-midi) OPEN_MIDI=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

echo "==> Checking uv"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

echo "==> Checking Homebrew / ffmpeg"
if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew not found. Install from https://brew.sh then re-run."
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg missing; installing via brew..."
  brew install ffmpeg
else
  echo "ffmpeg OK: $(command -v ffmpeg)"
fi

echo "==> uv sync"
uv sync

echo
echo "==> BlackHole / Multi-Output checklist"
cat <<'EOF'
1) Install BlackHole 2ch:
     brew install blackhole-2ch
   (or download from https://existential.audio/blackhole/)

2) Open Audio MIDI Setup and create a Multi-Output Device that includes:
     - your speakers/headphones
     - BlackHole 2ch

3) Set macOS (or Zoom/Teams/browser) output to that Multi-Output Device
   so you can hear audio while it is also routed into BlackHole.

4) Verify BlackHole appears as an input:
     uv run diarizer --list-devices
EOF

if [[ "$OPEN_MIDI" -eq 1 ]]; then
  open -a "Audio MIDI Setup" || true
fi

echo
echo "==> Checking for BlackHole in sounddevice"
if uv run python -c '
from diarizer.audio import list_input_devices
devs = list_input_devices()
hits = [d for d in devs if "blackhole" in d.name.lower()]
if not hits:
    raise SystemExit(2)
print("Found:", ", ".join(d.label() for d in hits))
'; then
  echo "BlackHole device detected."
else
  echo "WARNING: BlackHole not visible yet. Finish the checklist above, then re-run."
fi

if [[ "$DOWNLOAD_MODELS" -eq 1 ]]; then
  echo
  echo "==> Downloading pyannote model (one-time, needs HF_TOKEN)"
  if [[ -z "${HF_TOKEN:-}${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    echo "Set HF_TOKEN before --download-models"
    exit 1
  fi
  uv run diarizer-download-models
fi

echo
echo "Setup finished."
echo "Record a meeting:  uv run diarizer"
echo "List devices:      uv run diarizer --list-devices"
