# Diarizer

Local meeting recorder: microphone + system audio → ffmpeg mix → Whisper transcript → pyannote speaker labels.

## Setup

### macOS

```bash
./scripts/setup_mac.sh
# optional one-time model download (needs HF_TOKEN):
./scripts/setup_mac.sh --download-models
```

Install **BlackHole 2ch**, create a Multi-Output Device (speakers + BlackHole), and set app/system output to that device. The setup script prints the checklist.

### Windows

```powershell
.\scripts\setup_windows.ps1
# optional:
.\scripts\setup_windows.ps1 -DownloadModels
```

Uses WASAPI loopback via `pyaudiowpatch` (`uv sync --extra win`).

## Usage

```bash
uv run diarizer --list-devices
uv run diarizer
uv run diarizer --mic "MacBook" --system BlackHole --language ru --whisper-model small
uv run diarizer --no-summarize
```

Outputs land under `meetings/<timestamp>/`.

### Summarize an existing transcript

Uses `llama-cli` (Homebrew `llama.cpp`) and `[llm]` settings in `config.toml`:

```bash
uv run diarizer-summarize meetings/2026-01-01_12-00-00/
uv run diarizer-summarize meetings/2026-01-01_12-00-00/meeting_speakers.txt
uv run diarizer-summarize path/to/transcript.txt -o summary.md
```

## Models

Whisper downloads into its usual cache on first use.

Pyannote is fetched once into `models/` (snapshot + `models/hf-cache/`). Accept the model license on Hugging Face, set `HF_TOKEN`, then:

```bash
uv run diarizer-download-models
```

Later runs load from that local cache and do **not** need `HF_TOKEN`.

## Notes

- On macOS, grant **Microphone** access to your terminal app if `--list-devices` shows no inputs.
- System audio on macOS requires BlackHole + a Multi-Output Device (see setup script checklist).
- Windows loopback uses the `win` extra: `uv sync --extra win`.