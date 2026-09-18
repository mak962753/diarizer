from pathlib import Path

# Package root: .../src/diarizer
PACKAGE_DIR = Path(__file__).resolve().parent
# Repo root: .../ (parent of src/)
REPO_ROOT = PACKAGE_DIR.parent.parent

OUTPUT_ROOT = REPO_ROOT / "meetings"
MODELS_DIR = REPO_ROOT / "models"
CONFIG_PATH = REPO_ROOT / "config.toml"

PYANNOTE_REPO_ID = "pyannote/speaker-diarization-community-1"
PYANNOTE_LOCAL_DIR = MODELS_DIR / "pyannote-speaker-diarization-community-1"
