"""Download and resolve local pyannote model weights."""

from __future__ import annotations

import os
from pathlib import Path

from diarizer.paths import MODELS_DIR, PYANNOTE_LOCAL_DIR, PYANNOTE_REPO_ID

HF_CACHE_DIR = MODELS_DIR / "hf-cache"
MARKER_FILE = PYANNOTE_LOCAL_DIR / ".download_complete"


def _apply_local_hf_cache() -> None:
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(MODELS_DIR / "hf-home")
    os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR)


def model_is_ready() -> bool:
    return MARKER_FILE.exists() and HF_CACHE_DIR.is_dir()


def resolve_pipeline_source() -> str:
    """Return repo id for Pipeline.from_pretrained (weights from local cache)."""
    if not model_is_ready():
        raise RuntimeError(
            f"Local pyannote model cache not found under {MODELS_DIR}.\n"
            "Run a one-time download:\n"
            "  export HF_TOKEN=...\n"
            "  uv run diarizer-download-models\n"
            "or: ./scripts/setup_mac.sh --download-models"
        )

    _apply_local_hf_cache()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return PYANNOTE_REPO_ID


def download_pyannote_model(
    *,
    token: str | None = None,
) -> Path:
    from huggingface_hub import snapshot_download
    from pyannote.audio import Pipeline

    hf_token = token or os.getenv("HF_TOKEN") or os.getenv(
        "HUGGINGFACE_HUB_TOKEN"
    )

    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) is required "
            "for the one-time model download. Create a token at "
            "https://huggingface.co/settings/tokens and accept "
            f"the license for {PYANNOTE_REPO_ID}."
        )

    _apply_local_hf_cache()
    PYANNOTE_LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading pipeline snapshot: {PYANNOTE_REPO_ID}")
    snapshot_download(
        repo_id=PYANNOTE_REPO_ID,
        token=hf_token,
        local_dir=str(PYANNOTE_LOCAL_DIR),
    )

    print("Materializing nested pyannote weights into local HF cache...")
    # Instantiating once pulls dependency models into HF_HUB_CACHE.
    pipeline = Pipeline.from_pretrained(
        PYANNOTE_REPO_ID,
        token=hf_token,
        cache_dir=str(HF_CACHE_DIR),
    )
    del pipeline

    MARKER_FILE.write_text("ok\n", encoding="utf-8")
    print(f"Model ready. Cache: {HF_CACHE_DIR}")
    print("Normal runs do not need HF_TOKEN.")
    return PYANNOTE_LOCAL_DIR
