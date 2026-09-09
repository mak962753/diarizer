"""One-time download of gated pyannote weights into models/."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download pyannote speaker-diarization model once "
            "using HF_TOKEN, then use it locally forever."
        )
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token (default: HF_TOKEN env)",
    )
    args = parser.parse_args(argv)

    from diarizer.models import download_pyannote_model

    try:
        path = download_pyannote_model(token=args.token)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
