"""Local mic + system audio recording, Whisper transcription, and pyannote diarization."""

__version__ = "0.1.0"


def main() -> None:
    from diarizer.voice_to_subtitles import main as _main

    _main()
