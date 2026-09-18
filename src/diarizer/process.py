"""Run Whisper STT + pyannote diarization on an existing audio file."""

from __future__ import annotations

import argparse
from pathlib import Path

from diarizer.config import load_llm_config
from diarizer.diarize import AUDIO_CANDIDATES
from diarizer.summarize import summarize_meeting_dir
from diarizer.voice_to_subtitles import (
    DEFAULT_LANGUAGE,
    DEFAULT_WHISPER_MODEL,
    run_diarization,
    save_segments_json,
    save_srt,
    save_timestamped_txt,
    save_txt,
    save_vtt,
    transcribe,
)


def resolve_audio(path: Path) -> Path:
    path = path.expanduser().resolve()

    if path.is_file():
        return path

    if path.is_dir():
        for name in AUDIO_CANDIDATES:
            candidate = path / name
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"No audio found in {path}. "
            f"Looked for: {', '.join(AUDIO_CANDIDATES)}"
        )

    raise FileNotFoundError(path)


def process_audio(
    audio_file: Path,
    *,
    language: str = DEFAULT_LANGUAGE,
    whisper_model: str = DEFAULT_WHISPER_MODEL,
    diarize: bool = True,
    summarize: bool | None = None,
) -> Path:
    """Transcribe (and optionally diarize/summarize) an existing recording.

    Writes artifacts next to the audio file. Returns the meeting folder.
    """
    audio_file = resolve_audio(audio_file)
    output_dir = audio_file.parent

    txt_file = output_dir / "meeting.txt"
    timestamped_file = output_dir / "meeting_timestamps.txt"
    segments_file = output_dir / "meeting_segments.json"
    srt_file = output_dir / "meeting.srt"
    vtt_file = output_dir / "meeting.vtt"
    speakers_file = output_dir / "meeting_speakers.txt"
    summary_file = output_dir / "meeting_summary.md"

    print()
    print("=" * 60)
    print("PROCESS EXISTING AUDIO (STT + DIARIZATION)")
    print("=" * 60)
    print(f"Audio:  {audio_file}")
    print(f"Folder: {output_dir}")

    whisper_result = transcribe(
        audio_file,
        model_name=whisper_model,
        language=language,
    )

    save_txt(whisper_result, txt_file)
    save_timestamped_txt(whisper_result, timestamped_file)
    save_segments_json(whisper_result, segments_file)
    save_srt(whisper_result, srt_file)
    save_vtt(whisper_result, vtt_file)

    print()
    print("=" * 60)
    print("STANDARD TRANSCRIPTION DONE")
    print("=" * 60)

    if diarize:
        try:
            run_diarization(audio_file, segments_file, speakers_file)
        except Exception as exc:
            print()
            print("=" * 60)
            print("DIARIZATION FAILED")
            print("=" * 60)
            print(type(exc).__name__)
            print(exc)
            print("Обычная транскрипция УЖЕ сохранена.")

    do_summarize = summarize
    if do_summarize is None:
        try:
            do_summarize = load_llm_config().summarize
        except Exception:
            do_summarize = False

    if do_summarize:
        try:
            summarize_meeting_dir(output_dir)
        except Exception as exc:
            print()
            print("=" * 60)
            print("SUMMARIZATION FAILED")
            print("=" * 60)
            print(type(exc).__name__)
            print(exc)

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Folder:       {output_dir}")
    print(f"Audio:        {audio_file}")
    print(f"Transcript:   {txt_file}")
    print(f"Timed TXT:    {timestamped_file}")
    print(f"Segments:     {segments_file}")
    print(f"SRT:          {srt_file}")
    print(f"VTT:          {vtt_file}")
    if speakers_file.exists():
        print(f"Speakers:     {speakers_file}")
    if summary_file.exists():
        print(f"Summary:      {summary_file}")

    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Whisper STT (+ optional diarization/summary) "
            "on an existing meeting.wav or meetings/<timestamp>/ folder."
        )
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Audio file or meeting folder containing meeting.wav",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help=f"Whisper language (default: {DEFAULT_LANGUAGE})",
    )
    parser.add_argument(
        "--whisper-model",
        default=DEFAULT_WHISPER_MODEL,
        help=f"Whisper model name (default: {DEFAULT_WHISPER_MODEL})",
    )
    parser.add_argument(
        "--diarize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run pyannote diarization after STT (default: yes)",
    )
    parser.add_argument(
        "--summarize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Run local llama-cli summary "
            "(default: [llm].summarize in config.toml)"
        ),
    )
    args = parser.parse_args(argv)

    try:
        process_audio(
            args.path,
            language=args.language,
            whisper_model=args.whisper_model,
            diarize=args.diarize,
            summarize=args.summarize,
        )
    except Exception as exc:
        print()
        print("=" * 60)
        print("PROCESS ERROR")
        print("=" * 60)
        print(type(exc).__name__)
        print(exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
