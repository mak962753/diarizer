"""Record mic + system audio, mix, transcribe, diarize."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import whisper

from diarizer.audio import (
    auto_mic_device,
    auto_system_device,
    create_recorder,
    list_input_devices,
    print_devices,
    resolve_device,
)
from diarizer.diarize import run_diarization_files
from diarizer.paths import OUTPUT_ROOT

MIC_VOLUME = 1.0
SYSTEM_VOLUME = 1.0
DEFAULT_WHISPER_MODEL = "small"
DEFAULT_LANGUAGE = "ru"


def check_ffmpeg() -> None:
    result = subprocess.run(
        ["ffmpeg", "-version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise RuntimeError("ffmpeg not found in PATH")


def mix_audio(
    mic_file: Path,
    system_file: Path,
    output_file: Path,
) -> None:
    print()
    print("=" * 60)
    print("MIXING AUDIO")
    print("=" * 60)

    filter_complex = (
        f"[0:a]aresample=48000,aformat=channel_layouts=mono,"
        f"volume={MIC_VOLUME}[mic];"
        f"[1:a]aresample=48000,pan=mono|c0=0.5*c0+0.5*c1,"
        f"volume={SYSTEM_VOLUME}[system];"
        f"[mic][system]amix=inputs=2:duration=longest:normalize=0,"
        f"alimiter=limit=0.95[mixed]"
    )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(mic_file),
        "-i",
        str(system_file),
        "-filter_complex",
        filter_complex,
        "-map",
        "[mixed]",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output_file),
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("ffmpeg mixing failed")

    print(f"Mixed file saved: {output_file}")


def transcribe(
    audio_file: Path,
    *,
    model_name: str,
    language: str,
) -> dict:
    print()
    print("=" * 60)
    print("WHISPER TRANSCRIPTION")
    print("=" * 60)
    print(f"Loading Whisper model: {model_name}")

    model = whisper.load_model(model_name)
    print("Transcribing locally...")

    return model.transcribe(
        str(audio_file),
        language=language,
        task="transcribe",
        verbose=False,
        fp16=False,
    )


def srt_timestamp(seconds: float) -> str:
    ms = int(seconds * 1000)
    hours = ms // 3_600_000
    ms %= 3_600_000
    minutes = ms // 60_000
    ms %= 60_000
    secs = ms // 1000
    ms %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def vtt_timestamp(seconds: float) -> str:
    return srt_timestamp(seconds).replace(",", ".")


def simple_timestamp(seconds: float) -> str:
    seconds_i = int(seconds)
    h = seconds_i // 3600
    m = (seconds_i % 3600) // 60
    s = seconds_i % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def save_txt(result: dict, path: Path) -> None:
    path.write_text(result.get("text", "").strip(), encoding="utf-8")


def save_timestamped_txt(result: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for segment in result.get("segments", []):
            text = segment["text"].strip()
            if not text:
                continue
            f.write(
                f"[{simple_timestamp(segment['start'])}] {text}\n"
            )


def save_segments_json(result: dict, path: Path) -> None:
    segments = []
    for segment in result.get("segments", []):
        text = segment["text"].strip()
        if not text:
            continue
        segments.append(
            {
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": text,
            }
        )

    path.write_text(
        json.dumps({"segments": segments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_srt(result: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        index = 1
        for segment in result.get("segments", []):
            text = segment["text"].strip()
            if not text:
                continue
            f.write(f"{index}\n")
            f.write(
                f"{srt_timestamp(segment['start'])} --> "
                f"{srt_timestamp(segment['end'])}\n"
            )
            f.write(f"{text}\n\n")
            index += 1


def save_vtt(result: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for segment in result.get("segments", []):
            text = segment["text"].strip()
            if not text:
                continue
            f.write(
                f"{vtt_timestamp(segment['start'])} --> "
                f"{vtt_timestamp(segment['end'])}\n"
            )
            f.write(f"{text}\n\n")


def run_diarization(
    meeting_file: Path,
    segments_file: Path,
    speakers_file: Path,
) -> None:
    print()
    print("=" * 60)
    print("SPEAKER DIARIZATION")
    print("=" * 60)

    run_diarization_files(
        meeting_file,
        segments_file,
        speakers_file,
    )

    print(f"Speaker transcript saved: {speakers_file}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record microphone + system audio, mix with ffmpeg, "
            "transcribe with Whisper, diarize with local pyannote."
        )
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List input devices and exit",
    )
    parser.add_argument(
        "--mic",
        default=None,
        help="Mic device index or name substring (default: auto)",
    )
    parser.add_argument(
        "--system",
        default=None,
        help=(
            "System-audio device index or name substring "
            "(default: BlackHole on macOS, WASAPI loopback on Windows)"
        ),
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
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help=f"Output root directory (default: {OUTPUT_ROOT})",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_devices:
        print_devices()
        return

    print()
    print("=" * 60)
    print("MIC + SYSTEM AUDIO + WHISPER + DIARIZATION")
    print("=" * 60)

    check_ffmpeg()

    devices = list_input_devices()
    if args.mic is None:
        mic_device = auto_mic_device(devices)
    else:
        mic_device = resolve_device(
            args.mic,
            role="mic",
            candidates=[
                d
                for d in devices
                if not d.is_loopback
                and "blackhole" not in d.name.lower()
            ]
            or devices,
        )

    if args.system is None:
        system_device = auto_system_device(devices)
    else:
        system_device = resolve_device(
            args.system,
            role="system",
            candidates=devices,
        )

    print()
    print("MIC DEVICE:")
    print(f"  {mic_device.label()}")
    print()
    print("SYSTEM DEVICE:")
    print(f"  {system_device.label()}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.output_root) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    mic_file = output_dir / "mic.wav"
    system_file = output_dir / "system.wav"
    meeting_file = output_dir / "meeting.wav"
    txt_file = output_dir / "meeting.txt"
    timestamped_file = output_dir / "meeting_timestamps.txt"
    segments_file = output_dir / "meeting_segments.json"
    srt_file = output_dir / "meeting.srt"
    vtt_file = output_dir / "meeting.vtt"
    speakers_file = output_dir / "meeting_speakers.txt"

    mic = create_recorder(mic_device, mic_file, "MICROPHONE")
    system = create_recorder(
        system_device,
        system_file,
        "SYSTEM AUDIO",
    )

    mic_started = False
    system_started = False

    try:
        print()
        print("=" * 60)
        print("STARTING RECORDING")
        print("=" * 60)

        system.start()
        system_started = True
        mic.start()
        mic_started = True

        print()
        print("RECORDING...")
        print()
        print("YouTube / Teams / Zoom можно запускать.")
        print("Говори в микрофон.")
        print()
        print("Нажми ENTER для остановки.")
        print()
        input()

    finally:
        print()
        print("Stopping recording...")
        if mic_started:
            mic.stop()
        if system_started:
            system.stop()

    mix_audio(mic_file, system_file, meeting_file)

    whisper_result = transcribe(
        meeting_file,
        model_name=args.whisper_model,
        language=args.language,
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

    try:
        run_diarization(meeting_file, segments_file, speakers_file)
    except Exception as exc:
        print()
        print("=" * 60)
        print("DIARIZATION FAILED")
        print("=" * 60)
        print()
        print(type(exc).__name__)
        print(exc)
        print()
        print("Обычная транскрипция УЖЕ сохранена.")

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print()
    print(f"Folder:       {output_dir}")
    print(f"Mic:          {mic_file}")
    print(f"System:       {system_file}")
    print(f"Mixed:        {meeting_file}")
    print(f"Transcript:   {txt_file}")
    print(f"Timed TXT:    {timestamped_file}")
    print(f"Segments:     {segments_file}")
    print(f"SRT:          {srt_file}")
    print(f"VTT:          {vtt_file}")
    if speakers_file.exists():
        print(f"Speakers:     {speakers_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("Stopped by user.")
    except Exception as exc:
        print()
        print("=" * 60)
        print("ERROR")
        print("=" * 60)
        print()
        print(type(exc).__name__)
        print(exc)
        raise SystemExit(1)
