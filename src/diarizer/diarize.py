"""Speaker diarization using a locally cached pyannote pipeline."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import soundfile as sf
import torch
from pyannote.audio import Pipeline

from diarizer.models import HF_CACHE_DIR, resolve_pipeline_source


def simple_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def timestamp_to_seconds(value: str) -> float:
    parts = value.split(":")
    h = int(parts[0])
    m = int(parts[1])
    s = int(parts[2])
    return float(h * 3600 + m * 60 + s)


def load_transcript_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = data.get("segments", data)

    result = []
    for item in segments:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        result.append(
            {
                "start": float(item["start"]),
                "end": float(item["end"]),
                "text": text,
            }
        )
    return result


def load_transcript_txt(path: Path) -> list[dict]:
    segments: list[dict] = []
    pattern = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s*(.*)$")

    lines = path.read_text(encoding="utf-8").splitlines()

    for index, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        match = pattern.match(line)
        if not match:
            continue

        text = match.group(2).strip()
        start = timestamp_to_seconds(match.group(1))

        end = start + 5.0
        if index + 1 < len(lines):
            next_match = pattern.match(lines[index + 1].strip())
            if next_match:
                end = timestamp_to_seconds(next_match.group(1))

        segments.append(
            {
                "start": float(start),
                "end": float(end),
                "text": text,
            }
        )

    return segments


def load_transcript(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        return load_transcript_json(path)
    return load_transcript_txt(path)


def diarize(audio_file: Path) -> list[dict]:
    print()
    print("=" * 60)
    print("PYANNOTE DIARIZATION (local model)")
    print("=" * 60)

    model_id = resolve_pipeline_source()
    print(f"Loading pipeline: {model_id} (local cache)")

    print("Loading audio with soundfile...")
    audio_data, sample_rate = sf.read(
        str(audio_file),
        dtype="float32",
        always_2d=True,
    )

    waveform = torch.from_numpy(audio_data.T).contiguous()
    duration = waveform.shape[1] / sample_rate

    print(
        f"Audio: {waveform.shape[0]} channel(s), "
        f"{sample_rate} Hz, {duration:.1f} sec"
    )

    pipeline = Pipeline.from_pretrained(
        model_id,
        token=False,
        cache_dir=str(HF_CACHE_DIR),
    )

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # pyannote/MPS can be flaky; stay on CPU by default
        pass

    print("Detecting speakers...")
    output = pipeline(
        {
            "waveform": waveform,
            "sample_rate": int(sample_rate),
        }
    )

    diarization = getattr(
        output,
        "exclusive_speaker_diarization",
        None,
    )
    if diarization is None:
        diarization = output.speaker_diarization

    turns = []
    for turn, speaker in diarization:
        turns.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": speaker,
            }
        )

    print(f"Speaker turns: {len(turns)}")
    return turns


def overlap_duration(
    start1: float,
    end1: float,
    start2: float,
    end2: float,
) -> float:
    return max(0.0, min(end1, end2) - max(start1, start2))


def find_speaker(segment: dict, turns: list[dict]) -> str | None:
    best_speaker = None
    best_overlap = 0.0

    for turn in turns:
        overlap = overlap_duration(
            segment["start"],
            segment["end"],
            turn["start"],
            turn["end"],
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn["speaker"]

    return best_speaker


def create_mapping(segments: list[dict], turns: list[dict]) -> dict:
    mapping: dict[str, str] = {}
    counter = 1

    for segment in segments:
        speaker = find_speaker(segment, turns)
        if speaker is None:
            continue
        if speaker not in mapping:
            mapping[speaker] = f"Speaker {counter}"
            counter += 1

    return mapping


def save_result(
    segments: list[dict],
    turns: list[dict],
    output_file: Path,
) -> None:
    mapping = create_mapping(segments, turns)
    previous_speaker = None

    with output_file.open("w", encoding="utf-8") as f:
        for segment in segments:
            raw_speaker = find_speaker(segment, turns)
            if raw_speaker is None:
                speaker = "Unknown"
            else:
                speaker = mapping.get(raw_speaker, raw_speaker)

            if (
                previous_speaker is not None
                and speaker != previous_speaker
            ):
                f.write("\n")

            timestamp = simple_timestamp(segment["start"])
            f.write(
                f"[{timestamp}] {speaker}: {segment['text']}\n"
            )
            previous_speaker = speaker


def run_diarization_files(
    audio_file: Path,
    transcript_file: Path,
    output_file: Path,
) -> None:
    if not audio_file.exists():
        raise FileNotFoundError(audio_file)

    if not transcript_file.exists():
        raise FileNotFoundError(transcript_file)

    segments = load_transcript(transcript_file)
    if not segments:
        raise RuntimeError("No transcript segments found")

    turns = diarize(audio_file)
    save_result(segments, turns, output_file)

    print()
    print("=" * 60)
    print("DIARIZATION DONE")
    print("=" * 60)
    print(f"Output: {output_file}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diarize meeting.wav against a Whisper transcript."
    )
    parser.add_argument("audio_file", type=Path)
    parser.add_argument(
        "transcript_file",
        type=Path,
        help="meeting_segments.json (preferred) or meeting_timestamps.txt",
    )
    parser.add_argument("output_file", type=Path)
    args = parser.parse_args(argv)

    run_diarization_files(
        args.audio_file,
        args.transcript_file,
        args.output_file,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print()
        print("=" * 60)
        print("DIARIZATION ERROR")
        print("=" * 60)
        print(type(exc).__name__)
        print(exc)
        raise
