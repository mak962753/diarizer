"""Summarize meeting transcripts with local llama-cli."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from diarizer.config import LlmConfig, load_llm_config, resolve_llm_model_path
from diarizer.paths import CONFIG_PATH


TRANSCRIPT_CANDIDATES = (
    "meeting_speakers.txt",
    "meeting_timestamps.txt",
    "meeting.txt",
)

_THINKING_RE = re.compile(
    r"\[Start thinking\].*?\[End thinking\]",
    re.DOTALL | re.IGNORECASE,
)
_THINK_TAG_RE = re.compile(
    r"<think>.*?</think>",
    re.DOTALL | re.IGNORECASE,
)


def find_llama_cli() -> str:
    path = shutil.which("llama-cli")
    if not path:
        raise RuntimeError(
            "llama-cli not found on PATH. "
            "Install with: brew install llama.cpp"
        )
    return path


def estimate_tokens(text: str, chars_per_token: float) -> int:
    return max(1, int(len(text) / max(chars_per_token, 0.5)))


def render_prompt(template: str, transcript: str) -> str:
    if "{transcript}" not in template:
        raise RuntimeError(
            "Prompt template must contain the {transcript} placeholder"
        )
    return template.replace("{transcript}", transcript.strip())


def pick_transcript_path(path: Path) -> Path:
    """Accept a transcript file or a meeting folder."""
    path = path.expanduser().resolve()

    if path.is_file():
        return path

    if path.is_dir():
        for name in TRANSCRIPT_CANDIDATES:
            candidate = path / name
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"No transcript found in {path}. "
            f"Looked for: {', '.join(TRANSCRIPT_CANDIDATES)}"
        )

    raise FileNotFoundError(path)


def default_summary_path(transcript_path: Path) -> Path:
    if transcript_path.name in TRANSCRIPT_CANDIDATES:
        return transcript_path.parent / "meeting_summary.md"
    return transcript_path.with_name(transcript_path.stem + "_summary.md")


def prompt_overhead_tokens(template: str, cfg: LlmConfig) -> int:
    """Tokens reserved for instructions + answer inside ctx_size."""
    empty = render_prompt(template, "")
    # n_predict=-1 means until EOS; still leave room in the window for output.
    output_reserve = (
        cfg.n_predict if cfg.n_predict > 0 else max(2048, cfg.ctx_size // 8)
    )
    return estimate_tokens(empty, cfg.chars_per_token) + output_reserve + 256


def max_transcript_chars(template: str, cfg: LlmConfig) -> int:
    budget_tokens = cfg.ctx_size - prompt_overhead_tokens(template, cfg)
    if budget_tokens < 1024:
        raise RuntimeError(
            f"ctx_size={cfg.ctx_size} is too small for summarization. "
            "Increase [llm].ctx_size."
        )
    return int(budget_tokens * cfg.chars_per_token)


def chunk_transcript(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    lines = text.splitlines(keepends=True)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append("".join(current).strip())
            current = []
            current_len = 0

    for line in lines:
        if len(line) > max_chars:
            flush()
            for i in range(0, len(line), max_chars):
                piece = line[i : i + max_chars].strip()
                if piece:
                    chunks.append(piece)
            continue

        if current_len + len(line) > max_chars and current:
            flush()

        current.append(line)
        current_len += len(line)

    flush()
    return chunks or [text]


def split_system_and_user(prompt: str) -> tuple[str, str]:
    """Prefer instructions as system, transcript body as user turn."""
    markers = (
        "\nТранскрипт:\n",
        "\nTranscript:\n",
        "\nЧастичные резюме:\n",
    )
    for marker in markers:
        if marker in prompt:
            system, user = prompt.split(marker, 1)
            system = system.strip()
            user = user.strip()
            if system and user:
                return system, user
    return "", prompt.strip()


def truncate_repetition(text: str, *, max_repeats: int = 3) -> str:
    """Safety net if the model loops despite sampling penalties."""
    lines = text.splitlines()
    cleaned: list[str] = []
    repeat = 0
    prev = None

    for line in lines:
        normalized = re.sub(r"\s+", " ", line.strip())
        if normalized and normalized == prev:
            repeat += 1
            if repeat >= max_repeats:
                cleaned.append("\n[... truncated repeated output ...]")
                break
        else:
            repeat = 0
            prev = normalized or prev
        cleaned.append(line)

    joined = "\n".join(cleaned)
    loop = re.search(r"(.{40,}?)(\1){4,}", joined, re.DOTALL)
    if loop:
        joined = joined[: loop.start(2)] + "\n[... truncated repeated output ...]\n"
    return joined.strip() + "\n"


def clean_llama_output(raw: str) -> str:
    text = raw.strip()

    for marker in ("\nAssistant:", "\nassistant:"):
        if marker in text:
            text = text.split(marker, 1)[1].strip()
            break
    if text.startswith("Assistant:"):
        text = text[len("Assistant:") :].strip()

    text = _THINKING_RE.sub("", text)
    text = _THINK_TAG_RE.sub("", text)

    for start in ("[Start thinking]", "<think>"):
        idx = text.find(start)
        if idx != -1:
            text = text[:idx].strip()

    text = truncate_repetition(text)
    return text.strip() + "\n"


def run_llama_cli(
    *,
    model_path: Path,
    prompt: str,
    output_file: Path,
    cfg: LlmConfig,
    label: str = "LLM SUMMARIZATION",
) -> str:
    llama = find_llama_cli()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    prompt_tokens = estimate_tokens(prompt, cfg.chars_per_token)
    if prompt_tokens + 64 > cfg.ctx_size:
        raise RuntimeError(
            f"Prompt still too large for ctx_size={cfg.ctx_size}: "
            f"~{prompt_tokens} tokens. Increase [llm].ctx_size or "
            "lower chars_per_token conservatism."
        )

    system_prompt, user_prompt = split_system_and_user(prompt)

    with tempfile.TemporaryDirectory(prefix="diarizer-llm-") as tmp:
        tmp_dir = Path(tmp)
        user_file = tmp_dir / "user.txt"
        user_file.write_text(user_prompt, encoding="utf-8")

        command = [
            llama,
            "--model",
            str(model_path),
            "--file",
            str(user_file),
            "--no-display-prompt",
            "--n-predict",
            str(cfg.n_predict),
            "--ctx-size",
            str(cfg.ctx_size),
            "--temp",
            str(cfg.temperature),
            "--n-gpu-layers",
            str(cfg.gpu_layers),
            "--repeat-penalty",
            str(cfg.repeat_penalty),
            "--presence-penalty",
            str(cfg.presence_penalty),
            "--frequency-penalty",
            str(cfg.frequency_penalty),
            "--dry-multiplier",
            str(cfg.dry_multiplier),
            "--reasoning",
            cfg.reasoning,
            "--reasoning-budget",
            str(cfg.reasoning_budget),
            "--single-turn",
            "--output",
            str(output_file),
        ]

        if system_prompt:
            system_file = tmp_dir / "system.txt"
            system_file.write_text(system_prompt, encoding="utf-8")
            command.extend(["--system-prompt-file", str(system_file)])

        predict_label = "until EOS" if cfg.n_predict < 0 else str(cfg.n_predict)

        print()
        print("=" * 60)
        print(label)
        print("=" * 60)
        print(f"Model:     {model_path}")
        print(f"Output:    {output_file}")
        print(
            f"Context:   {cfg.ctx_size} "
            f"(prompt ~{prompt_tokens} tokens, n_predict={predict_label})"
        )
        print(
            f"Sampling:  temp={cfg.temperature} "
            f"repeat={cfg.repeat_penalty} dry={cfg.dry_multiplier} "
            f"reasoning={cfg.reasoning}/{cfg.reasoning_budget}"
        )
        print(f"Cmd:       {' '.join(command[:6])} ...")

        result = subprocess.run(command, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"llama-cli failed with exit code {result.returncode}"
            )

    if not output_file.is_file() or output_file.stat().st_size == 0:
        raise RuntimeError(
            f"llama-cli produced no output at {output_file}"
        )

    text = clean_llama_output(output_file.read_text(encoding="utf-8"))
    if not text.strip():
        raise RuntimeError(
            "llama-cli produced empty summary after cleaning "
            "(model may have only emitted a thinking block)."
        )
    output_file.write_text(text, encoding="utf-8")
    return text


def summarize_text(
    transcript: str,
    output_file: Path,
    *,
    cfg: LlmConfig | None = None,
    config_path: Path | None = None,
) -> str:
    cfg = cfg or load_llm_config(config_path)
    model_path = resolve_llm_model_path(cfg.model)

    max_chars = max_transcript_chars(cfg.prompt, cfg)
    chunks = chunk_transcript(transcript, max_chars)

    print(
        f"Transcript chars: {len(transcript.strip())} | "
        f"chunks: {len(chunks)} | "
        f"max chars/chunk: {max_chars} | "
        f"ctx_size: {cfg.ctx_size}"
    )

    if len(chunks) == 1:
        return run_llama_cli(
            model_path=model_path,
            prompt=render_prompt(cfg.prompt, chunks[0]),
            output_file=output_file,
            cfg=cfg,
            label="LLM SUMMARIZATION (llama-cli)",
        )

    partials: list[str] = []
    with tempfile.TemporaryDirectory(prefix="diarizer-partial-") as tmp:
        tmp_dir = Path(tmp)
        for index, chunk in enumerate(chunks, start=1):
            labeled = (
                f"[Фрагмент {index}/{len(chunks)} транскрипта встречи]\n\n"
                f"{chunk}"
            )
            partial_path = tmp_dir / f"partial_{index:02d}.md"
            partial = run_llama_cli(
                model_path=model_path,
                prompt=render_prompt(cfg.prompt, labeled),
                output_file=partial_path,
                cfg=cfg,
                label=f"LLM SUMMARIZATION chunk {index}/{len(chunks)}",
            )
            partials.append(
                f"### Фрагмент {index}/{len(chunks)}\n\n{partial.strip()}\n"
            )

        merged_body = "\n".join(partials)
        merge_chunks = chunk_transcript(
            merged_body,
            max_transcript_chars(cfg.merge_prompt, cfg),
        )
        if len(merge_chunks) == 1:
            return run_llama_cli(
                model_path=model_path,
                prompt=render_prompt(cfg.merge_prompt, merge_chunks[0]),
                output_file=output_file,
                cfg=cfg,
                label="LLM SUMMARIZATION merge",
            )

        mid_partials: list[str] = []
        for index, chunk in enumerate(merge_chunks, start=1):
            mid_path = tmp_dir / f"merge_mid_{index:02d}.md"
            mid = run_llama_cli(
                model_path=model_path,
                prompt=render_prompt(cfg.merge_prompt, chunk),
                output_file=mid_path,
                cfg=cfg,
                label=f"LLM SUMMARIZATION merge-pass {index}/{len(merge_chunks)}",
            )
            mid_partials.append(mid.strip())

        return run_llama_cli(
            model_path=model_path,
            prompt=render_prompt(
                cfg.merge_prompt,
                "\n\n".join(
                    f"### Промежуточное резюме {i}\n\n{text}"
                    for i, text in enumerate(mid_partials, start=1)
                ),
            ),
            output_file=output_file,
            cfg=cfg,
            label="LLM SUMMARIZATION final merge",
        )


def summarize_file(
    input_path: Path,
    output_file: Path | None = None,
    *,
    cfg: LlmConfig | None = None,
    config_path: Path | None = None,
) -> Path:
    transcript_path = pick_transcript_path(input_path)
    out = output_file or default_summary_path(transcript_path)
    transcript = transcript_path.read_text(encoding="utf-8")
    if not transcript.strip():
        raise RuntimeError(f"Transcript is empty: {transcript_path}")

    print(f"Transcript: {transcript_path}")
    summarize_text(
        transcript,
        out,
        cfg=cfg,
        config_path=config_path,
    )
    print(f"Summary saved: {out}")
    return out


def summarize_meeting_dir(
    meeting_dir: Path,
    *,
    cfg: LlmConfig | None = None,
    config_path: Path | None = None,
) -> Path:
    meeting_dir = meeting_dir.expanduser().resolve()
    return summarize_file(
        meeting_dir,
        meeting_dir / "meeting_summary.md",
        cfg=cfg,
        config_path=config_path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize a meeting transcript with local llama-cli. "
            "Pass a transcript file or a meetings/<timestamp>/ folder."
        )
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Transcript file or meeting folder",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output markdown path (default: meeting_summary.md next to input)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help=f"Path to config.toml (default: {CONFIG_PATH})",
    )
    args = parser.parse_args(argv)

    try:
        out = summarize_file(
            args.path,
            args.output,
            config_path=args.config,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    print()
    print(f"Done: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
