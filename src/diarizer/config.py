"""Load repo config.toml (single source of settings)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from diarizer.paths import CONFIG_PATH, MODELS_DIR


@dataclass(frozen=True)
class LlmConfig:
    model: str
    prompt: str
    merge_prompt: str
    summarize: bool
    n_predict: int
    ctx_size: int
    temperature: float
    gpu_layers: int
    chars_per_token: float
    repeat_penalty: float
    presence_penalty: float
    frequency_penalty: float
    dry_multiplier: float
    reasoning: str
    reasoning_budget: int


def load_config(path: Path | None = None) -> dict:
    config_path = path or CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config not found: {config_path}\n"
            "Expected config.toml at the repo root."
        )

    with config_path.open("rb") as f:
        return tomllib.load(f)


def _require(section: dict[str, Any], key: str) -> Any:
    if key not in section:
        raise RuntimeError(
            f"config.toml [llm] is missing required key: {key}"
        )
    return section[key]


def _require_str(section: dict[str, Any], key: str) -> str:
    value = _require(section, key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"[llm].{key} must be a non-empty string")
    return value.strip()


def _require_bool(section: dict[str, Any], key: str) -> bool:
    value = _require(section, key)
    if not isinstance(value, bool):
        raise RuntimeError(f"[llm].{key} must be a boolean")
    return value


def _require_int(section: dict[str, Any], key: str) -> int:
    value = _require(section, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"[llm].{key} must be an integer")
    return value


def _require_float(section: dict[str, Any], key: str) -> float:
    value = _require(section, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"[llm].{key} must be a number")
    return float(value)


def load_llm_config(path: Path | None = None) -> LlmConfig:
    data = load_config(path)
    section = data.get("llm")
    if not isinstance(section, dict):
        raise RuntimeError("config.toml is missing an [llm] section")

    reasoning = _require_str(section, "reasoning").lower()
    if reasoning not in {"on", "off", "auto"}:
        raise RuntimeError("[llm].reasoning must be on|off|auto")

    return LlmConfig(
        model=_require_str(section, "model"),
        prompt=_require_str(section, "prompt") + "\n",
        merge_prompt=_require_str(section, "merge_prompt") + "\n",
        summarize=_require_bool(section, "summarize"),
        n_predict=_require_int(section, "n_predict"),
        ctx_size=_require_int(section, "ctx_size"),
        temperature=_require_float(section, "temperature"),
        gpu_layers=_require_int(section, "gpu_layers"),
        chars_per_token=_require_float(section, "chars_per_token"),
        repeat_penalty=_require_float(section, "repeat_penalty"),
        presence_penalty=_require_float(section, "presence_penalty"),
        frequency_penalty=_require_float(section, "frequency_penalty"),
        dry_multiplier=_require_float(section, "dry_multiplier"),
        reasoning=reasoning,
        reasoning_budget=_require_int(section, "reasoning_budget"),
    )


def resolve_llm_model_path(relative: str) -> Path:
    model_path = (MODELS_DIR / relative).resolve()
    if not model_path.exists():
        raise FileNotFoundError(
            f"LLM model not found: {model_path}\n"
            f"(config value relative to models/: {relative!r})\n"
            "Create a symlink under models/llm/ or fix [llm].model."
        )
    return model_path
