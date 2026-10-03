"""Shared paths and settings for the Day 17 lab."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    try:
        from dotenv import load_dotenv
        load_dotenv(root / ".env", override=False)
    except ImportError:
        pass
    provider = normalize_provider(os.getenv("LLM_PROVIDER", "openai"))
    keys = {"openai": "OPENAI_API_KEY", "custom": "CUSTOM_API_KEY",
            "gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
            "openrouter": "OPENROUTER_API_KEY"}
    urls = {"custom": "CUSTOM_BASE_URL", "ollama": "OLLAMA_BASE_URL",
            "openrouter": "OPENROUTER_BASE_URL"}
    defaults = {"openai": "gpt-4o-mini", "custom": "custom-model", "gemini": "gemini-2.0-flash",
                "anthropic": "claude-3-5-haiku-latest", "ollama": "llama3.2",
                "openrouter": "openai/gpt-4o-mini"}
    model = ProviderConfig(provider, os.getenv("LLM_MODEL", defaults[provider]),
                           float(os.getenv("LLM_TEMPERATURE", "0")),
                           os.getenv(keys.get(provider, "")), os.getenv(urls.get(provider, "")))
    judge = ProviderConfig(normalize_provider(os.getenv("JUDGE_PROVIDER", provider)),
                           os.getenv("JUDGE_MODEL", model.model_name), 0.0,
                           os.getenv("JUDGE_API_KEY", model.api_key),
                           os.getenv("JUDGE_BASE_URL", model.base_url))
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    threshold = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "600"))
    keep = int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))
    if threshold < 1 or keep < 1:
        raise ValueError("Compact threshold and keep count must be positive")
    return LabConfig(root, root / "data", state_dir, threshold, keep, model, judge)
