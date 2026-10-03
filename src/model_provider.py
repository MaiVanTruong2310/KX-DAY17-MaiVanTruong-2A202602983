"""Optional live chat-model adapters. Offline mode needs no third-party packages."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    aliases = {"anthorpic": "anthropic", "google": "gemini", "open-router": "openrouter"}
    name = aliases.get(value.strip().lower(), value.strip().lower())
    if name not in {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}:
        raise ValueError(f"Unsupported provider: {value}")
    return name


def build_chat_model(config: ProviderConfig):
    """Import only the selected integration, with a useful installation error."""
    provider = normalize_provider(config.provider)
    try:
        if provider in {"openai", "custom"}:
            from langchain_openai import ChatOpenAI
            if provider == "custom" and not config.base_url:
                raise ValueError("CUSTOM_BASE_URL is required for custom provider")
            return ChatOpenAI(model=config.model_name, temperature=config.temperature,
                              api_key=config.api_key, base_url=config.base_url)
        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=config.model_name, temperature=config.temperature,
                                          google_api_key=config.api_key)
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=config.model_name, temperature=config.temperature,
                                 api_key=config.api_key)
        if provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(model=config.model_name, temperature=config.temperature,
                              base_url=config.base_url)
        from langchain_openrouter import ChatOpenRouter
        return ChatOpenRouter(model=config.model_name, temperature=config.temperature,
                              api_key=config.api_key)
    except ImportError as exc:
        raise RuntimeError(f"Install the LangChain integration for {provider} to use live mode") from exc
