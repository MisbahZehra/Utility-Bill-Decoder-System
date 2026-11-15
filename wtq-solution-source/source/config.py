"""Configuration loading and validation (only competition-approved models are accepted)."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

ALLOWED_MODELS = {
    "anthropic": {"claude-haiku-4-5"},
    "google": {"gemini-3.5-flash-lite", "gemini-3.1-flash-lite"},
    "openai": {"gpt-4.1-mini", "gpt-5-mini"},
}


class ConfigError(Exception):
    pass


@dataclass
class Config:
    provider: str
    model: str
    api_key: str
    min_call_interval: float = 0.0
    max_retries: int = 6
    timeout: int = 240


def load_config() -> Config:
    load_dotenv()
    provider = os.getenv("MODEL_PROVIDER", "").strip().lower()
    model = os.getenv("MODEL_NAME", "").strip()
    api_key = os.getenv("API_KEY", "").strip()
    if provider not in ALLOWED_MODELS:
        raise ConfigError(f"MODEL_PROVIDER must be one of {sorted(ALLOWED_MODELS)}, got '{provider}'")
    if model not in ALLOWED_MODELS[provider]:
        raise ConfigError(f"MODEL_NAME '{model}' is not allowed for {provider}. Allowed: {sorted(ALLOWED_MODELS[provider])}")
    if not api_key or api_key == "your-api-key-here":
        raise ConfigError("API_KEY is missing. Copy .env.example to .env and put your key in it.")
    try:
        interval = float(os.getenv("MIN_CALL_INTERVAL", "0") or 0)
    except ValueError:
        interval = 0.0
    return Config(provider=provider, model=model, api_key=api_key, min_call_interval=interval)
