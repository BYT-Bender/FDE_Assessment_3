"""Gemini API configuration loaded from project root without exposing secrets."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)

DEFAULT_MODEL = "gemini-3.1-flash-lite"
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class MissingAPIKeyError(RuntimeError):
    """Raised when Gemini real mode is requested without a configured key."""


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str
    model: str
    timeout_seconds: float
    request_pacing_seconds: float = 1.0
    rate_limit_retries: int = 2
    retry_after_cap_seconds: float = 30.0

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key.strip())


def get_gemini_config() -> GeminiConfig:
    """Read Gemini settings after loading the project-root .env (OS values win)."""
    raw_timeout = os.getenv("GEMINI_TIMEOUT_SECONDS", "45")
    try:
        timeout = float(raw_timeout)
    except ValueError as exc:
        raise ValueError("GEMINI_TIMEOUT_SECONDS must be a positive number.") from exc
    if not 1 <= timeout <= 180:
        raise ValueError("GEMINI_TIMEOUT_SECONDS must be between 1 and 180 seconds.")
    try:
        pacing = float(os.getenv("GEMINI_REQUEST_PACING_SECONDS", "1"))
        retry_after_cap = float(os.getenv("GEMINI_RETRY_AFTER_CAP_SECONDS", "30"))
        retries = int(os.getenv("GEMINI_RATE_LIMIT_RETRIES", "2"))
    except ValueError as exc:
        raise ValueError("Gemini pacing, retry, and Retry-After settings must be numeric.") from exc
    if not 0 <= pacing <= 60:
        raise ValueError("GEMINI_REQUEST_PACING_SECONDS must be between 0 and 60 seconds.")
    if not 0 <= retries <= 3:
        raise ValueError("GEMINI_RATE_LIMIT_RETRIES must be between 0 and 3.")
    if not 0 <= retry_after_cap <= 120:
        raise ValueError("GEMINI_RETRY_AFTER_CAP_SECONDS must be between 0 and 120 seconds.")
    return GeminiConfig(
        api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        timeout_seconds=timeout,
        request_pacing_seconds=pacing,
        rate_limit_retries=retries,
        retry_after_cap_seconds=retry_after_cap,
    )


def require_gemini_config() -> GeminiConfig:
    config = get_gemini_config()
    if not config.is_configured:
        raise MissingAPIKeyError(
            "Real LLM mode requires GEMINI_API_KEY. Create a free Gemini API key in Google AI Studio, "
            "add it to the project-root .env file, or explicitly choose deterministic demo mode."
        )
    return config
