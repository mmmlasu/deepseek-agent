"""Explicit environment-to-configuration boundary."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    DEFAULT_BASE_URL = "https://api.deepseek.com"
    DEFAULT_MODEL = "deepseek-v4-pro"
    DEFAULT_TIMEOUT_SECONDS = 60.0
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_MAX_HISTORY_MESSAGES = 40
    DEFAULT_STATE_PATH = Path("~/.local/state/deepseek-agent/conversation.json")
    DEFAULT_LANGFUSE_BASE_URL = "https://us.cloud.langfuse.com"

    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    max_history_messages: int = DEFAULT_MAX_HISTORY_MESSAGES
    state_path: Path = DEFAULT_STATE_PATH
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_base_url: str = DEFAULT_LANGFUSE_BASE_URL

    @classmethod
    def from_env(cls, *, api_key: str | None = None) -> Settings:
        key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        if not key:
            raise ValueError("Missing API key. Set DEEPSEEK_API_KEY or pass --api-key.")
        return cls(
            api_key=key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", cls.DEFAULT_BASE_URL).rstrip("/"),
            model=os.getenv("DEEPSEEK_MODEL", cls.DEFAULT_MODEL),
            timeout_seconds=float(os.getenv("DEEPSEEK_TIMEOUT", cls.DEFAULT_TIMEOUT_SECONDS)),
            max_retries=int(os.getenv("DEEPSEEK_MAX_RETRIES", cls.DEFAULT_MAX_RETRIES)),
            max_history_messages=int(
                os.getenv("DEEPSEEK_MAX_HISTORY_MESSAGES", cls.DEFAULT_MAX_HISTORY_MESSAGES)
            ),
            state_path=Path(
                os.getenv("DEEPSEEK_STATE_PATH", str(cls.DEFAULT_STATE_PATH))
            ).expanduser(),
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            langfuse_base_url=os.getenv(
                "LANGFUSE_BASE_URL", cls.DEFAULT_LANGFUSE_BASE_URL
            ).rstrip("/"),
        )
