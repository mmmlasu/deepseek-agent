"""Explicit environment-to-configuration boundary."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-pro"
    timeout_seconds: float = 60.0
    max_retries: int = 3
    max_history_messages: int = 40
    state_path: Path = Path("~/.local/state/deepseek-agent/conversation.json")

    @classmethod
    def from_env(cls, *, api_key: str | None = None) -> Settings:
        key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        if not key:
            raise ValueError("Missing API key. Set DEEPSEEK_API_KEY or pass --api-key.")
        return cls(
            api_key=key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", cls.base_url).rstrip("/"),
            model=os.getenv("DEEPSEEK_MODEL", cls.model),
            timeout_seconds=float(os.getenv("DEEPSEEK_TIMEOUT", cls.timeout_seconds)),
            max_retries=int(os.getenv("DEEPSEEK_MAX_RETRIES", cls.max_retries)),
            max_history_messages=int(
                os.getenv("DEEPSEEK_MAX_HISTORY_MESSAGES", cls.max_history_messages)
            ),
            state_path=Path(
                os.getenv("DEEPSEEK_STATE_PATH", str(cls.state_path))
            ).expanduser(),
        )
