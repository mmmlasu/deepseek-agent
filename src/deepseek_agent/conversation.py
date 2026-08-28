"""Bounded conversation state with atomic persistence."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("Message content cannot be empty")


@dataclass(slots=True)
class Conversation:
    messages: list[Message] = field(default_factory=list)
    max_history_messages: int = 40

    def add(self, role: Role, content: str) -> Message:
        message = Message(role=role, content=content)
        self.messages.append(message)
        self._trim()
        return message

    def as_api_messages(self) -> list[dict[str, str]]:
        return [asdict(message) for message in self.messages]

    def clear(self, *, keep_system: bool = True) -> None:
        self.messages = (
            [message for message in self.messages if message.role == "system"]
            if keep_system
            else []
        )

    def _trim(self) -> None:
        system = next((message for message in self.messages if message.role == "system"), None)
        dialogue = [message for message in self.messages if message.role != "system"]
        self.messages = ([system] if system else []) + dialogue[-self.max_history_messages :]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"version": 1, "messages": self.as_api_messages()},
            ensure_ascii=False,
            indent=2,
        )
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    @classmethod
    def load(cls, path: Path, *, max_history_messages: int = 40) -> Conversation:
        if not path.exists():
            return cls(max_history_messages=max_history_messages)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            messages = [Message(**item) for item in payload["messages"]]
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Could not read conversation state from {path}: {exc}") from exc
        conversation = cls(messages=messages, max_history_messages=max_history_messages)
        conversation._trim()
        return conversation
