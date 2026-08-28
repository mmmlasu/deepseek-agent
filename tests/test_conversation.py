from __future__ import annotations

import json

import pytest

from deepseek_agent.conversation import Conversation


def test_trim_keeps_system_and_latest_dialogue() -> None:
    conversation = Conversation(max_history_messages=2)
    conversation.add("system", "Be kind")
    conversation.add("user", "one")
    conversation.add("assistant", "two")
    conversation.add("user", "three")
    assert [(m.role, m.content) for m in conversation.messages] == [
        ("system", "Be kind"),
        ("assistant", "two"),
        ("user", "three"),
    ]


def test_round_trip_state(tmp_path) -> None:
    path = tmp_path / "state" / "chat.json"
    original = Conversation()
    original.add("user", "你好")
    original.save(path)
    loaded = Conversation.load(path)
    assert loaded.as_api_messages() == [{"role": "user", "content": "你好"}]
    assert json.loads(path.read_text())["version"] == 1


def test_corrupt_state_has_actionable_error(tmp_path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("not-json")
    with pytest.raises(ValueError, match="Could not read conversation state"):
        Conversation.load(path)
