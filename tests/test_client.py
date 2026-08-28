from __future__ import annotations

import json

import httpx
import pytest

from deepseek_agent.client import DeepSeekClient
from deepseek_agent.config import Settings
from deepseek_agent.errors import AuthenticationError, ProtocolError


def settings(**overrides: object) -> Settings:
    values: dict[str, object] = {"api_key": "test-key", "max_retries": 0}
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_complete_sends_documented_request_and_reads_message() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("https://api.deepseek.com/chat/completions")
        assert request.headers["authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload["model"] == "deepseek-v4-pro"
        assert payload["stream"] is False
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "Hello!"}}]}
        )

    async with DeepSeekClient(settings(), transport=httpx.MockTransport(handler)) as client:
        result = await client.complete([{"role": "user", "content": "Hi"}])
    assert result == "Hello!"


@pytest.mark.asyncio
async def test_stream_parses_comments_multiline_events_and_done() -> None:
    body = (
        ': keepalive\n\n'
        'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
        'data: {"choices":[{"delta":\n'
        'data: {"content":" world"}}]}\n\n'
        'data: [DONE]\n\n'
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    async with DeepSeekClient(settings(), transport=httpx.MockTransport(handler)) as client:
        chunks = [
            chunk
            async for chunk in client.stream([{"role": "user", "content": "Hi"}])
        ]
    assert chunks == ["Hello", " world"]


@pytest.mark.asyncio
async def test_thinking_option_uses_current_api_shape() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["thinking"] == {"type": "enabled"}
        return httpx.Response(200, text="data: [DONE]\n\n")

    async with DeepSeekClient(settings(), transport=httpx.MockTransport(handler)) as client:
        assert [chunk async for chunk in client.stream([], thinking=True)] == []


@pytest.mark.asyncio
async def test_authentication_error_does_not_leak_key() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized", headers={"x-request-id": "req-123"})

    async with DeepSeekClient(settings(), transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(AuthenticationError, match="req-123") as raised:
            await client.complete([{"role": "user", "content": "Hi"}])
    assert "test-key" not in str(raised.value)


@pytest.mark.asyncio
async def test_malformed_success_response_is_protocol_error() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    async with DeepSeekClient(settings(), transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProtocolError):
            await client.complete([{"role": "user", "content": "Hi"}])


class RecordingSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value


@pytest.mark.parametrize("enabled", [False, True])
def test_ip_attribute_follows_explicit_privacy_setting(enabled: bool) -> None:
    client = DeepSeekClient(settings(otel_capture_client_ip=enabled))
    span = RecordingSpan()
    client._set_request_attributes(  # type: ignore[arg-type]
        span,
        {"stream": False, "temperature": 0.2, "max_tokens": 8},
        "203.0.113.7",
    )
    assert ("client.address" in span.attributes) is enabled
    if enabled:
        assert span.attributes["client.address"] == "203.0.113.7"


@pytest.mark.asyncio
async def test_complete_accepts_reasoning_only_response() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "", "reasoning_content": "fallback"}}]},
        )

    async with DeepSeekClient(settings(), transport=httpx.MockTransport(handler)) as client:
        result = await client.complete([{"role": "user", "content": "Hi"}])
    assert result == "fallback"
