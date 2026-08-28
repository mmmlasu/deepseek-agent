"""Async DeepSeek chat-completions client with strict SSE parsing."""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import AsyncIterator
from typing import Any

import httpx
from opentelemetry import trace

from .config import Settings
from .errors import APIError, AuthenticationError, ProtocolError, RateLimitError

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
_tracer = trace.get_tracer("deepseek-agent.client")


class DeepSeekClient:
    def __init__(self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.base_url,
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "deepseek-agent/0.1.0",
            },
            timeout=httpx.Timeout(settings.timeout_seconds),
            transport=transport,
        )

    async def __aenter__(self) -> DeepSeekClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    @staticmethod
    def _payload(
        messages: list[dict[str, str]],
        *,
        model: str,
        stream: bool,
        temperature: float,
        max_tokens: int | None,
        thinking: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if thinking:
            payload["thinking"] = {"type": "enabled"}
        return payload

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        thinking: bool = False,
    ) -> str:
        response = await self._request_json(
            self._payload(
                messages,
                model=self.settings.model,
                stream=False,
                temperature=temperature,
                max_tokens=max_tokens,
                thinking=thinking,
            )
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProtocolError("Response did not contain an assistant message") from exc
        if not isinstance(content, str):
            raise ProtocolError("Assistant message content was not text")
        return content

    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        thinking: bool = False,
    ) -> AsyncIterator[str]:
        payload = self._payload(
            messages,
            model=self.settings.model,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking,
        )
        response = await self._open_stream(payload)
        try:
            async for event in self._iter_sse(response):
                if event == "[DONE]":
                    return
                try:
                    chunk = json.loads(event)
                    content = chunk["choices"][0]["delta"].get("content")
                except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
                    raise ProtocolError("Malformed streaming event from DeepSeek API") from exc
                if isinstance(content, str) and content:
                    yield content
        finally:
            await response.aclose()

    async def _request_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self.settings.max_retries + 1):
            try:
                with _tracer.start_as_current_span("deepseek.chat.completions") as span:
                    span.set_attribute("gen_ai.system", "deepseek")
                    span.set_attribute("gen_ai.request.model", self.settings.model)
                    span.set_attribute("gen_ai.request.stream", False)
                    response = await self._client.post("/chat/completions", json=payload)
                    span.set_attribute("http.response.status_code", response.status_code)
            except httpx.RequestError as exc:
                if attempt >= self.settings.max_retries:
                    raise APIError(f"Could not reach DeepSeek API: {exc}") from exc
                await self._backoff(attempt)
                continue
            if response.status_code in _RETRYABLE_STATUS and attempt < self.settings.max_retries:
                await self._backoff(attempt, response)
                continue
            self._raise_for_status(response)
            try:
                data = response.json()
            except ValueError as exc:
                raise ProtocolError("DeepSeek API returned invalid JSON") from exc
            if not isinstance(data, dict):
                raise ProtocolError("DeepSeek API returned an unexpected JSON value")
            return data
        raise AssertionError("retry loop exhausted")

    async def _open_stream(self, payload: dict[str, Any]) -> httpx.Response:
        for attempt in range(self.settings.max_retries + 1):
            request = self._client.build_request("POST", "/chat/completions", json=payload)
            try:
                with _tracer.start_as_current_span("deepseek.chat.completions.stream") as span:
                    span.set_attribute("gen_ai.system", "deepseek")
                    span.set_attribute("gen_ai.request.model", self.settings.model)
                    span.set_attribute("gen_ai.request.stream", True)
                    response = await self._client.send(request, stream=True)
                    span.set_attribute("http.response.status_code", response.status_code)
            except httpx.RequestError as exc:
                if attempt >= self.settings.max_retries:
                    raise APIError(f"Could not reach DeepSeek API: {exc}") from exc
                await self._backoff(attempt)
                continue
            if response.status_code in _RETRYABLE_STATUS and attempt < self.settings.max_retries:
                await response.aclose()
                await self._backoff(attempt, response)
                continue
            self._raise_for_status(response)
            return response
        raise AssertionError("retry loop exhausted")

    @staticmethod
    async def _iter_sse(response: httpx.Response) -> AsyncIterator[str]:
        data_lines: list[str] = []
        async for line in response.aiter_lines():
            if not line:
                if data_lines:
                    yield "\n".join(data_lines)
                    data_lines.clear()
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if data_lines:
            yield "\n".join(data_lines)

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.is_success:
            return
        request_id = response.headers.get("x-request-id")
        suffix = f" (request {request_id})" if request_id else ""
        if response.status_code in {401, 403}:
            raise AuthenticationError(f"DeepSeek rejected the API key{suffix}")
        if response.status_code == 429:
            raise RateLimitError(f"DeepSeek rate limit exceeded{suffix}")
        detail = response.text.strip()[:300] or response.reason_phrase
        raise APIError(f"DeepSeek returned HTTP {response.status_code}: {detail}{suffix}")

    @staticmethod
    async def _backoff(attempt: int, response: httpx.Response | None = None) -> None:
        if response is not None and (retry_after := response.headers.get("retry-after")):
            try:
                delay = min(float(retry_after), 30.0)
            except ValueError:
                delay = 0.0
        else:
            delay = 0.5 * (2**attempt) + random.uniform(0, 0.2)
        await asyncio.sleep(delay)
