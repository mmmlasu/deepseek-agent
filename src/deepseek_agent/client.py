"""Async DeepSeek chat-completions client with strict SSE parsing."""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import AsyncIterator
from typing import Any

import httpx
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from .config import Settings
from .errors import APIError, AuthenticationError, ProtocolError, RateLimitError
from .telemetry import client_ip_attribute

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
        client_ip: str | None = None,
    ) -> str:
        response = await self._request_json(
            self._payload(
                messages,
                model=self.settings.model,
                stream=False,
                temperature=temperature,
                max_tokens=max_tokens,
                thinking=thinking,
            ),
            client_ip=client_ip,
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
        client_ip: str | None = None,
    ) -> AsyncIterator[str]:
        payload = self._payload(
            messages,
            model=self.settings.model,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking,
        )
        response, span = await self._open_stream(payload, client_ip=client_ip)
        try:
            async for event in self._iter_sse(response):
                if event == "[DONE]":
                    return
                try:
                    chunk = json.loads(event)
                    self._set_response_attributes(span, chunk, response)
                    content = chunk["choices"][0]["delta"].get("content")
                except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
                    raise ProtocolError("Malformed streaming event from DeepSeek API") from exc
                if isinstance(content, str) and content:
                    yield content
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            raise
        finally:
            await response.aclose()
            span.end()

    async def _request_json(
        self, payload: dict[str, Any], *, client_ip: str | None = None
    ) -> dict[str, Any]:
        for attempt in range(self.settings.max_retries + 1):
            response: httpx.Response | None = None
            try:
                with _tracer.start_as_current_span("deepseek.chat") as span:
                    self._set_request_attributes(span, payload, client_ip)
                    response = await self._client.post("/chat/completions", json=payload)
                    span.set_attribute("http.response.status_code", response.status_code)
                    self._raise_for_status(response)
                    try:
                        data = response.json()
                    except ValueError as exc:
                        raise ProtocolError("DeepSeek API returned invalid JSON") from exc
                    if not isinstance(data, dict):
                        raise ProtocolError("DeepSeek API returned an unexpected JSON value")
                    self._set_response_attributes(span, data, response)
                    return data
            except httpx.RequestError as exc:
                if attempt >= self.settings.max_retries:
                    raise APIError(f"Could not reach DeepSeek API: {exc}") from exc
                await self._backoff(attempt)
            except (AuthenticationError, RateLimitError, APIError):
                if (
                    response is not None
                    and response.status_code in _RETRYABLE_STATUS
                    and attempt < self.settings.max_retries
                ):
                    await self._backoff(attempt, response)
                    continue
                raise
        raise AssertionError("retry loop exhausted")

    async def _open_stream(
        self, payload: dict[str, Any], *, client_ip: str | None = None
    ) -> tuple[httpx.Response, Span]:
        for attempt in range(self.settings.max_retries + 1):
            request = self._client.build_request("POST", "/chat/completions", json=payload)
            span = _tracer.start_span("deepseek.chat.stream")
            self._set_request_attributes(span, payload, client_ip)
            try:
                response = await self._client.send(request, stream=True)
                span.set_attribute("http.response.status_code", response.status_code)
            except httpx.RequestError as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                span.end()
                if attempt >= self.settings.max_retries:
                    raise APIError(f"Could not reach DeepSeek API: {exc}") from exc
                await self._backoff(attempt)
                continue
            try:
                self._raise_for_status(response)
            except (AuthenticationError, RateLimitError, APIError) as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                span.end()
                if (
                    response.status_code in _RETRYABLE_STATUS
                    and attempt < self.settings.max_retries
                ):
                    await response.aclose()
                    await self._backoff(attempt, response)
                    continue
                raise
            return response, span
        raise AssertionError("retry loop exhausted")

    def _set_request_attributes(
        self, span: Span, payload: dict[str, Any], client_ip: str | None
    ) -> None:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "deepseek")
        span.set_attribute("gen_ai.system", "deepseek")
        span.set_attribute("gen_ai.request.model", self.settings.model)
        span.set_attribute("gen_ai.request.stream", bool(payload["stream"]))
        span.set_attribute("gen_ai.request.temperature", float(payload["temperature"]))
        span.set_attribute("server.address", self._client.base_url.host)
        if payload.get("max_tokens") is not None:
            span.set_attribute("gen_ai.request.max_tokens", int(payload["max_tokens"]))
        if payload.get("thinking") is not None:
            span.set_attribute("deepseek.request.thinking", True)
        if normalized_ip := client_ip_attribute(self.settings, client_ip):
            span.set_attribute("client.address", normalized_ip)

    @staticmethod
    def _set_response_attributes(
        span: Span, data: dict[str, Any], response: httpx.Response
    ) -> None:
        if request_id := response.headers.get("x-request-id"):
            span.set_attribute("deepseek.request_id", request_id)
        if isinstance(data.get("model"), str):
            span.set_attribute("gen_ai.response.model", data["model"])
        if isinstance(data.get("id"), str):
            span.set_attribute("gen_ai.response.id", data["id"])
        choices = data.get("choices")
        if isinstance(choices, list):
            reasons = [c.get("finish_reason") for c in choices if isinstance(c, dict)]
            finish_reasons = [r for r in reasons if isinstance(r, str)]
            if finish_reasons:
                span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)
        usage = data.get("usage")
        if isinstance(usage, dict):
            if isinstance(usage.get("prompt_tokens"), int):
                span.set_attribute("gen_ai.usage.input_tokens", usage["prompt_tokens"])
            if isinstance(usage.get("completion_tokens"), int):
                span.set_attribute("gen_ai.usage.output_tokens", usage["completion_tokens"])

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
