"""Langfuse-backed OpenTelemetry setup and privacy-safe span helpers."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import Any

from langfuse import Langfuse
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider

from .config import Settings


@dataclass(slots=True)
class Telemetry:
    langfuse: Langfuse | None

    def flush(self) -> None:
        if self.langfuse is not None:
            self.langfuse.flush()


def _export_every_span(_: Any) -> bool:
    return True


def client_ip_attribute(settings: Settings, client_ip: str | None) -> str | None:
    """Return a normalized IP only when collection was explicitly enabled."""
    if not settings.otel_capture_client_ip or not client_ip:
        return None
    try:
        return str(ipaddress.ip_address(client_ip))
    except ValueError as exc:
        raise ValueError("client_ip must be a valid IPv4 or IPv6 address") from exc


def configure_telemetry(settings: Settings) -> Telemetry:
    """Configure a shared global OTEL provider and Langfuse exporter."""
    public_key = settings.langfuse_public_key
    secret_key = settings.langfuse_secret_key
    if not public_key and not secret_key:
        return Telemetry(langfuse=None)
    if not public_key or not secret_key:
        raise ValueError("Set both LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")

    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider(resource=Resource.create({"service.name": "deepseek-agent"}))
        trace.set_tracer_provider(provider)

    langfuse = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=settings.langfuse_base_url,
        tracer_provider=provider,
        should_export_span=_export_every_span,
    )
    return Telemetry(langfuse=langfuse)
