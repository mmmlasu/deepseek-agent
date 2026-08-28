from __future__ import annotations

from deepseek_agent.config import Settings
from deepseek_agent.telemetry import configure_telemetry


def test_telemetry_is_optional() -> None:
    telemetry = configure_telemetry(Settings(api_key="test"))
    assert telemetry.langfuse is None
    telemetry.flush()


def test_half_configured_telemetry_is_rejected() -> None:
    settings = Settings(api_key="test", langfuse_public_key="pk-test")
    try:
        configure_telemetry(settings)
    except ValueError as exc:
        assert "both" in str(exc)
    else:
        raise AssertionError("expected configuration error")


def test_client_ip_is_opt_in_and_default_off() -> None:
    from deepseek_agent.telemetry import client_ip_attribute

    assert client_ip_attribute(Settings(api_key="test"), "203.0.113.7") is None
    enabled = Settings(api_key="test", otel_capture_client_ip=True)
    assert client_ip_attribute(enabled, "203.0.113.7") == "203.0.113.7"


def test_client_ip_rejects_invalid_values_when_enabled() -> None:
    import pytest

    from deepseek_agent.telemetry import client_ip_attribute

    enabled = Settings(api_key="test", otel_capture_client_ip=True)
    with pytest.raises(ValueError, match="valid IPv4 or IPv6"):
        client_ip_attribute(enabled, "not-an-ip")
