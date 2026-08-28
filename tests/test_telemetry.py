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
