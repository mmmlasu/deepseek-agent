"""Two-call live smoke test used by the manually dispatched GitHub workflow."""

from __future__ import annotations

import asyncio
import os

from deepseek_agent import DeepSeekClient, Settings
from deepseek_agent.telemetry import configure_telemetry


async def run_case(*, capture_ip: bool, marker: str) -> None:
    settings = Settings.from_env()
    settings = Settings(
        **{
            name: getattr(settings, name)
            for name in settings.__dataclass_fields__
            if name != "otel_capture_client_ip"
        },
        otel_capture_client_ip=capture_ip,
    )
    telemetry = configure_telemetry(settings)
    try:
        async with DeepSeekClient(settings) as client:
            answer = await client.complete(
                [{"role": "user", "content": f"Reply with exactly: {marker}"}],
                temperature=0,
                max_tokens=16,
                client_ip="203.0.113.7",
            )
        if not answer.strip():
            raise RuntimeError(f"empty live response for {marker!r}")
        print(f"{marker}: live DeepSeek response verified; capture_ip={capture_ip}")
    finally:
        telemetry.flush()


async def main() -> None:
    run_id = os.environ["GITHUB_RUN_ID"]
    await run_case(capture_ip=False, marker=f"otel-ip-off-{run_id}")
    await run_case(capture_ip=True, marker=f"otel-ip-on-{run_id}")


if __name__ == "__main__":
    asyncio.run(main())
