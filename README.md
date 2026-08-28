# DeepSeek Agent

A focused terminal agent for the [DeepSeek API](https://api-docs.deepseek.com/). It streams answers as they arrive, keeps a bounded conversation across sessions, and fails with useful errors instead of stack traces or leaked credentials.

## What it does well

- Parses real server-sent events rather than faking streaming
- Persists bounded conversation history with atomic file replacement
- Retries only transient failures, respecting `Retry-After`
- Separates configuration, transport, state, and terminal presentation
- Offers optional DeepSeek thinking mode
- Maps auth, rate-limit, API, and protocol failures to secret-safe errors
- Exposes an async client for embedding
- Runs a deterministic mocked test suite without spending API credits

## Setup

Requires Python 3.10+ and a DeepSeek API key.

```bash
git clone https://github.com/mmmlasu/deepseek-agent.git
cd deepseek-agent
python -m venv .venv
source .venv/bin/activate
pip install -e .
export DEEPSEEK_API_KEY="your-fresh-key"
deepseek-agent chat
```

Do not reuse a key that has been pasted into a chat or committed to a file. Revoke it and create a fresh one.

Useful options:

```bash
deepseek-agent chat \
  --system "Be concise, warm, and practical." \
  --thinking \
  --temperature 0.7 \
  --max-tokens 800
```

Inside a chat:

- `/new` clears the dialogue while retaining the system message
- `/history` prints the context that will be sent on the next turn
- `/quit` exits

State defaults to `~/.local/state/deepseek-agent/conversation.json`. Use `--state PATH` or `--no-save`.

## Library use

```python
import asyncio

from deepseek_agent import Conversation, DeepSeekClient, Settings


async def main() -> None:
    conversation = Conversation()
    conversation.add("user", "Explain event loops with one concrete example.")

    async with DeepSeekClient(Settings.from_env()) as client:
        async for text in client.stream(conversation.as_api_messages()):
            print(text, end="", flush=True)


asyncio.run(main())
```

Use `DeepSeekClient.complete(...)` when a non-streaming response fits better.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | required | API key, read only at runtime |
| `DEEPSEEK_MODEL` | `deepseek-v4-pro` | Current DeepSeek model alias |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | OpenAI-compatible API host |
| `DEEPSEEK_TIMEOUT` | `60` | Request timeout in seconds |
| `DEEPSEEK_MAX_RETRIES` | `3` | Retries for 408, 429, and transient 5xx responses |
| `DEEPSEEK_MAX_HISTORY_MESSAGES` | `40` | Recent non-system messages retained |
| `DEEPSEEK_STATE_PATH` | `~/.local/state/deepseek-agent/conversation.json` | State file |

The app deliberately does not auto-load `.env`: importing the package never changes process state. [`.env.example`](.env.example) is only a template.

## Architecture

```text
CLI (Rich + Typer)
  ├── Conversation: validation, trimming, atomic JSON persistence
  ├── Settings: environment-to-config boundary
  └── DeepSeekClient: HTTP, retries, status mapping, SSE parsing
```

The client accepts an `httpx` transport, so tests are deterministic and downstream apps can add tracing or network policy. Streaming retries happen only before a successful body begins. Once output starts, a request is never silently replayed.

## Development

```bash
pip install -e '.[dev]'
ruff check .
mypy src
pytest
```

Tests cover the official request URL and auth shape, SSE comments and multiline events, `[DONE]`, thinking mode, state trimming and round trips, malformed responses, and credential-safe authentication failures.

## Security and privacy

- API keys are never written to state.
- Errors never include the `Authorization` header.
- `.env` is ignored.
- State stores prompts and answers as plain JSON. Use `--no-save` for sensitive chats.

## API references

- Quick start and current model aliases: <https://api-docs.deepseek.com/>
- API reference: <https://api-docs.deepseek.com/api/deepseek-api/>
- Streaming thinking example: <https://api-docs.deepseek.com/guides/thinking_mode_api_example_streaming>

This is an independent client and is not affiliated with or endorsed by DeepSeek.

## License

MIT
