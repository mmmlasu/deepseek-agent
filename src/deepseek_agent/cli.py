"""Interactive terminal interface."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from .client import DeepSeekClient
from .config import Settings
from .conversation import Conversation
from .errors import DeepSeekError
from .telemetry import configure_telemetry

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def chat(
    api_key: Annotated[str | None, typer.Option(envvar="DEEPSEEK_API_KEY", hidden=True)] = None,
    system: Annotated[str | None, typer.Option(help="Optional system message.")] = None,
    state: Annotated[Path | None, typer.Option(help="Conversation state file.")] = None,
    no_save: Annotated[bool, typer.Option(help="Do not load or save state.")] = False,
    thinking: Annotated[bool, typer.Option(help="Enable DeepSeek thinking mode.")] = False,
    temperature: Annotated[float, typer.Option(min=0.0, max=2.0)] = 0.7,
    max_tokens: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    """Start a streaming conversation with DeepSeek."""
    try:
        settings = Settings.from_env(api_key=api_key)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    state_path = state.expanduser() if state else settings.state_path
    try:
        conversation = (
            Conversation(max_history_messages=settings.max_history_messages)
            if no_save
            else Conversation.load(state_path, max_history_messages=settings.max_history_messages)
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    if system and not any(message.role == "system" for message in conversation.messages):
        conversation.add("system", system)
    try:
        telemetry = configure_telemetry(settings)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        asyncio.run(
            _chat_loop(
            settings,
            conversation,
            state_path=None if no_save else state_path,
            temperature=temperature,
            max_tokens=max_tokens,
                thinking=thinking,
            )
        )
    finally:
        telemetry.flush()


async def _chat_loop(
    settings: Settings,
    conversation: Conversation,
    *,
    state_path: Path | None,
    temperature: float,
    max_tokens: int | None,
    thinking: bool,
) -> None:
    console.print(
        Panel.fit(
            "[bold]DeepSeek Agent[/bold]\n"
            "Type [cyan]/new[/cyan] to clear, [cyan]/history[/cyan] to inspect, "
            "or [cyan]/quit[/cyan] to exit.",
            border_style="blue",
        )
    )
    async with DeepSeekClient(settings) as client:
        while True:
            try:
                prompt = await asyncio.to_thread(Prompt.ask, "[bold green]You[/bold green]")
            except (EOFError, KeyboardInterrupt):
                console.print()
                return
            command = prompt.strip().lower()
            if command in {"/quit", "/exit"}:
                return
            if command == "/new":
                conversation.clear()
                _save(conversation, state_path)
                console.print("[dim]Conversation cleared.[/dim]")
                continue
            if command == "/history":
                for message in conversation.messages:
                    console.print(f"[bold]{message.role}:[/bold] {message.content}")
                continue
            if not prompt.strip():
                continue

            conversation.add("user", prompt)
            pieces: list[str] = []
            try:
                with Live(Markdown(""), console=console, refresh_per_second=12) as live:
                    async for piece in client.stream(
                        conversation.as_api_messages(),
                        temperature=temperature,
                        max_tokens=max_tokens,
                        thinking=thinking,
                    ):
                        pieces.append(piece)
                        live.update(Markdown("".join(pieces)))
            except DeepSeekError as exc:
                conversation.messages.pop()
                console.print(f"[red]{exc}[/red]")
                continue
            answer = "".join(pieces)
            if not answer:
                conversation.messages.pop()
                console.print("[red]DeepSeek returned an empty response.[/red]")
                continue
            conversation.add("assistant", answer)
            _save(conversation, state_path)


def _save(conversation: Conversation, state_path: Path | None) -> None:
    if state_path is not None:
        try:
            conversation.save(state_path)
        except OSError as exc:
            console.print(f"[yellow]Could not save state: {exc}[/yellow]")


if __name__ == "__main__":
    app()
