"""``bedrock`` command-line interface — the first way to drive the engine by hand."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .engine import detect_file, detect_text
from .report import format_verdict

app = typer.Typer(
    add_completion=False,
    help="Bedrock - calibrated, provenance-first AI-content detection.",
)


@app.callback()
def _root() -> None:
    """Keep ``detect`` a named subcommand (Typer collapses a lone command to the root)."""


@app.command()
def detect(
    source: str = typer.Argument(..., help="Path to a file to analyze (or raw text with --text)."),
    text: bool = typer.Option(False, "--text", "-t", help="Treat SOURCE as raw text, not a path."),
    prior: float = typer.Option(0.5, help="Prior probability of AI, in (0, 1)."),
    alpha: float = typer.Option(0.05, help="Target error rate for the abstention gate."),
    as_json: bool = typer.Option(False, "--json", help="Emit the raw Verdict as JSON."),
) -> None:
    """Analyze SOURCE and print a calibrated verdict with its evidence trail."""
    if text or not Path(source).is_file():
        verdict = detect_text(source, prior_ai=prior, alpha=alpha)
    else:
        verdict = detect_file(source, prior_ai=prior, alpha=alpha)

    if as_json:
        # ensure_ascii (the json default) keeps redirected stdout safe on Windows cp1252,
        # where pydantic's model_dump_json would emit raw non-ASCII and raise.
        typer.echo(json.dumps(verdict.model_dump(mode="json"), indent=2))
    else:
        typer.echo(format_verdict(verdict))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
