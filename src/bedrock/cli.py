"""``bedrock`` command-line interface — the first way to drive the engine by hand."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .calibration import CalibrationArtifact
from .engine import detect_file, detect_text
from .report import format_verdict

app = typer.Typer(
    add_completion=False,
    help="Bedrock - calibrated, provenance-first AI-content detection.",
)


@app.callback()
def _root() -> None:
    """Group callback (Typer collapses a lone command to the root; this keeps subcommands named)."""


@app.command()
def detect(
    source: str = typer.Argument(..., help="Path to a file to analyze (or raw text with --text)."),
    text: bool = typer.Option(False, "--text", "-t", help="Treat SOURCE as raw text, not a path."),
    prior: float = typer.Option(0.5, help="Prior probability of AI, in (0, 1)."),
    alpha: float = typer.Option(0.05, help="Target error rate for the abstention gate."),
    calibration: Path | None = typer.Option(
        None, "--calibration", "-c", help="Path to a fitted calibration artifact (JSON)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the raw Verdict as JSON."),
) -> None:
    """Analyze SOURCE and print a calibrated verdict with its evidence trail."""
    artifact = CalibrationArtifact.load(calibration) if calibration is not None else None
    if text or not Path(source).is_file():
        verdict = detect_text(source, prior_ai=prior, alpha=alpha, calibration=artifact)
    else:
        verdict = detect_file(source, prior_ai=prior, alpha=alpha, calibration=artifact)

    if as_json:
        # ensure_ascii (the json default) keeps redirected stdout safe on Windows cp1252,
        # where pydantic's model_dump_json would emit raw non-ASCII and raise.
        typer.echo(json.dumps(verdict.model_dump(mode="json"), indent=2))
    else:
        typer.echo(format_verdict(verdict))


@app.command()
def fit(
    corpus: Path = typer.Argument(..., help="Labeled JSONL corpus: {text, label, group?}."),
    out: Path = typer.Option(Path("calibration.json"), "--out", "-o", help="Artifact output path."),
    method: str = typer.Option("isotonic", help="Calibration method: isotonic | platt."),
    alpha: float = typer.Option(0.1, help="Target error rate for the conformal gate."),
    prior: float = typer.Option(0.5, help="Prior probability of AI, in (0, 1)."),
) -> None:
    """Fit a calibration + abstention artifact from a labeled corpus and write it to disk."""
    from .eval import load_corpus, score_corpus
    from .training import fit_artifact

    scored = score_corpus(load_corpus(corpus), prior_ai=prior)
    labeled = [(s.log_odds, s.label) for s in scored if s.log_odds is not None]
    if not labeled:
        raise typer.BadParameter("No example produced an applicable signal; nothing to fit.")
    log_odds, labels = zip(*labeled, strict=True)
    try:
        artifact = fit_artifact(
            list(log_odds),
            list(labels),
            method=method,
            alpha=alpha,
            prior_ai=prior,
            meta={"corpus": str(corpus), "n_no_signal": len(scored) - len(labeled)},
        )
    except ValueError as exc:  # single-class corpus, unknown method, ...
        raise typer.BadParameter(str(exc)) from exc
    artifact.save(out)
    typer.echo(f"Fitted {artifact.version}")
    typer.echo(f"  {len(labeled)}/{len(scored)} examples scored -> {out}")


@app.command(name="eval")
def eval_cmd(
    corpus: Path = typer.Argument(..., help="Labeled JSONL corpus: {text, label, group?}."),
    calibration: Path | None = typer.Option(
        None, "--calibration", "-c", help="Fitted artifact to evaluate (default: uncalibrated)."
    ),
    prior: float = typer.Option(0.5, help="Prior probability of AI, in (0, 1)."),
) -> None:
    """Evaluate calibration + decisions on a labeled corpus (ECE, FPR, coverage, ...)."""
    from .eval import format_report, run_eval

    report = run_eval(corpus, artifact_path=calibration, prior_ai=prior)
    typer.echo(format_report(report))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
