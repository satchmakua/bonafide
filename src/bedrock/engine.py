"""The orchestrator: ingest -> route to a modality's signals -> collect Evidence -> fuse.

This is the thin spine the whole product hangs off. Public entry points are ``detect`` and its
typed siblings (``detect_text`` / ``detect_file`` / ``detect_bytes``). A ``CalibrationArtifact``
(fitted calibrator + abstention gate) is threaded through and stamped into every Verdict; the
default is the uncalibrated identity/dead-band fallback until a fitted artifact is supplied.
"""

from __future__ import annotations

from pathlib import Path

from ._version import __version__
from .calibration import CalibrationArtifact, default_artifact
from .fusion import FUSION_VERSION, fuse
from .ingest import context_from_bytes, context_from_path, context_from_text
from .signals.base import SignalRegistry, not_applicable
from .signals.lexical import LexicalHeuristicSignal
from .types import Evidence, InputContext, Verdict


def engine_version(artifact: CalibrationArtifact | None = None) -> str:
    """Version string stamped into every Verdict for reproducibility (incl. the calibration)."""
    cal = (artifact or default_artifact()).version
    return f"bedrock/{__version__}+fuse-{FUSION_VERSION}+cal-{cal}"


def default_registry() -> SignalRegistry:
    """The signals wired up by default. Grows one adapter per milestone; signals whose
    optional extra isn't installed are simply not offered (never an import error)."""
    registry = SignalRegistry()
    registry.register(LexicalHeuristicSignal())
    try:
        from .signals.provenance import C2paSignal  # requires the [provenance] extra
    except ImportError:
        pass
    else:
        registry.register(C2paSignal())
    return registry


def _run(
    ctx: InputContext,
    *,
    registry: SignalRegistry,
    prior_ai: float,
    alpha: float,
    calibration: CalibrationArtifact,
) -> Verdict:
    evidence: list[Evidence] = []
    for signal in registry.for_modality(ctx.modality):
        if signal.applies_to(ctx):
            evidence.append(signal.analyze(ctx))
        else:
            evidence.append(not_applicable(signal, note="out of distribution / not applicable"))
    # A fitted calibrator was trained against log-odds computed at its fit-time prior; using a
    # different prior here would shift inputs off the calibrated scale. So the artifact's recorded
    # prior wins whenever present (the uncalibrated default records none, so the caller's holds).
    effective_prior = calibration.fit_prior if calibration.fit_prior is not None else prior_ai
    return fuse(
        evidence,
        modality=ctx.modality,
        engine_version=engine_version(calibration),
        prior_ai=effective_prior,
        alpha=alpha,
        calibrator=calibration.calibrator,
        gate=calibration.gate,
    )


def detect_text(
    text: str,
    *,
    prior_ai: float = 0.5,
    alpha: float = 0.05,
    registry: SignalRegistry | None = None,
    calibration: CalibrationArtifact | None = None,
) -> Verdict:
    """Detect on a raw text string."""
    return _run(
        context_from_text(text),
        registry=registry or default_registry(),
        prior_ai=prior_ai,
        alpha=alpha,
        calibration=calibration or default_artifact(),
    )


def detect_bytes(
    data: bytes,
    *,
    path: str | None = None,
    prior_ai: float = 0.5,
    alpha: float = 0.05,
    registry: SignalRegistry | None = None,
    calibration: CalibrationArtifact | None = None,
) -> Verdict:
    """Detect on raw bytes (modality is sniffed; ``path`` helps the sniffer)."""
    return _run(
        context_from_bytes(data, path=path),
        registry=registry or default_registry(),
        prior_ai=prior_ai,
        alpha=alpha,
        calibration=calibration or default_artifact(),
    )


def detect_file(
    path: str | Path,
    *,
    prior_ai: float = 0.5,
    alpha: float = 0.05,
    registry: SignalRegistry | None = None,
    calibration: CalibrationArtifact | None = None,
) -> Verdict:
    """Detect on a file, read from disk."""
    return _run(
        context_from_path(path),
        registry=registry or default_registry(),
        prior_ai=prior_ai,
        alpha=alpha,
        calibration=calibration or default_artifact(),
    )


def detect(
    source: str | Path,
    *,
    prior_ai: float = 0.5,
    alpha: float = 0.05,
    registry: SignalRegistry | None = None,
    calibration: CalibrationArtifact | None = None,
) -> Verdict:
    """Convenience entry point: treat ``source`` as a file if it exists on disk, else as text."""
    try:
        is_file = Path(source).is_file()
    except OSError:
        is_file = False
    if is_file:
        return detect_file(
            source, prior_ai=prior_ai, alpha=alpha, registry=registry, calibration=calibration
        )
    return detect_text(
        str(source), prior_ai=prior_ai, alpha=alpha, registry=registry, calibration=calibration
    )
