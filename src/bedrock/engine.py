"""The orchestrator: ingest -> route to a modality's signals -> collect Evidence -> fuse.

This is the thin spine the whole product hangs off. Public entry points are ``detect`` and its
typed siblings (``detect_text`` / ``detect_file`` / ``detect_bytes``).
"""

from __future__ import annotations

from pathlib import Path

from ._version import __version__
from .fusion import FUSION_VERSION, fuse
from .ingest import context_from_bytes, context_from_path, context_from_text
from .signals.base import SignalRegistry, not_applicable
from .signals.lexical import LexicalHeuristicSignal
from .types import Evidence, InputContext, Verdict


def engine_version() -> str:
    """Version string stamped into every Verdict for reproducibility."""
    return f"bedrock/{__version__}+fuse-{FUSION_VERSION}"


def default_registry() -> SignalRegistry:
    """The signals wired up by default. Grows one adapter per milestone."""
    registry = SignalRegistry()
    registry.register(LexicalHeuristicSignal())
    return registry


def _run(ctx: InputContext, *, registry: SignalRegistry, prior_ai: float, alpha: float) -> Verdict:
    evidence: list[Evidence] = []
    for signal in registry.for_modality(ctx.modality):
        if signal.applies_to(ctx):
            evidence.append(signal.analyze(ctx))
        else:
            evidence.append(not_applicable(signal, note="out of distribution / not applicable"))
    return fuse(
        evidence,
        modality=ctx.modality,
        engine_version=engine_version(),
        prior_ai=prior_ai,
        alpha=alpha,
    )


def detect_text(
    text: str,
    *,
    prior_ai: float = 0.5,
    alpha: float = 0.05,
    registry: SignalRegistry | None = None,
) -> Verdict:
    """Detect on a raw text string."""
    return _run(
        context_from_text(text),
        registry=registry or default_registry(),
        prior_ai=prior_ai,
        alpha=alpha,
    )


def detect_bytes(
    data: bytes,
    *,
    path: str | None = None,
    prior_ai: float = 0.5,
    alpha: float = 0.05,
    registry: SignalRegistry | None = None,
) -> Verdict:
    """Detect on raw bytes (modality is sniffed; ``path`` helps the sniffer)."""
    return _run(
        context_from_bytes(data, path=path),
        registry=registry or default_registry(),
        prior_ai=prior_ai,
        alpha=alpha,
    )


def detect_file(
    path: str | Path,
    *,
    prior_ai: float = 0.5,
    alpha: float = 0.05,
    registry: SignalRegistry | None = None,
) -> Verdict:
    """Detect on a file, read from disk."""
    return _run(
        context_from_path(path),
        registry=registry or default_registry(),
        prior_ai=prior_ai,
        alpha=alpha,
    )


def detect(
    source: str | Path,
    *,
    prior_ai: float = 0.5,
    alpha: float = 0.05,
    registry: SignalRegistry | None = None,
) -> Verdict:
    """Convenience entry point: treat ``source`` as a file if it exists on disk, else as text."""
    try:
        is_file = Path(source).is_file()
    except OSError:
        is_file = False
    if is_file:
        return detect_file(source, prior_ai=prior_ai, alpha=alpha, registry=registry)
    return detect_text(str(source), prior_ai=prior_ai, alpha=alpha, registry=registry)
