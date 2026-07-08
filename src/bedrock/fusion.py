"""The evidence-fusion engine — Bedrock's core (DESIGN.md §4, ADR-0002).

Accumulate applicable evidence in log-odds space, calibrate to a probability, gate the
decision (with abstention), and surface conflicts. Pure and deterministic: a ``Verdict`` is a
function of ``(evidence, prior, calibrator, gate, alpha)`` alone — which is what makes it
reproducible and auditable.
"""

from __future__ import annotations

from collections.abc import Sequence

from .calibration import Calibrator, ConformalGate, DeadbandGate, IdentityCalibrator, logit
from .types import Evidence, Modality, SignalTier, Verdict

FUSION_VERSION = "logodds-v0"


def detect_conflicts(applicable: Sequence[Evidence]) -> list[str]:
    """Flag opposite-sign CONCLUSIVE signals — e.g. C2PA 'camera capture' vs. a verified
    AI watermark. High-trust signals must never be silently averaged against each other."""
    conclusive = [e for e in applicable if e.tier is SignalTier.CONCLUSIVE and e.reliability > 0.0]
    pos = [e.signal_id for e in conclusive if e.llr > 0.0]
    neg = [e.signal_id for e in conclusive if e.llr < 0.0]
    if pos and neg:
        return [
            f"Conclusive signals disagree: {', '.join(pos)} indicate AI; "
            f"{', '.join(neg)} indicate human/authentic."
        ]
    return []


def fuse(
    evidence: Sequence[Evidence],
    *,
    modality: Modality,
    engine_version: str,
    prior_ai: float = 0.5,
    alpha: float = 0.05,
    calibrator: Calibrator | None = None,
    gate: ConformalGate | None = None,
) -> Verdict:
    """Combine ``evidence`` into a single calibrated ``Verdict``.

    ``prior_ai`` is the base rate of AI content before any evidence (domain-configurable).
    ``alpha`` is the target error rate the abstention gate aims for.
    """
    active_calibrator: Calibrator = calibrator or IdentityCalibrator()
    active_gate: ConformalGate = gate or DeadbandGate()
    coverage = 1.0 - alpha
    ev = tuple(evidence)
    applicable = [e for e in ev if e.applicable]
    conflicts = detect_conflicts(applicable)

    if not applicable:
        # No signal spoke to this input. Absence is never evidence of "human" — we abstain.
        return Verdict(
            p_ai=prior_ai,
            ci=(0.0, 1.0),
            decision="abstain",
            coverage=coverage,
            modality=modality,
            evidence=ev,
            conflicts=("no applicable signals",),
            engine_version=engine_version,
        )

    log_odds = logit(prior_ai) + sum(e.contribution for e in applicable)
    p_ai = active_calibrator.probability(log_odds)
    ci = active_calibrator.interval(log_odds)
    decision = active_gate.decide(log_odds, alpha)

    if conflicts and decision != "abstain":
        decision = "abstain"  # honesty beats a coin-flip when high-trust signals disagree

    return Verdict(
        p_ai=p_ai,
        ci=ci,
        decision=decision,
        coverage=coverage,
        modality=modality,
        evidence=ev,
        conflicts=tuple(conflicts),
        engine_version=engine_version,
    )
