"""The fusion core is the moat — its log-odds math is checked against hand computation."""

from __future__ import annotations

import math

from bonafide.calibration import DeadbandGate, sigmoid
from bonafide.fusion import fuse
from bonafide.types import Evidence, Modality, SignalTier

OPEN_GATE = DeadbandGate(deadband=0.0)  # never abstains, so we can assert on the raw sign


def _ev(
    llr: float,
    reliability: float,
    *,
    applicable: bool = True,
    tier: SignalTier = SignalTier.WEAK,
    signal_id: str = "s",
) -> Evidence:
    return Evidence(
        signal_id=signal_id,
        applicable=applicable,
        llr=llr,
        reliability=reliability,
        tier=tier,
        model_version="test/0",
    )


def test_logodds_accumulation_matches_hand_computation() -> None:
    # L = logit(0.5) + (1.0 * 2.0) + (0.5 * -1.0) = 0 + 2.0 - 0.5 = 1.5
    ev = [_ev(2.0, 1.0, signal_id="a"), _ev(-1.0, 0.5, signal_id="b")]
    v = fuse(ev, modality=Modality.TEXT, engine_version="t", prior_ai=0.5, gate=OPEN_GATE)
    assert math.isclose(v.p_ai, sigmoid(1.5), rel_tol=1e-9)
    assert v.decision == "ai"


def test_non_applicable_signal_contributes_zero() -> None:
    ev = [_ev(2.0, 1.0, signal_id="real"), _ev(99.0, 1.0, applicable=False, signal_id="ghost")]
    v = fuse(ev, modality=Modality.TEXT, engine_version="t", gate=OPEN_GATE)
    assert math.isclose(v.p_ai, sigmoid(2.0), rel_tol=1e-9)


def test_no_applicable_signals_abstains_at_prior() -> None:
    ev = [_ev(5.0, 1.0, applicable=False)]
    v = fuse(ev, modality=Modality.TEXT, engine_version="t", prior_ai=0.3)
    assert v.decision == "abstain"
    assert v.p_ai == 0.3
    assert "no applicable signals" in v.conflicts


def test_deadband_forces_abstain_on_weak_evidence() -> None:
    ev = [_ev(0.5, 1.0)]
    v = fuse(ev, modality=Modality.TEXT, engine_version="t", gate=DeadbandGate(deadband=1.0))
    assert v.decision == "abstain"


def test_conflicting_conclusive_signals_force_abstain() -> None:
    ev = [
        _ev(8.0, 1.0, tier=SignalTier.CONCLUSIVE, signal_id="watermark-ai"),
        _ev(-8.0, 1.0, tier=SignalTier.CONCLUSIVE, signal_id="c2pa-camera"),
    ]
    v = fuse(ev, modality=Modality.IMAGE, engine_version="t", gate=OPEN_GATE)
    assert v.decision == "abstain"
    assert v.conflicts and "disagree" in v.conflicts[0]


def test_signal_declared_conflict_forces_abstain() -> None:
    # A signal can surface its own conflict (e.g. a tampered provenance chain) via
    # detail["conflict"]; the engine must abstain and report it, even with strong evidence.
    tampered = Evidence(
        signal_id="c2pa",
        applicable=True,
        llr=0.0,
        reliability=0.0,
        tier=SignalTier.CONCLUSIVE,
        detail={"conflict": "manifest INVALID - tampering"},
        model_version="test/0",
    )
    ev = [_ev(5.0, 1.0, signal_id="other"), tampered]
    v = fuse(ev, modality=Modality.IMAGE, engine_version="t", gate=OPEN_GATE)
    assert v.decision == "abstain"
    assert any("tampering" in c for c in v.conflicts)


def test_every_signal_is_reported_in_the_verdict() -> None:
    ev = [_ev(2.0, 1.0, signal_id="a"), _ev(0.0, 0.0, applicable=False, signal_id="b")]
    v = fuse(ev, modality=Modality.TEXT, engine_version="t")
    assert {e.signal_id for e in v.evidence} == {"a", "b"}
