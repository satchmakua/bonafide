"""Calibration + abstention: math checked against hand computation, and round-trips preserved."""

from __future__ import annotations

import json
import math
import random
from itertools import pairwise

from bedrock.calibration import (
    CalibrationArtifact,
    DeadbandGate,
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    SplitConformalGate,
    calibrator_from_dict,
    default_artifact,
    gate_from_dict,
    logit,
    sigmoid,
)
from bedrock.eval import ece
from bedrock.training import conformal_quantile, fit_isotonic, fit_platt


def test_sigmoid_logit_round_trip() -> None:
    for p in (0.01, 0.3, 0.5, 0.73, 0.99):
        assert math.isclose(sigmoid(logit(p)), p, rel_tol=1e-6)


def test_conformal_quantile_exact_hand_values() -> None:
    scores = list(range(1, 11))  # [1..10]
    assert conformal_quantile(scores, alpha=0.1) == 10  # k = ceil(11*0.9) = 10 -> 10th smallest
    assert conformal_quantile(scores, alpha=0.2) == 9  # k = ceil(11*0.8) = 9
    assert conformal_quantile([10, 20, 30, 40, 50], alpha=0.1) == math.inf  # k=6 > n=5 -> inf
    assert conformal_quantile([], alpha=0.1) == math.inf


def test_split_conformal_gate_set_logic() -> None:
    gate = SplitConformalGate(q_ai=1.0, q_human=1.0, alpha=0.1)
    assert gate.predict_set(5.0) == {"ai"}  # s_ai=-5<=1, s_human=5>1
    assert gate.decide(5.0, 0.1) == "ai"
    assert gate.predict_set(-5.0) == {"human"}
    assert gate.decide(-5.0, 0.1) == "human"
    assert gate.predict_set(0.0) == {"ai", "human"}  # both plausible
    assert gate.decide(0.0, 0.1) == "abstain"


def test_calibrator_serialization_round_trip() -> None:
    for cal in (
        IdentityCalibrator(ci_margin=0.7),
        PlattCalibrator(a=1.3, b=-0.2, ci_halfwidth=0.9),
        IsotonicCalibrator(xs=(-2.0, 0.0, 2.0), ys=(0.1, 0.5, 0.9), ci_halfwidth=0.4),
    ):
        clone = calibrator_from_dict(cal.to_dict())
        assert clone.to_dict() == cal.to_dict()
        assert math.isclose(clone.probability(0.4), cal.probability(0.4), rel_tol=1e-12)


def test_gate_serialization_round_trip() -> None:
    for gate in (DeadbandGate(deadband=0.8), SplitConformalGate(q_ai=1.2, q_human=0.9, alpha=0.1)):
        clone = gate_from_dict(gate.to_dict())
        assert clone.to_dict() == gate.to_dict()
        assert clone.decide(0.5, 0.1) == gate.decide(0.5, 0.1)


def test_artifact_round_trip_and_fingerprint(tmp_path: object) -> None:
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    art = CalibrationArtifact(
        calibrator=PlattCalibrator(a=1.1, b=0.0),
        gate=SplitConformalGate(1.0, 1.0, 0.1),
        meta={"x": 1},
    )
    path = tmp_path / "cal.json"
    art.save(path)
    loaded = CalibrationArtifact.load(path)
    assert loaded.version == art.version
    assert loaded.calibrator.to_dict() == art.calibrator.to_dict()
    # Fingerprint is content-derived: a different fit yields a different version.
    other = CalibrationArtifact(calibrator=PlattCalibrator(a=2.0, b=0.0), gate=art.gate)
    assert other.fingerprint != art.fingerprint


def test_isotonic_calibration_reduces_ece() -> None:
    rng = random.Random(1)

    def draw(n: int) -> tuple[list[float], list[int]]:
        xs, ys = [], []
        for _ in range(n):
            ai = rng.random() < 0.5
            xs.append(rng.gauss(1.6 if ai else -1.6, 1.5))
            ys.append(1 if ai else 0)
        return xs, ys

    cx, cy = draw(2000)
    cal = fit_isotonic(cx, cy)
    tx, ty = draw(4000)
    raw_ece = ece([sigmoid(x) for x in tx], ty)
    cal_ece = ece([cal.probability(x) for x in tx], ty)
    assert cal_ece < raw_ece
    assert cal_ece < 0.05  # the design's calibration target


def test_platt_calibration_is_monotone_and_reasonable() -> None:
    rng = random.Random(2)
    xs = [rng.gauss(2 if i % 2 else -2, 1.0) for i in range(1000)]
    ys = [1 if x > 0 else 0 for x in xs]  # cleanly separable
    cal = fit_platt(xs, ys)
    assert cal.probability(-3.0) < cal.probability(0.0) < cal.probability(3.0)
    assert cal.a > 0  # positive slope: higher log-odds -> more AI


def test_isotonic_is_monotone_nondecreasing() -> None:
    cal = fit_isotonic([-3.0, -1.0, 0.0, 1.0, 3.0], [0, 0, 1, 0, 1])
    grid = [cal.probability(x / 10) for x in range(-40, 41)]
    assert all(b >= a - 1e-9 for a, b in pairwise(grid))


def test_default_artifact_is_the_uncalibrated_fallback() -> None:
    art = default_artifact()
    assert isinstance(art.calibrator, IdentityCalibrator)
    assert isinstance(art.gate, DeadbandGate)


def test_conformal_gate_reports_its_own_coverage_not_the_runtime_alpha() -> None:
    # A gate fit at alpha=0.1 guarantees 90% regardless of the alpha it's later called with;
    # reporting 1 - runtime_alpha would overstate the guarantee — the failure we must not make.
    gate = SplitConformalGate(q_ai=1.0, q_human=1.0, alpha=0.1)
    for runtime_alpha in (0.01, 0.05, 0.2):
        assert gate.coverage(runtime_alpha) == 0.9
    # The dead-band default has no guarantee, so it reports the nominal target.
    assert DeadbandGate().coverage(0.05) == 0.95


def test_infinite_threshold_serializes_to_valid_json_and_round_trips() -> None:
    gate = SplitConformalGate(q_ai=math.inf, q_human=1.5, alpha=0.1)
    art = CalibrationArtifact(calibrator=IdentityCalibrator(), gate=gate)
    raw = json.dumps(art.to_dict())
    assert "Infinity" not in raw  # bare Infinity is invalid JSON
    json.loads(raw, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))  # strict OK
    restored = CalibrationArtifact.from_dict(art.to_dict())
    assert isinstance(restored.gate, SplitConformalGate)
    assert math.isinf(restored.gate.q_ai) and restored.gate.q_human == 1.5
