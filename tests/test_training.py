"""Fitters: degenerate-input guards and the prior-provenance contract (review-driven)."""

from __future__ import annotations

import math

import pytest

from bedrock.calibration import CalibrationArtifact
from bedrock.training import fit_artifact, fit_isotonic, fit_platt


def _separable(n: int = 40) -> tuple[list[float], list[int]]:
    xs = [float(i - n // 2) for i in range(n)]
    ys = [1 if x > 0 else 0 for x in xs]
    return xs, ys


def test_single_class_corpus_is_rejected() -> None:
    # A human-only calibration set would give q_ai = inf -> the gate decides "ai" for everything:
    # a confident false accusation from zero AI examples. Fitting must refuse.
    with pytest.raises(ValueError, match="single-class"):
        fit_artifact([-1.0, -2.0, -3.0], [0, 0, 0])
    with pytest.raises(ValueError, match="single-class"):
        fit_artifact([1.0, 2.0, 3.0], [1, 1, 1])


def test_empty_corpus_fails_at_fit_time_not_inference() -> None:
    with pytest.raises(ValueError, match="at least one"):
        fit_isotonic([], [])
    with pytest.raises(ValueError, match="single-class"):
        fit_artifact([], [])


def test_fit_records_the_prior_and_folds_it_into_the_fingerprint() -> None:
    xs, ys = _separable()
    a = fit_artifact(xs, ys, prior_ai=0.5)
    b = fit_artifact(xs, ys, prior_ai=0.7)
    assert a.fit_prior == 0.5 and b.fit_prior == 0.7
    assert a.fingerprint != b.fingerprint  # different prior -> different calibration -> distinct id
    assert a.version != b.version


def test_platt_stays_finite_on_perfectly_separable_data() -> None:
    xs, ys = _separable(60)
    cal = fit_platt(xs, ys)
    assert math.isfinite(cal.a) and math.isfinite(cal.b)
    assert 0.0 <= cal.probability(0.0) <= 1.0


def test_artifact_without_a_recorded_prior_reports_none() -> None:
    from bedrock.calibration import default_artifact

    assert default_artifact().fit_prior is None
    assert CalibrationArtifact.from_dict(fit_artifact(*_separable()).to_dict()).fit_prior == 0.5
