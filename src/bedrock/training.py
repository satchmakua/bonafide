"""Fit calibrators and abstention gates from labeled ``(log_odds, label)`` data.

Pure stdlib — no numpy/sklearn. The three fitters:

- ``fit_platt``          — logistic (Platt/temperature) scaling by Newton-Raphson.
- ``fit_isotonic``       — monotone calibration by pool-adjacent-violators (PAVA).
- ``fit_conformal_gate`` — class-conditional split-conformal thresholds (the coverage guarantee).

``fit_artifact`` bundles a calibrator + gate into a ``CalibrationArtifact``. Labels are 1 = AI,
0 = human (``as_label`` accepts the string forms too).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .calibration import (
    CalibrationArtifact,
    Calibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    SplitConformalGate,
    sigmoid,
)

_EPS = 1e-4


def as_label(value: object) -> int:
    """Normalize a label to 1 (AI) or 0 (human)."""
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"ai", "1", "machine", "generated", "synthetic"}:
            return 1
        if v in {"human", "0", "real"}:
            return 0
        raise ValueError(f"unrecognized label: {value!r}")
    return 1 if int(value) == 1 else 0  # type: ignore[call-overload]


def fit_platt(
    log_odds: Sequence[float], labels: Sequence[int], *, iterations: int = 100, ridge: float = 1e-6
) -> PlattCalibrator:
    """Fit ``p = sigmoid(a*L + b)`` by Newton-Raphson on the logistic negative log-likelihood."""
    xs = [float(v) for v in log_odds]
    ys = [as_label(v) for v in labels]
    a, b = 1.0, 0.0
    for _ in range(iterations):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for x, y in zip(xs, ys, strict=True):
            p = sigmoid(a * x + b)
            r = p - y
            w = max(p * (1.0 - p), 1e-9)
            g0 += r * x
            g1 += r
            h00 += w * x * x
            h01 += w * x
            h11 += w
        h00 += ridge
        h11 += ridge
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            break
        da = (h11 * g0 - h01 * g1) / det
        db = (h00 * g1 - h01 * g0) / det
        a -= da
        b -= db
        if abs(da) < 1e-9 and abs(db) < 1e-9:
            break
    residuals = [y - sigmoid(a * x + b) for x, y in zip(xs, ys, strict=True)]
    return PlattCalibrator(a=a, b=b, ci_halfwidth=_std(residuals) * 2.0 + 0.5)


def _pava(values: list[float], weights: list[float]) -> list[float]:
    """Pool-adjacent-violators: the non-decreasing least-squares fit of ``values``."""
    blocks: list[list[float]] = []  # each: [weighted_sum, weight, n_items]
    for value, weight in zip(values, weights, strict=True):
        block = [value * weight, weight, 1.0]
        while blocks and blocks[-1][0] / blocks[-1][1] >= block[0] / block[1]:
            prev = blocks.pop()
            block = [block[0] + prev[0], block[1] + prev[1], block[2] + prev[2]]
        blocks.append(block)
    fitted: list[float] = []
    for wsum, weight, n_items in blocks:
        fitted.extend([wsum / weight] * round(n_items))
    return fitted


def fit_isotonic(log_odds: Sequence[float], labels: Sequence[int]) -> IsotonicCalibrator:
    """Fit a monotone calibration table via PAVA over labels sorted by log-odds."""
    paired = sorted(zip((float(v) for v in log_odds), (as_label(v) for v in labels), strict=True))
    if not paired:
        raise ValueError("fit_isotonic requires at least one (log_odds, label) example")
    # Average duplicate log-odds before PAVA so the table has unique, sorted breakpoints.
    grouped: dict[float, list[int]] = {}
    for x, y in paired:
        grouped.setdefault(x, []).append(y)
    xs = sorted(grouped)
    means = [sum(grouped[x]) / len(grouped[x]) for x in xs]
    weights = [float(len(grouped[x])) for x in xs]
    fitted = _pava(means, weights)
    ys = [min(max(v, _EPS), 1.0 - _EPS) for v in fitted]
    halfwidth = max((xs[-1] - xs[0]) / 10.0, 0.5) if len(xs) >= 2 else 1.0
    return IsotonicCalibrator(xs=tuple(xs), ys=tuple(ys), ci_halfwidth=halfwidth)


def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """The split-conformal threshold: the ceil((n+1)(1-alpha))-th smallest score (inf if it
    exceeds n, meaning "include this class for every input")."""
    n = len(scores)
    if n == 0:
        return math.inf
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return math.inf
    return sorted(scores)[k - 1]


def fit_conformal_gate(
    log_odds: Sequence[float], labels: Sequence[int], *, alpha: float = 0.1
) -> SplitConformalGate:
    """Fit class-conditional conformal thresholds (nonconformity: -L for ai, +L for human)."""
    xs = [float(v) for v in log_odds]
    ys = [as_label(v) for v in labels]
    ai_scores = [-x for x, y in zip(xs, ys, strict=True) if y == 1]
    human_scores = [x for x, y in zip(xs, ys, strict=True) if y == 0]
    return SplitConformalGate(
        q_ai=conformal_quantile(ai_scores, alpha),
        q_human=conformal_quantile(human_scores, alpha),
        alpha=alpha,
    )


def fit_artifact(
    log_odds: Sequence[float],
    labels: Sequence[int],
    *,
    method: str = "isotonic",
    alpha: float = 0.1,
    prior_ai: float = 0.5,
    meta: dict[str, object] | None = None,
) -> CalibrationArtifact:
    """Fit a calibrator (``method`` = isotonic | platt) plus a conformal gate at level ``alpha``.

    ``prior_ai`` is the prior the ``log_odds`` were computed at; it is recorded so inference can
    lock to the same calibrated scale. Both classes must be present — calibrating a detector on a
    one-class corpus produces a gate that emits confident false 'ai'/'human' decisions.
    """
    ys = [as_label(v) for v in labels]
    n_ai = sum(ys)
    if n_ai == 0 or n_ai == len(ys):
        raise ValueError(
            f"calibration corpus is single-class ({n_ai} AI / {len(ys) - n_ai} human); "
            "fitting needs both classes"
        )
    calibrator: Calibrator
    if method == "isotonic":
        calibrator = fit_isotonic(log_odds, ys)
    elif method == "platt":
        calibrator = fit_platt(log_odds, ys)
    else:
        raise ValueError(f"unknown calibration method: {method!r}")
    gate = fit_conformal_gate(log_odds, ys, alpha=alpha)
    info: dict[str, object] = {
        "method": method,
        "alpha": alpha,
        "prior_ai": prior_ai,
        "n_calibration": len(ys),
        "n_ai": n_ai,
    }
    if meta:
        info.update(meta)
    return CalibrationArtifact(calibrator=calibrator, gate=gate, meta=info)


def _std(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
