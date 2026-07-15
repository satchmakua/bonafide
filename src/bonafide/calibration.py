"""Calibration + abstention: turn a fused log-odds into an honest probability and decision.

This module is **inference only** and pure-Python (no numpy) so it stays a zero-heavy-dep core
that ``fusion.py`` can import. Fitting the calibrators/gates from labeled data lives in
``training.py`` (the ``[calibrate]`` extra); the fitted objects serialize to JSON and load back
here. Everything is a monotone function of the fused log-odds ``L``:

- ``Calibrator``    maps ``L`` to a calibrated probability (identity/sigmoid, Platt, or isotonic).
- ``ConformalGate`` decides ai / human / abstain (a dead-band, or split-conformal with a
  distribution-free, class-conditional coverage guarantee).

Conformal works on ANY nonconformity score, so we score directly on ``L`` (``s_ai = -L``,
``s_human = +L``) — no dependence on the calibrator, which keeps the two orthogonal.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ._version import __version__
from .types import Decision

INF = float("inf")


def logit(p: float) -> float:
    """Inverse sigmoid, clamped away from 0/1 to keep the log-odds finite."""
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    """Numerically stable logistic function."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _encode_threshold(value: float) -> float | None:
    """+inf ('always include this class') -> JSON null, so the artifact stays valid JSON.

    ``json.dumps`` would otherwise emit the bare token ``Infinity``, which Python accepts but is
    invalid per the JSON spec and rejected by strict parsers (jq, other languages, validators).
    """
    return None if math.isinf(value) else value


def _decode_threshold(value: float | None) -> float:
    return math.inf if value is None else float(value)


def _interp(x: float, xs: tuple[float, ...], ys: tuple[float, ...]) -> float:
    """Piecewise-linear interpolation over a sorted table, clamped at both ends."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_right(xs, x)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    if x1 == x0:
        return y1
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


class Calibrator(Protocol):
    """Maps a fused log-odds to a calibrated probability + an interval."""

    @property
    def version(self) -> str: ...

    def probability(self, log_odds: float) -> float: ...

    def interval(self, log_odds: float) -> tuple[float, float]: ...

    def to_dict(self) -> dict[str, Any]: ...


class ConformalGate(Protocol):
    """Decides ai / human / abstain from a fused log-odds at a target error rate ``alpha``."""

    @property
    def version(self) -> str: ...

    def decide(self, log_odds: float, alpha: float) -> Decision: ...

    def coverage(self, alpha: float) -> float:
        """The coverage the gate actually delivers — from the gate, never the runtime alpha, so
        a reported guarantee can never overstate what the gate enforces."""
        ...

    def to_dict(self) -> dict[str, Any]: ...


# --- Calibrators ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IdentityCalibrator:
    """Default: ``p = sigmoid(L)``, with a fixed-width interval in log-odds space.

    Honest but uncalibrated — used until a fitted calibrator is available. The engine falls
    back to this whenever no calibration artifact is loaded.
    """

    ci_margin: float = 1.0
    version: str = "identity-v0"

    def probability(self, log_odds: float) -> float:
        return sigmoid(log_odds)

    def interval(self, log_odds: float) -> tuple[float, float]:
        return (sigmoid(log_odds - self.ci_margin), sigmoid(log_odds + self.ci_margin))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "identity", "ci_margin": self.ci_margin, "version": self.version}


@dataclass(frozen=True)
class PlattCalibrator:
    """Temperature/Platt scaling: ``p = sigmoid(a * L + b)`` with a scalar slope + bias.

    Fit by ``training.fit_platt``. ``ci_halfwidth`` is a residual spread (in the scaled-logit
    space) used to report an interval around the point estimate.
    """

    a: float
    b: float
    ci_halfwidth: float = 1.0
    version: str = "platt-v1"

    def probability(self, log_odds: float) -> float:
        return sigmoid(self.a * log_odds + self.b)

    def interval(self, log_odds: float) -> tuple[float, float]:
        center = self.a * log_odds + self.b
        return (sigmoid(center - self.ci_halfwidth), sigmoid(center + self.ci_halfwidth))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "platt",
            "a": self.a,
            "b": self.b,
            "ci_halfwidth": self.ci_halfwidth,
            "version": self.version,
        }


@dataclass(frozen=True)
class IsotonicCalibrator:
    """Isotonic (monotone, non-parametric) calibration: a fitted step/interpolation table.

    ``xs`` are sorted log-odds breakpoints, ``ys`` the calibrated probabilities (non-decreasing).
    Fit by ``training.fit_isotonic`` (pool-adjacent-violators). Inference is a pure-Python
    interpolation, so no numpy is needed at runtime.
    """

    xs: tuple[float, ...]
    ys: tuple[float, ...]
    ci_halfwidth: float = 1.0
    version: str = "isotonic-v1"

    def probability(self, log_odds: float) -> float:
        return _interp(log_odds, self.xs, self.ys)

    def interval(self, log_odds: float) -> tuple[float, float]:
        return (
            _interp(log_odds - self.ci_halfwidth, self.xs, self.ys),
            _interp(log_odds + self.ci_halfwidth, self.xs, self.ys),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "isotonic",
            "xs": list(self.xs),
            "ys": list(self.ys),
            "ci_halfwidth": self.ci_halfwidth,
            "version": self.version,
        }


def calibrator_from_dict(data: dict[str, Any]) -> Calibrator:
    kind = data["kind"]
    if kind == "identity":
        return IdentityCalibrator(ci_margin=data["ci_margin"], version=data["version"])
    if kind == "platt":
        return PlattCalibrator(
            a=data["a"], b=data["b"], ci_halfwidth=data["ci_halfwidth"], version=data["version"]
        )
    if kind == "isotonic":
        return IsotonicCalibrator(
            xs=tuple(data["xs"]),
            ys=tuple(data["ys"]),
            ci_halfwidth=data["ci_halfwidth"],
            version=data["version"],
        )
    raise ValueError(f"unknown calibrator kind: {kind!r}")


# --- Abstention gates ----------------------------------------------------------------------


@dataclass(frozen=True)
class DeadbandGate:
    """Default: abstain when the fused log-odds sits inside a dead-band around zero.

    Honest (it abstains on weak evidence) but with no coverage guarantee. Replaced by
    ``SplitConformalGate`` once a calibration set is available.
    """

    deadband: float = 1.0
    version: str = "deadband-v0"

    def decide(self, log_odds: float, alpha: float) -> Decision:
        if abs(log_odds) < self.deadband:
            return "abstain"
        return "ai" if log_odds > 0.0 else "human"

    def coverage(self, alpha: float) -> float:
        return 1.0 - alpha  # a nominal target only — this gate carries no formal guarantee

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "deadband", "deadband": self.deadband, "version": self.version}


@dataclass(frozen=True)
class SplitConformalGate:
    """Split-conformal abstention with a class-conditional (Mondrian) coverage guarantee.

    Fit by ``training.fit_conformal_gate`` at a target error rate ``alpha`` (baked in). Scores
    are nonconformities on the log-odds: ``s_ai = -L`` and ``s_human = +L``. A class is included
    in the prediction set when its score is at or below that class's fitted quantile:

        ai in C(x)     iff  -L <= q_ai      (i.e. L >= -q_ai)
        human in C(x)  iff  +L <= q_human   (i.e. L <=  q_human)

    A singleton set yields that label; an ambiguous ({ai, human}) or empty set yields abstain.
    Guarantee (exchangeability): for each class, P(true label in C) >= 1 - alpha.
    """

    q_ai: float
    q_human: float
    alpha: float
    version: str = "split-conformal-v1"

    def predict_set(self, log_odds: float) -> frozenset[str]:
        """The conformal prediction set — the classes whose nonconformity is within tolerance."""
        members = set()
        if -log_odds <= self.q_ai:
            members.add("ai")
        if log_odds <= self.q_human:
            members.add("human")
        return frozenset(members)

    def decide(self, log_odds: float, alpha: float) -> Decision:
        members = self.predict_set(log_odds)
        if members == {"ai"}:
            return "ai"
        if members == {"human"}:
            return "human"
        return "abstain"  # both classes plausible, or neither

    def coverage(self, alpha: float) -> float:
        # The guarantee is baked into the fitted thresholds at self.alpha; the runtime alpha is
        # irrelevant (decide ignores it too), so reporting 1 - self.alpha can never overstate it.
        return 1.0 - self.alpha

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "split-conformal",
            "q_ai": _encode_threshold(self.q_ai),
            "q_human": _encode_threshold(self.q_human),
            "alpha": self.alpha,
            "version": self.version,
        }


def gate_from_dict(data: dict[str, Any]) -> ConformalGate:
    kind = data["kind"]
    if kind == "deadband":
        return DeadbandGate(deadband=data["deadband"], version=data["version"])
    if kind == "split-conformal":
        return SplitConformalGate(
            q_ai=_decode_threshold(data["q_ai"]),
            q_human=_decode_threshold(data["q_human"]),
            alpha=data["alpha"],
            version=data["version"],
        )
    raise ValueError(f"unknown gate kind: {kind!r}")


# --- Artifact: a calibrator + gate + provenance, serialized to one JSON file ---------------


@dataclass(frozen=True)
class CalibrationArtifact:
    """A fitted calibrator + abstention gate, plus the provenance to make a Verdict auditable.

    Saved to / loaded from a single JSON file. The ``fingerprint`` is a content hash of the
    fitted parameters, stamped into the engine version so two runs with different calibration
    can never look identical.
    """

    calibrator: Calibrator
    gate: ConformalGate
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def fit_prior(self) -> float | None:
        """The prior_ai the log-odds were calibrated at, if recorded — locked in at inference."""
        value = self.meta.get("prior_ai")
        return float(value) if value is not None else None

    @property
    def fingerprint(self) -> str:
        # Prior is included: the same calibrator/gate fit at a different prior is a different
        # calibration (the log-odds scale shifts), so it must get a distinct fingerprint.
        payload = json.dumps(
            {
                "calibrator": self.calibrator.to_dict(),
                "gate": self.gate.to_dict(),
                "prior_ai": self.fit_prior,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    @property
    def version(self) -> str:
        return f"{self.calibrator.version}+{self.gate.version}+fit/{self.fingerprint}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "bonafide_version": __version__,
            "calibrator": self.calibrator.to_dict(),
            "gate": self.gate.to_dict(),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationArtifact:
        return cls(
            calibrator=calibrator_from_dict(data["calibrator"]),
            gate=gate_from_dict(data["gate"]),
            meta=data.get("meta", {}),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CalibrationArtifact:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def default_artifact() -> CalibrationArtifact:
    """The uncalibrated fallback used until a fitted artifact is trained and shipped."""
    return CalibrationArtifact(
        calibrator=IdentityCalibrator(),
        gate=DeadbandGate(),
        meta={"note": "uncalibrated default (identity + dead-band)"},
    )
