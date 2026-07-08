"""Calibration + abstention: turn a fused log-odds into an honest probability and decision.

M0 ships deliberately simple placeholders (identity calibration + a log-odds dead-band gate)
so the engine runs with zero ML dependencies. The real work lands in M2:

- ``Calibrator`` → isotonic / temperature scaling fit on labeled data (target ECE < 0.05).
- ``ConformalGate`` → split conformal prediction with a distribution-free coverage guarantee.

Both are Protocols so the M2 implementations drop in without touching ``fusion.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .types import Decision


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


class Calibrator(Protocol):
    """Maps a fused log-odds to a calibrated probability + an interval."""

    @property
    def version(self) -> str: ...

    def probability(self, log_odds: float) -> float: ...

    def interval(self, log_odds: float) -> tuple[float, float]: ...


class ConformalGate(Protocol):
    """Decides ai / human / abstain from a fused log-odds at a target error rate ``alpha``."""

    @property
    def version(self) -> str: ...

    def decide(self, log_odds: float, alpha: float) -> Decision: ...


@dataclass(frozen=True)
class IdentityCalibrator:
    """M0 placeholder: ``p = sigmoid(L)``, with a fixed-width interval in log-odds space.

    Not calibrated in any real sense — replaced by a fitted isotonic/temperature model in M2.
    """

    ci_margin: float = 1.0
    version: str = "identity-v0"

    def probability(self, log_odds: float) -> float:
        return sigmoid(log_odds)

    def interval(self, log_odds: float) -> tuple[float, float]:
        return (sigmoid(log_odds - self.ci_margin), sigmoid(log_odds + self.ci_margin))


@dataclass(frozen=True)
class DeadbandGate:
    """M0 placeholder: abstain when the fused log-odds sits inside a dead-band around zero.

    A stand-in for real conformal abstention — same interface, honest behavior (it abstains
    on weak evidence), but no coverage guarantee yet. Replaced in M2.
    """

    deadband: float = 1.0
    version: str = "deadband-v0"

    def decide(self, log_odds: float, alpha: float) -> Decision:
        if abs(log_odds) < self.deadband:
            return "abstain"
        return "ai" if log_odds > 0.0 else "human"
