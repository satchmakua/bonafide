"""Core domain types — the contract the whole engine is built around (DESIGN.md §4).

Two rules make everything else work:

1. A detector never returns a bare boolean or an uncalibrated score. It returns an
   ``Evidence`` record: a log-likelihood-ratio toward AI plus a reliability weight.
2. A ``Verdict`` lists *every* signal that ran — including the ones that didn't
   apply — so the answer is always fully explainable and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Modality(StrEnum):
    """The kind of media an input is. Text + image ship in v1; audio/video are planned."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class SignalTier(StrEnum):
    """How much a signal is trusted, in principle. Drives default reliability + conflict logic."""

    CONCLUSIVE = "conclusive"  # cryptographic provenance, verified watermark
    STRONG = "strong"  # supervised detector, in-distribution
    WEAK = "weak"  # zero-shot / heuristic / stylometric
    CONTEXT = "context"  # metadata hints — never decisive


Decision = Literal["ai", "human", "abstain"]


class Evidence(BaseModel):
    """One signal's contribution toward the verdict.

    ``llr`` is a log-likelihood-ratio: ``log[ p(evidence | AI) / p(evidence | human) ]``.
    Positive leans AI, negative leans human, zero is neutral. ``reliability`` (0..1) is how
    much we trust this signal *for this input* — the out-of-distribution guard drives it to 0.
    """

    model_config = ConfigDict(frozen=True)

    signal_id: str
    applicable: bool
    llr: float = 0.0
    reliability: float = Field(default=0.0, ge=0.0, le=1.0)
    tier: SignalTier = SignalTier.WEAK
    detail: dict[str, Any] = Field(default_factory=dict)
    model_version: str = "unversioned"

    @property
    def contribution(self) -> float:
        """Signed weight this signal adds to the fused log-odds (0 when not applicable)."""
        return self.reliability * self.llr if self.applicable else 0.0


class Verdict(BaseModel):
    """The fused, calibrated answer — plus the full evidence trail behind it."""

    model_config = ConfigDict(frozen=True)

    p_ai: float
    ci: tuple[float, float]
    decision: Decision
    coverage: float
    modality: Modality
    evidence: tuple[Evidence, ...]
    engine_version: str
    conflicts: tuple[str, ...] = ()


@dataclass(frozen=True)
class InputContext:
    """The normalized input handed to signals. Internal carrier — not serialized."""

    modality: Modality
    text: str | None = None
    data: bytes | None = None
    path: str | None = None
    mime: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
