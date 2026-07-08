"""The ``Signal`` port and its registry (hexagonal architecture, DESIGN.md §5).

The core depends only on this protocol. Every detector — provenance checker, watermark reader,
ML model, third-party API — is an adapter that satisfies it. A new modality or detector is a
new adapter plus a registry entry; the fusion core never changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..types import Evidence, InputContext, Modality, SignalTier


class Signal(Protocol):
    """A detector: a cheap ``applies_to`` gate, then ``analyze`` returns Evidence."""

    id: str
    modalities: frozenset[Modality]
    tier: SignalTier
    version: str

    def applies_to(self, ctx: InputContext) -> bool: ...

    def analyze(self, ctx: InputContext) -> Evidence: ...


def not_applicable(signal: Signal, note: str = "") -> Evidence:
    """A zero-contribution Evidence record for a signal that didn't apply.

    We still emit it (rather than dropping it) so the verdict lists *every* signal that ran —
    transparency is the product.
    """
    return Evidence(
        signal_id=signal.id,
        applicable=False,
        tier=signal.tier,
        model_version=signal.version,
        detail={"note": note} if note else {},
    )


@dataclass
class SignalRegistry:
    """Holds the registered signals and hands out the ones relevant to a modality."""

    signals: list[Signal] = field(default_factory=list)

    def register(self, signal: Signal) -> Signal:
        self.signals.append(signal)
        return signal

    def for_modality(self, modality: Modality) -> list[Signal]:
        return [s for s in self.signals if modality in s.modalities]
