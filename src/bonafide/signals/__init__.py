"""Signal adapters — the plug-in detectors behind the ``Signal`` port (DESIGN.md §5, §6.1).

M0 ships one placeholder text signal (``LexicalHeuristicSignal``). Real adapters land per
milestone: C2PA provenance (M1); the text ensemble + SynthID-Text (M2); image forensics (M3).
"""

from __future__ import annotations

from .base import Signal, SignalRegistry, not_applicable
from .lexical import LexicalHeuristicSignal

__all__ = ["LexicalHeuristicSignal", "Signal", "SignalRegistry", "not_applicable"]
