"""Bedrock — a calibrated, provenance-first framework for detecting AI-generated media.

Public API:

    >>> from bedrock import detect
    >>> v = detect("some text or a file path")
    >>> v.decision, v.p_ai
"""

from __future__ import annotations

from ._version import __version__
from .engine import (
    default_registry,
    detect,
    detect_bytes,
    detect_file,
    detect_text,
    engine_version,
)
from .fusion import fuse
from .report import format_verdict
from .types import Decision, Evidence, InputContext, Modality, SignalTier, Verdict

__all__ = [
    "Decision",
    "Evidence",
    "InputContext",
    "Modality",
    "SignalTier",
    "Verdict",
    "__version__",
    "default_registry",
    "detect",
    "detect_bytes",
    "detect_file",
    "detect_text",
    "engine_version",
    "format_verdict",
    "fuse",
]
