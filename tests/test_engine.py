"""End-to-end engine tests: ingest -> route -> signals -> fuse."""

from __future__ import annotations

from bonafide import detect_bytes, detect_text
from bonafide.types import Modality

# 24 tokens — comfortably over the lexical signal's 20-token applicability floor.
LONG_TEXT = (
    "The quick brown fox jumps over the lazy dog while the old stone bridge "
    "quietly watches the river drift past the mossy banks below."
)


def test_detects_text_and_reports_the_lexical_signal() -> None:
    v = detect_text(LONG_TEXT)
    assert v.modality == Modality.TEXT
    lex = next(e for e in v.evidence if e.signal_id == "lexical-heuristic")
    assert lex.applicable
    assert "type_token_ratio" in lex.detail


def test_short_text_makes_the_signal_not_applicable_and_abstains() -> None:
    v = detect_text("too short")
    lex = next(e for e in v.evidence if e.signal_id == "lexical-heuristic")
    assert not lex.applicable
    assert v.decision == "abstain"


def test_image_bytes_route_to_image_with_no_applicable_text_signals() -> None:
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    v = detect_bytes(png, path="sample.png")
    assert v.modality == Modality.IMAGE
    assert v.decision == "abstain"  # no image signals wired until M1/M3


def test_verdict_carries_a_reproducible_engine_version() -> None:
    v = detect_text(LONG_TEXT)
    assert v.engine_version.startswith("bonafide/")
    assert "fuse-logodds-v0" in v.engine_version
