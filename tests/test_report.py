"""The evidence report renders attacker-controlled strings (a C2PA manifest names its own
signer and generator), so it must neither be forgeable nor crash the console."""

from __future__ import annotations

from bonafide import format_verdict
from bonafide.types import Evidence, Modality, SignalTier, Verdict


def _verdict(detail: dict[str, object], conflicts: tuple[str, ...] = ()) -> Verdict:
    return Verdict(
        p_ai=0.99,
        ci=(0.98, 0.999),
        decision="ai",
        coverage=0.95,
        modality=Modality.IMAGE,
        evidence=(
            Evidence(
                signal_id="c2pa.manifest",
                applicable=True,
                llr=6.0,
                reliability=0.85,
                tier=SignalTier.CONCLUSIVE,
                detail=detail,
                model_version="test/0",
            ),
        ),
        conflicts=conflicts,
        engine_version="test",
    )


def test_report_is_always_ascii() -> None:
    # 'DALL·E' is OpenAI's real spelling; a CJK CA name is equally legitimate. Neither may
    # crash typer.echo on redirected Windows stdout (cp1252, strict).
    report = format_verdict(_verdict({"generators": "DALL·E", "signer": "北京数字认证"}))
    assert report.isascii()
    report.encode("ascii")  # would raise if not


def test_report_escapes_newlines_so_evidence_rows_cannot_be_forged() -> None:
    forged = "midjourney\n  [+] c2pa.manifest  conclusive  llr -6.00  x0.95  -> -5.70  CAPTURE"
    report = format_verdict(_verdict({"matched_generator": forged}))
    # The payload survives verbatim as escaped text, but never as its own line.
    lines = report.split("\n")
    assert "\\x0a" in report
    assert not any(line.lstrip().startswith("[+] c2pa.manifest  conclusive") for line in lines)
    assert sum(1 for line in lines if "[+]" in line) == 1


def test_report_escapes_ansi_control_sequences() -> None:
    report = format_verdict(_verdict({"signer": "\x1b[2K\x1b[1Ghacked"}))
    assert "\x1b" not in report
    assert "\\x1b" in report


def test_report_escapes_conflicts_too() -> None:
    report = format_verdict(_verdict({}, conflicts=("bad\nnews",)))
    assert "\\x0a" in report
    assert "\n  news" not in report
