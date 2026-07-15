"""Render a Verdict as a human-readable evidence report.

M0 uses a deterministic, local template (no API key, no network). The Claude-backed narrator
that turns this structure into fluent prose lands in M4 (the ``explain`` extra) — it reads the
Verdict, it never votes in it.

**Everything dynamic here is untrusted.** A C2PA manifest names its own signer and generator,
so those strings are attacker-controlled and flow straight into the evidence rows. They are
escaped before interpolation: a raw newline would otherwise forge a fabricated evidence row
that a human reviewer reads as genuine, and an ANSI escape could erase the real verdict line.
Output is forced to ASCII, both for the Windows-console constraint and because a non-ASCII
byte in a manifest would otherwise crash the CLI on redirected stdout (cp1252, strict).
"""

from __future__ import annotations

import re

from .types import Verdict

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def _safe(value: object) -> str:
    """Escape control characters (newlines, ANSI, NUL) in an untrusted value."""
    return _CONTROL.sub(lambda m: f"\\x{ord(m.group()):02x}", str(value))


def _ascii(text: str) -> str:
    """Force ASCII losslessly — 'DALL·E' renders as 'DALL\\xb7E' rather than crashing."""
    return text.encode("ascii", "backslashreplace").decode("ascii")


def format_verdict(verdict: Verdict) -> str:
    lo, hi = verdict.ci
    lines = [
        f"Bonafide verdict: {verdict.p_ai * 100:.0f}% AI-generated "
        f"(CI {lo * 100:.0f}-{hi * 100:.0f}%)  |  decision: {verdict.decision.upper()}",
        f"modality: {verdict.modality.value}  |  engine: {verdict.engine_version}  "
        f"|  coverage: {verdict.coverage * 100:.0f}%",
        "",
        "Evidence:",
    ]
    for e in verdict.evidence:
        if e.applicable:
            detail = ", ".join(f"{k}={_safe(v)}" for k, v in e.detail.items())
            row = (
                f"  [+] {e.signal_id:<20} {e.tier.value:<10} "
                f"llr {e.llr:+.2f}  x{e.reliability:.2f}  -> {e.contribution:+.2f}"
            )
            lines.append(row + (f"   ({detail})" if detail else ""))
        else:
            lines.append(f"  [-] {e.signal_id:<20} not applicable")
    lines.append("")
    conflicts = "; ".join(_safe(c) for c in verdict.conflicts) if verdict.conflicts else "none"
    lines.append(f"Conflicts: {conflicts}")
    return _ascii("\n".join(lines))
