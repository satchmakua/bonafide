"""Render a Verdict as a human-readable evidence report.

M0 uses a deterministic, local template (no API key, no network). The Claude-backed narrator
that turns this structure into fluent prose lands in M4 (the ``explain`` extra) — it reads the
Verdict, it never votes in it. Output is ASCII-only so it prints cleanly on any terminal.
"""

from __future__ import annotations

from .types import Verdict


def format_verdict(verdict: Verdict) -> str:
    lo, hi = verdict.ci
    lines = [
        f"Bedrock verdict: {verdict.p_ai * 100:.0f}% AI-generated "
        f"(CI {lo * 100:.0f}-{hi * 100:.0f}%)  |  decision: {verdict.decision.upper()}",
        f"modality: {verdict.modality.value}  |  engine: {verdict.engine_version}  "
        f"|  coverage: {verdict.coverage * 100:.0f}%",
        "",
        "Evidence:",
    ]
    for e in verdict.evidence:
        if e.applicable:
            detail = ", ".join(f"{k}={v}" for k, v in e.detail.items())
            row = (
                f"  [+] {e.signal_id:<20} {e.tier.value:<10} "
                f"llr {e.llr:+.2f}  x{e.reliability:.2f}  -> {e.contribution:+.2f}"
            )
            lines.append(row + (f"   ({detail})" if detail else ""))
        else:
            lines.append(f"  [-] {e.signal_id:<20} not applicable")
    lines.append("")
    lines.append("Conflicts: " + ("; ".join(verdict.conflicts) if verdict.conflicts else "none"))
    return "\n".join(lines)
