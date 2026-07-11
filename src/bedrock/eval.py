"""Evaluation harness — the discipline that keeps calibration honest (DESIGN.md §6.8).

Runs a labeled corpus through the engine and reports the metrics that matter for a detector
that must not falsely accuse: calibration (ECE, Brier), separation (AUROC), the human false-
positive rate at the operating point (with a subgroup slice — e.g. non-native English), the
abstention rate, and the empirical conformal coverage vs. its guarantee.

Pure stdlib. Corpus format is JSONL: ``{"text": "...", "label": "ai"|"human", "group": "..."}``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .calibration import CalibrationArtifact, SplitConformalGate, default_artifact
from .fusion import raw_log_odds
from .signals.base import SignalRegistry
from .training import as_label
from .types import Decision


@dataclass(frozen=True)
class Example:
    text: str
    label: int  # 1 = AI, 0 = human
    group: str = "all"


@dataclass(frozen=True)
class Scored:
    log_odds: float | None  # None when no signal applied to this input
    label: int
    group: str


def load_corpus(path: str | Path) -> list[Example]:
    examples: list[Example] = []
    # utf-8-sig strips a leading BOM (Windows Notepad / Excel / PowerShell default) and decodes
    # BOM-less UTF-8 identically, so a valid corpus doesn't crash on line 1.
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        examples.append(
            Example(text=row["text"], label=as_label(row["label"]), group=row.get("group", "all"))
        )
    return examples


def score_corpus(
    examples: Sequence[Example],
    *,
    registry: SignalRegistry | None = None,
    prior_ai: float = 0.5,
) -> list[Scored]:
    """Run the detectors over each example and keep the raw fused log-odds (pre-calibration)."""
    from .engine import default_registry
    from .ingest import context_from_text

    reg = registry or default_registry()
    scored: list[Scored] = []
    for ex in examples:
        ctx = context_from_text(ex.text)
        evidence = [s.analyze(ctx) for s in reg.for_modality(ctx.modality) if s.applies_to(ctx)]
        scored.append(Scored(raw_log_odds(evidence, prior_ai=prior_ai), ex.label, ex.group))
    return scored


# --- metrics (pure functions over (probability, label) pairs) ------------------------------


def ece(probs: Sequence[float], labels: Sequence[int], *, bins: int = 10) -> float:
    """Expected Calibration Error: |confidence - accuracy| averaged over confidence bins."""
    n = len(probs)
    if n == 0:
        return math.nan
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [
            i
            for i in range(n)
            if (max(probs[i], 1 - probs[i]) > lo or b == 0) and max(probs[i], 1 - probs[i]) <= hi
        ]
        if not idx:
            continue
        conf = sum(max(probs[i], 1 - probs[i]) for i in idx) / len(idx)
        acc = sum(1 for i in idx if (probs[i] >= 0.5) == (labels[i] == 1)) / len(idx)
        total += (len(idx) / n) * abs(conf - acc)
    return total


def brier(probs: Sequence[float], labels: Sequence[int]) -> float:
    if not probs:
        return math.nan
    return sum((p - y) ** 2 for p, y in zip(probs, labels, strict=True)) / len(probs)


def auroc(probs: Sequence[float], labels: Sequence[int]) -> float:
    """Area under the ROC curve via average-rank (Mann-Whitney U), tie-aware."""
    data = sorted(zip(probs, labels, strict=True), key=lambda t: t[0])
    n = len(data)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and data[j][0] == data[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0  # average of the 1-based ranks i+1..j
        for k in range(i, j):
            ranks[k] = avg
        i = j
    n_pos = sum(1 for _, y in data if y == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.nan
    rank_sum_pos = sum(ranks[k] for k in range(n) if data[k][1] == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


@dataclass(frozen=True)
class EvalReport:
    n: int
    n_ai: int
    n_human: int
    n_scored: int  # a signal applied
    ece: float
    brier: float
    auroc: float
    fpr: float  # P(decision = ai | label = human), over ALL humans
    fnr: float  # P(decision = human | label = ai)
    abstention_rate: float
    coverage_ai: float  # empirical conformal coverage (nan if gate is non-conformal)
    coverage_human: float
    target_coverage: float
    fpr_by_group: dict[str, float] = field(default_factory=dict)
    artifact_version: str = ""


def evaluate(
    scored: Sequence[Scored], artifact: CalibrationArtifact, *, prior_ai: float = 0.5
) -> EvalReport:
    """Apply a fitted artifact to pre-scored examples and compute the report."""
    probs: list[float] = []
    prob_labels: list[int] = []
    decisions: list[tuple[Decision, int, str]] = []
    cov_ai_hits = cov_ai_n = cov_hu_hits = cov_hu_n = 0
    gate = artifact.gate
    gate_alpha = _gate_alpha(artifact)
    call_alpha = 0.1 if math.isnan(gate_alpha) else gate_alpha

    for s in scored:
        if s.log_odds is None:
            decisions.append(("abstain", s.label, s.group))
            continue
        probs.append(artifact.calibrator.probability(s.log_odds))
        prob_labels.append(s.label)
        decisions.append((gate.decide(s.log_odds, call_alpha), s.label, s.group))
        if isinstance(gate, SplitConformalGate):
            members = gate.predict_set(s.log_odds)
            if s.label == 1:
                cov_ai_n += 1
                cov_ai_hits += "ai" in members
            else:
                cov_hu_n += 1
                cov_hu_hits += "human" in members

    humans = [d for d in decisions if d[1] == 0]
    ais = [d for d in decisions if d[1] == 1]
    fpr = _rate(humans, "ai")
    groups = {g for _, _, g in decisions}
    fpr_by_group = {
        g: _rate([d for d in humans if d[2] == g], "ai") for g in sorted(groups)
    }

    target = 1.0 - gate_alpha
    return EvalReport(
        n=len(scored),
        n_ai=len(ais),
        n_human=len(humans),
        n_scored=len(probs),
        ece=ece(probs, prob_labels),
        brier=brier(probs, prob_labels),
        auroc=auroc(probs, prob_labels),
        fpr=fpr,
        fnr=_rate(ais, "human"),
        abstention_rate=sum(1 for d in decisions if d[0] == "abstain") / max(len(decisions), 1),
        coverage_ai=cov_ai_hits / cov_ai_n if cov_ai_n else math.nan,
        coverage_human=cov_hu_hits / cov_hu_n if cov_hu_n else math.nan,
        target_coverage=target,
        fpr_by_group=fpr_by_group,
        artifact_version=artifact.version,
    )


def _rate(rows: Sequence[tuple[Decision, int, str]], decision: Decision) -> float:
    if not rows:
        return math.nan
    return sum(1 for d in rows if d[0] == decision) / len(rows)


def _gate_alpha(artifact: CalibrationArtifact) -> float:
    gate = artifact.gate
    return gate.alpha if isinstance(gate, SplitConformalGate) else float("nan")


def format_report(report: EvalReport) -> str:
    def pct(x: float) -> str:
        return "n/a" if math.isnan(x) else f"{x * 100:.1f}%"

    lines = [
        f"Bedrock eval  |  artifact: {report.artifact_version}",
        f"  corpus: {report.n} examples ({report.n_ai} AI, {report.n_human} human); "
        f"{report.n_scored} scored, {report.n - report.n_scored} no-signal",
        f"  calibration:  ECE {pct(report.ece)}   Brier {report.brier:.4f}   "
        f"AUROC {'n/a' if math.isnan(report.auroc) else f'{report.auroc:.3f}'}",
        f"  decisions:    FPR {pct(report.fpr)}   FNR {pct(report.fnr)}   "
        f"abstain {pct(report.abstention_rate)}",
        f"  conformal:    coverage AI {pct(report.coverage_ai)} / "
        f"human {pct(report.coverage_human)}   (target {pct(report.target_coverage)})",
    ]
    if len(report.fpr_by_group) > 1:
        slices = "   ".join(f"{g}: {pct(v)}" for g, v in report.fpr_by_group.items())
        lines.append(f"  FPR by group: {slices}")
    return "\n".join(lines)


def run_eval(
    corpus_path: str | Path,
    *,
    artifact_path: str | Path | None = None,
    prior_ai: float = 0.5,
) -> EvalReport:
    artifact = (
        CalibrationArtifact.load(artifact_path) if artifact_path is not None else default_artifact()
    )
    # Score at the artifact's fit-time prior when it records one, so eval measures the calibrated
    # scale (matching how detection locks the prior); otherwise use the caller's prior.
    fit_prior = artifact.fit_prior
    effective_prior = fit_prior if fit_prior is not None else prior_ai
    scored = score_corpus(load_corpus(corpus_path), prior_ai=effective_prior)
    return evaluate(scored, artifact, prior_ai=effective_prior)
