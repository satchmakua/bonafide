# 2. Evidence fusion in log-odds space

- **Status:** Accepted
- **Date:** 2026-07-08

## Context

Bedrock's whole thesis (DESIGN.md §1, §4) is that a trustworthy verdict comes from
*fusing many signals of very different trust levels*, not from one model's opinion.
A valid C2PA signature from a trusted CA is near-conclusive; a zero-shot perplexity
score is a faint hint. The signals also arrive on incompatible scales and any given
one may not apply to a given input at all (no C2PA manifest present; text too short
for a detector).

Naive score-averaging — what thin aggregators (e.g. Eden AI) do — fails this:
three weak hints can outvote one cryptographic proof, and there's no principled place
to encode "how much do we trust this signal here?" We also must be able to **abstain**
and to **surface conflicts** rather than average them away.

## Decision

Every signal emits an **`Evidence`** record — never a raw boolean or bare score —
carrying a **log-likelihood-ratio (`llr`)** toward AI, a **`reliability`** weight in
`[0, 1]`, an `applicable` flag, and a `tier`. Fusion accumulates evidence in
**log-odds space**:

```
L = logit(prior_ai) + Σ_i  reliability_i · llr_i     # applicable signals only
```

Then a **calibrator** maps `L → p_ai` (identity/sigmoid in M0; isotonic/temperature
fit on labeled data in M2) and a **gate** decides `ai | human | abstain` (a log-odds
dead-band in M0; split-conformal prediction with a coverage guarantee in M2).

Key rules, enforced in `fusion.py`:
- `applicable = False` ⇒ the signal contributes **exactly 0** (absence never
  penalizes; "no C2PA manifest" ≠ "human").
- `reliability → 0` under an out-of-distribution flag ⇒ a shaky signal self-mutes
  instead of voting wrong.
- Conflicting **CONCLUSIVE** signals (opposite-sign, high-trust) force `abstain` and
  are reported in `conflicts` — honesty beats a coin-flip.
- A `Verdict` is a pure function of `(evidence, prior, calibrator, gate, alpha)` — so
  it is reproducible and auditable (and litigation-defensible).

## Consequences

- **Enables** the moat: a single conclusive signal dominates *correctly* (large
  `|llr|`), weak signals accumulate honestly, and one uniform code path calibrates
  every modality. New detectors/modalities plug in by emitting `Evidence` — no core
  change (ADR pairs with the hexagonal `Signal` port).
- **Requires** each adapter to express its output as a calibrated `llr` + reliability.
  This is more work than returning a score, and mis-specified `llr`s are a real risk —
  mitigated by the M2 calibration + evaluation harness (ECE, FPR, coverage).
- Independence between signals is assumed by additive log-odds; correlated signals
  (e.g. two perplexity detectors) will double-count. The M2 meta-classifier
  (stacked ensemble) is the planned upgrade that learns those correlations; the
  additive rule is the correct, transparent default until then.
