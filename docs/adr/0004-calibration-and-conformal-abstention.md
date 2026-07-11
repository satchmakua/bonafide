# 4. Calibration and conformal abstention

- **Status:** Accepted
- **Date:** 2026-07-10

## Context

Bedrock's whole pitch is *honesty*: a probability you can trust and an explicit "I don't know",
instead of the confident binaries that made incumbent text detectors dangerous (5–12% false
positives on non-native English writers; §1 of DESIGN). Two things have to be real for that
claim to hold:

1. **Calibration** — when the engine says 80%, it should be wrong about 20% of the time.
2. **Abstention with a guarantee** — the "abstain" outcome must rest on a real coverage
   property, not a hand-tuned threshold.

DESIGN §3 tentatively named scikit-learn + MAPIE/crepes for this.

## Decision

**Hand-roll the calibration + conformal core in pure Python (stdlib only); drop sklearn/MAPIE.**
The algorithms are short, classic, and — most importantly — *hand-verifiable against their
definitions*, which this project values (cf. the log-odds fusion tests). It also keeps
`calibration.py` a zero-heavy-dependency core module that `fusion.py` can import, and keeps the
whole thesis runnable on a CPU with no ML stack.

- **Calibrators** (`calibration.py`, inference only): `PlattCalibrator` (`p = sigmoid(aL+b)`) and
  `IsotonicCalibrator` (a monotone interpolation table). Fitting lives in `training.py`:
  Newton-Raphson for Platt, pool-adjacent-violators (PAVA) for isotonic. Inference is pure
  Python (no numpy) so it runs inside the core.
- **Abstention** (`SplitConformalGate`): class-conditional (Mondrian) split conformal on the
  fused log-odds. Nonconformity `s_ai = -L`, `s_human = +L`; per-class threshold is the
  `ceil((n_g+1)(1-α))`-th smallest score. The prediction set is the classes within tolerance; a
  singleton decides, an ambiguous/empty set abstains. Guarantee (exchangeability):
  `P(true label ∈ C) ≥ 1-α` for each class. Verified empirically in
  `test_conformal_coverage.py` across many draws.
- Conformal scores directly on `L`, independent of the calibrator, so the two are orthogonal.
- **Artifact** (`CalibrationArtifact`): calibrator + gate + provenance in one JSON file,
  content-**fingerprinted** and stamped into `engine_version`, so a Verdict pins the exact
  calibration that produced it (the reproducibility hard constraint).

**Train/test discipline.** Both the calibration fit and the conformal guarantee assume the
calibration data is exchangeable with, and *not reused as*, the evaluation/production data.
Fitting and evaluating on the same corpus reports optimistically biased ECE/coverage. The eval
harness and `fit`/`eval` CLI therefore treat held-out evaluation as the honest path (the docs
and CLI say so); the coverage guarantee is a statement about *future* exchangeable inputs.

## Consequences

- The thesis (calibrated probabilities + guaranteed abstention) is real and testable on a
  laptop: isotonic cut ECE from 0.069 to 0.008 on synthetic scores; empirical coverage tracks
  1-α. Zero heavy dependencies.
- Isotonic is *weakly* monotone: it can pool a mis-ordered region into a tie, which legitimately
  *raises* AUROC (not a bug — a strictly-monotone map preserves AUROC; a flattening one can
  improve it by discarding uninformative ordering).
- Small calibration sets are a real hazard: if a class has too few points, its conformal
  threshold is +∞ ("always include"), which quietly makes the gate abstain-happy. This must be
  surfaced, not hidden (fit-time signal / eval visibility).
- We own the numerics (Platt convergence on separable data, PAVA correctness). Covered by tests;
  the tradeoff for dropping a maintained library is ours to carry.
- A trained, validated artifact is **not** shipped yet — the default stays uncalibrated
  (identity + dead-band) and honestly labels itself as such. A calibrator fit on a real
  benchmark (RAID) ships with the M2b detector ensemble.
