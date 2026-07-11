"""The conformal guarantee, checked empirically: over many independent calibration/test draws,
per-class coverage must meet the target 1 - alpha. This is the property that lets Bedrock claim
honest abstention rather than hand-waving."""

from __future__ import annotations

import random

from bedrock.training import fit_conformal_gate


def _draw(rng: random.Random, n: int, mu: float, sigma: float) -> tuple[list[float], list[int]]:
    xs, ys = [], []
    for _ in range(n):
        ai = rng.random() < 0.5
        xs.append(rng.gauss(mu if ai else -mu, sigma))
        ys.append(1 if ai else 0)
    return xs, ys


def _mean_coverage(
    mu: float, sigma: float, alpha: float, trials: int
) -> tuple[float, float, float]:
    cov_ai = cov_human = abstain = 0.0
    for seed in range(trials):
        rng = random.Random(seed)
        cx, cy = _draw(rng, 600, mu, sigma)
        gate = fit_conformal_gate(cx, cy, alpha=alpha)
        tx, ty = _draw(rng, 600, mu, sigma)
        ai_hits = ai_n = hu_hits = hu_n = abst = 0
        for x, y in zip(tx, ty, strict=True):
            members = gate.predict_set(x)
            if len(members) != 1:
                abst += 1
            if y == 1:
                ai_n += 1
                ai_hits += "ai" in members
            else:
                hu_n += 1
                hu_hits += "human" in members
        cov_ai += ai_hits / ai_n
        cov_human += hu_hits / hu_n
        abstain += abst / len(tx)
    return cov_ai / trials, cov_human / trials, abstain / trials


def test_marginal_coverage_meets_the_guarantee() -> None:
    for alpha in (0.05, 0.1, 0.2):
        cov_ai, cov_human, _ = _mean_coverage(mu=1.2, sigma=1.5, alpha=alpha, trials=120)
        target = 1.0 - alpha
        # The split-conformal guarantee is coverage >= 1 - alpha (in expectation over the draw);
        # allow a small Monte-Carlo slack below target.
        assert cov_ai >= target - 0.02, (alpha, cov_ai)
        assert cov_human >= target - 0.02, (alpha, cov_human)
        # And it should not be wildly over-conservative (that would mean never deciding).
        assert cov_ai <= target + 0.08


def test_overlapping_classes_force_more_abstention_than_separated_ones() -> None:
    _, _, abstain_overlap = _mean_coverage(mu=0.4, sigma=2.0, alpha=0.1, trials=60)
    _, _, abstain_separated = _mean_coverage(mu=3.0, sigma=1.0, alpha=0.1, trials=60)
    assert abstain_overlap > abstain_separated
    assert abstain_overlap > 0.05  # genuine uncertainty -> the gate actually abstains
