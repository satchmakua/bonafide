"""Evaluation metrics, checked against hand computation, plus the end-to-end fit/eval flow."""

from __future__ import annotations

import math
from pathlib import Path

from bedrock import CalibrationArtifact, detect_text
from bedrock.calibration import PlattCalibrator, SplitConformalGate
from bedrock.eval import (
    Scored,
    auroc,
    brier,
    ece,
    evaluate,
    format_report,
    load_corpus,
    run_eval,
    score_corpus,
)
from bedrock.training import fit_artifact


def test_ece_hand_value() -> None:
    # conf/acc per singleton bin: 0.6/0, 0.7/1, 0.8/1, 0.9/1 -> (0.6+0.3+0.2+0.1)/4 = 0.3
    probs = [0.9, 0.8, 0.3, 0.6]
    labels = [1, 1, 0, 0]
    assert math.isclose(ece(probs, labels), 0.3, abs_tol=1e-9)


def test_brier_hand_value() -> None:
    # ((0.9-1)^2 + (0.2-0)^2) / 2 = (0.01 + 0.04)/2 = 0.025
    assert math.isclose(brier([0.9, 0.2], [1, 0]), 0.025, abs_tol=1e-12)


def test_auroc_separation_and_ties() -> None:
    assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0  # perfect
    assert auroc([0.4, 0.6], [1, 0]) == 0.0  # perfectly wrong
    assert math.isclose(auroc([0.5, 0.5], [1, 0]), 0.5, abs_tol=1e-9)  # tie -> 0.5
    assert math.isnan(auroc([0.5, 0.6], [1, 1]))  # one class -> undefined


def test_decision_rates_and_coverage() -> None:
    # A gate that decides ai for L>0, human for L<0, abstains at exactly 0.
    art = CalibrationArtifact(
        calibrator=PlattCalibrator(a=1.0, b=0.0),
        gate=SplitConformalGate(q_ai=0.0, q_human=0.0, alpha=0.1),
    )
    scored = [
        Scored(2.0, 1, "en"),  # ai, decided ai   -> correct
        Scored(-2.0, 1, "es"),  # ai, decided human -> a false-negative + miscovered
        Scored(-2.0, 0, "en"),  # human, decided human
        Scored(3.0, 0, "es"),  # human, decided ai  -> a false-positive (the harm we track)
        Scored(None, 0, "en"),  # no signal -> abstain
    ]
    rep = evaluate(scored, art)
    assert rep.n == 5 and rep.n_scored == 4
    assert math.isclose(rep.fpr, 1 / 3)  # 1 of 3 humans decided "ai"
    assert math.isclose(rep.fnr, 1 / 2)  # 1 of 2 ais decided "human"
    assert math.isclose(rep.abstention_rate, 1 / 5)  # the no-signal example
    assert rep.fpr_by_group["es"] == 1.0 and rep.fpr_by_group["en"] == 0.0
    # Each misclassified point is also a coverage miss for its true class: 1 of 2 per class.
    assert math.isclose(rep.coverage_ai, 0.5) and math.isclose(rep.coverage_human, 0.5)
    assert "ECE" in format_report(rep)


def test_end_to_end_fit_eval_and_detect(tmp_path: Path) -> None:
    # A tiny corpus where AI text is lexically repetitive (low type-token ratio) and human text
    # is diverse — enough for the lexical signal to separate and the calibrator to fit.
    corpus = tmp_path / "corpus.jsonl"
    lines = []
    for i in range(40):
        lines.append(
            f'{{"text": "{("the same words over and over " * 6).strip()} {i}", "label": "ai"}}'
        )
        varied = " ".join(f"word{i}x{j}" for j in range(30))
        lines.append(f'{{"text": "{varied}", "label": "human"}}')
    corpus.write_text("\n".join(lines), encoding="utf-8")

    examples = load_corpus(corpus)
    assert len(examples) == 80

    scored = score_corpus(examples)
    labeled = [(s.log_odds, s.label) for s in scored if s.log_odds is not None]
    log_odds, labels = zip(*labeled, strict=True)
    artifact = fit_artifact(list(log_odds), list(labels), method="isotonic", alpha=0.1)

    path = tmp_path / "cal.json"
    artifact.save(path)
    loaded = CalibrationArtifact.load(path)

    # The fitted artifact flows into detection and is stamped into the verdict.
    verdict = detect_text("the same words over and over " * 6, calibration=loaded)
    assert loaded.version in verdict.engine_version
    assert 0.0 <= verdict.p_ai <= 1.0

    report = run_eval(corpus, artifact_path=path)
    assert report.n == 80
    assert not math.isnan(report.ece)


def test_bom_prefixed_corpus_loads(tmp_path: Path) -> None:
    # Windows Notepad / Excel / PowerShell write a UTF-8 BOM by default.
    corpus = tmp_path / "bom.jsonl"
    body = '{"text": "hello world", "label": "ai"}\n{"text": "goodbye", "label": "human"}'
    corpus.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))
    examples = load_corpus(corpus)
    assert len(examples) == 2 and examples[0].label == 1


def test_fitted_prior_is_locked_at_inference() -> None:
    # A calibrator fit at prior 0.5 must not be silently re-scaled by a caller's --prior; the
    # artifact's recorded prior wins, so the verdict stays on the calibrated scale.
    from bedrock.calibration import PlattCalibrator
    from bedrock.training import fit_conformal_gate

    xs = [float(i - 20) for i in range(40)]
    ys = [1 if x > 0 else 0 for x in xs]
    art = CalibrationArtifact(
        calibrator=PlattCalibrator(a=1.0, b=0.0),
        gate=fit_conformal_gate(xs, ys, alpha=0.1),
        meta={"prior_ai": 0.5},
    )
    v_default = detect_text("the same words over and over " * 6, calibration=art)
    v_biased = detect_text("the same words over and over " * 6, calibration=art, prior_ai=0.99)
    assert v_default.p_ai == v_biased.p_ai  # the caller's prior was ignored (locked to 0.5)
    # And the coverage reported is the gate's guarantee (1 - 0.1), not 1 - the detect alpha.
    assert v_default.coverage == 0.9
