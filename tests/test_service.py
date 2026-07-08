"""Smoke tests for the FastAPI service."""

from __future__ import annotations

from fastapi.testclient import TestClient

from bedrock.service.app import app

client = TestClient(app)

LONG_TEXT = (
    "The quick brown fox jumps over the lazy dog while the old stone bridge "
    "quietly watches the river drift past the mossy banks below."
)


def test_healthz() -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_signals_lists_the_lexical_signal() -> None:
    r = client.get("/signals")
    assert r.status_code == 200
    assert any(s["id"] == "lexical-heuristic" for s in r.json())


def test_detect_text_form_returns_a_verdict() -> None:
    r = client.post("/detect", data={"text": LONG_TEXT})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] in {"ai", "human", "abstain"}
    assert 0.0 <= body["p_ai"] <= 1.0
    assert isinstance(body["evidence"], list)


def test_detect_without_input_is_rejected() -> None:
    r = client.post("/detect", data={})
    assert r.status_code == 422
