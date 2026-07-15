"""Smoke tests for the FastAPI service."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from bonafide.service import app as service_app
from bonafide.service.app import app

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


def test_oversized_upload_is_rejected_with_413(monkeypatch: pytest.MonkeyPatch) -> None:
    # An unbounded read() of an attacker-sized upload would exhaust memory.
    monkeypatch.setattr(service_app, "MAX_UPLOAD_BYTES", 32)
    r = client.post("/detect", files={"file": ("big.png", b"x" * 64, "image/png")})
    assert r.status_code == 413
    assert "upload limit" in r.json()["detail"]


def test_upload_within_the_limit_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service_app, "MAX_UPLOAD_BYTES", 1024)
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    r = client.post("/detect", files={"file": ("small.png", png, "image/png")})
    assert r.status_code == 200
    assert r.json()["modality"] == "image"
