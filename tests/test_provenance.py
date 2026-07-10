"""C2PA provenance signal tests — all offline: assets are signed at run time with the
vendored test certs. The test signer is NOT on any real trust list, which lets us exercise
the untrusted paths; injecting the test CA as an anchor exercises the Trusted paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("c2pa", reason="provenance extra not installed")

from c2pa import Builder

from bedrock import default_registry, detect_file
from bedrock.fusion import fuse
from bedrock.signals.provenance import C2paSignal, classify_store, reliability_for
from bedrock.types import Evidence, Modality, SignalTier
from conftest import (
    actions_manifest,
    ca_anchor_pem,
    make_cert_chain,
    make_tiny_png,
    sign_png,
    signer_from,
)

AI_DST = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
CAPTURE_DST = "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture"


def analyze_file(path: Path, signal: C2paSignal) -> Evidence:
    from bedrock.ingest import context_from_path

    return signal.analyze(context_from_path(path))


# --- signed-asset round trips (real Reader, real signatures) ------------------------------


def test_ai_claim_from_untrusted_signer_is_strong_evidence(tmp_path: Path) -> None:
    signed = sign_png(tmp_path, actions_manifest(AI_DST))
    ev = analyze_file(signed, C2paSignal())  # default = real trust lists; test signer off-list
    assert ev.applicable
    assert ev.tier is SignalTier.CONCLUSIVE
    assert ev.llr == 6.0
    assert ev.reliability == 0.85  # statement against interest — discounted, not dismissed
    assert ev.detail["validation_state"] == "Valid"
    assert ev.detail["basis"] == "digitalSourceType:trainedAlgorithmicMedia"


def test_ai_claim_from_trusted_signer_gets_full_reliability(tmp_path: Path) -> None:
    signed = sign_png(tmp_path, actions_manifest(AI_DST))
    ev = analyze_file(signed, C2paSignal(trust_anchors_pem=ca_anchor_pem()))
    assert ev.detail["validation_state"] == "Trusted"
    assert ev.llr == 6.0
    assert ev.reliability == 0.95


def test_untrusted_capture_claim_counts_for_zero(tmp_path: Path) -> None:
    # Anyone can self-sign "I am a camera" onto an AI image — the spoof this policy blocks.
    signed = sign_png(tmp_path, actions_manifest(CAPTURE_DST))
    ev = analyze_file(signed, C2paSignal())
    assert ev.applicable
    assert ev.llr == -6.0
    assert ev.reliability == 0.0
    assert ev.contribution == 0.0
    assert "spoofed" in ev.detail["note"]


def test_trusted_capture_claim_is_strong_human_evidence(tmp_path: Path) -> None:
    signed = sign_png(tmp_path, actions_manifest(CAPTURE_DST))
    ev = analyze_file(signed, C2paSignal(trust_anchors_pem=ca_anchor_pem()))
    assert ev.detail["validation_state"] == "Trusted"
    assert ev.llr == -6.0
    assert ev.reliability == 0.95


def test_manifest_without_origin_markers_is_not_human_evidence(tmp_path: Path) -> None:
    # Semantic omission: validly signed, no digitalSourceType, unknown generator -> llr 0.
    signed = sign_png(tmp_path, actions_manifest(None))
    ev = analyze_file(signed, C2paSignal(trust_anchors_pem=ca_anchor_pem()))
    assert ev.applicable
    assert ev.llr == 0.0
    assert "NOT evidence of human origin" in ev.detail["note"]


def test_unsigned_image_is_not_applicable(tmp_path: Path, tiny_png: bytes) -> None:
    plain = tmp_path / "plain.png"
    plain.write_bytes(tiny_png)
    ev = analyze_file(plain, C2paSignal())
    assert not ev.applicable
    assert ev.contribution == 0.0
    assert ev.detail["note"] == "no C2PA manifest"


def test_tampered_asset_is_invalid_and_raises_a_conflict(tmp_path: Path) -> None:
    signed = sign_png(tmp_path, actions_manifest(AI_DST))
    blob = bytearray(signed.read_bytes())
    idat = blob.rindex(b"IDAT")  # flip a pixel byte after signing -> content hash mismatch
    blob[idat + 8] ^= 0xFF
    tampered = tmp_path / "tampered.png"
    tampered.write_bytes(bytes(blob))

    ev = analyze_file(tampered, C2paSignal())
    assert ev.applicable
    assert ev.detail["validation_state"] == "Invalid"
    assert ev.contribution == 0.0
    assert "INVALID" in ev.detail["conflict"]


# --- end-to-end through the engine ---------------------------------------------------------


def test_engine_verdict_ai_for_signed_ai_image(tmp_path: Path) -> None:
    signed = sign_png(tmp_path, actions_manifest(AI_DST))
    verdict = detect_file(signed)
    assert verdict.decision == "ai"
    assert verdict.p_ai > 0.95
    assert any(e.signal_id == "c2pa.manifest" and e.applicable for e in verdict.evidence)


def test_engine_abstains_for_unsigned_image(tmp_path: Path, tiny_png: bytes) -> None:
    plain = tmp_path / "plain.png"
    plain.write_bytes(tiny_png)
    verdict = detect_file(plain)
    assert verdict.decision == "abstain"
    c2pa_ev = next(e for e in verdict.evidence if e.signal_id == "c2pa.manifest")
    assert not c2pa_ev.applicable


def test_engine_abstains_and_surfaces_conflict_for_tampered_image(tmp_path: Path) -> None:
    signed = sign_png(tmp_path, actions_manifest(AI_DST))
    blob = bytearray(signed.read_bytes())
    blob[blob.rindex(b"IDAT") + 8] ^= 0xFF
    tampered = tmp_path / "tampered.png"
    tampered.write_bytes(bytes(blob))

    verdict = detect_file(tampered)
    assert verdict.decision == "abstain"
    assert any("INVALID" in c for c in verdict.conflicts)


def test_default_registry_offers_the_c2pa_signal() -> None:
    assert any(s.id == "c2pa.manifest" for s in default_registry().signals)


# --- pure classification (hand-built stores; no signing needed) ----------------------------


def _store(*manifests: dict[str, Any], active: int = 0) -> dict[str, Any]:
    labels = [f"urn:m{i}" for i in range(len(manifests))]
    return {
        "manifests": dict(zip(labels, manifests, strict=True)),
        "active_manifest": labels[active] if labels else "",
    }


def test_classify_finds_ai_marker_in_ingredient_manifest() -> None:
    # A Photoshop-edited AI image: the ACTIVE manifest only says "opened", the AI marker
    # lives in the ingredient's manifest. classify_store must walk the whole store.
    active = {
        "claim_generator_info": [{"name": "Adobe Photoshop"}],
        "assertions": [
            {"label": "c2pa.actions.v2", "data": {"actions": [{"action": "c2pa.opened"}]}}
        ],
    }
    ingredient = actions_manifest(AI_DST)
    llr, basis, detail = classify_store(_store(active, ingredient))
    assert llr == 6.0
    assert basis == "digitalSourceType:trainedAlgorithmicMedia"
    assert detail["basis_on_active"] is False  # it came from the ingredient


# --- ADR-0003: trust is per-manifest, not per-store (ingredient-laundering defense) ---------


def test_reliability_trusted_only_when_the_active_manifest_makes_the_claim() -> None:
    # The laundering attack: self-sign AI as "digitalCapture", round-trip through a
    # trust-listed editor. The store reads Trusted, but the capture claim rides in an
    # untrusted ingredient — it must NOT inherit the editor's trust.
    assert reliability_for(validation_state="Trusted", llr=-6.0, basis_on_active=False) == 0.0
    # A genuine camera: the trusted signer itself asserts capture.
    assert reliability_for(validation_state="Trusted", llr=-6.0, basis_on_active=True) == 0.95


def test_reliability_ai_claims_survive_an_untrusted_signer() -> None:
    # Statement against interest: discounted, never dismissed — from anywhere in the store.
    assert reliability_for(validation_state="Valid", llr=6.0, basis_on_active=True) == 0.85
    assert reliability_for(validation_state="Trusted", llr=6.0, basis_on_active=False) == 0.85
    assert reliability_for(validation_state="Trusted", llr=6.0, basis_on_active=True) == 0.95


def test_reliability_is_zero_for_tampered_assets() -> None:
    for llr in (6.0, -6.0, 0.0):
        assert reliability_for(validation_state="Invalid", llr=llr, basis_on_active=True) == 0.0


def test_laundered_capture_claim_in_an_ingredient_earns_no_trust() -> None:
    # End-to-end through classify_store + the policy: trusted active manifest asserts nothing,
    # untrusted ingredient asserts digitalCapture.
    active = {
        "claim_generator_info": [{"name": "Adobe Photoshop"}],
        "assertions": [
            {"label": "c2pa.actions.v2", "data": {"actions": [{"action": "c2pa.created"}]}}
        ],
    }
    llr, _, detail = classify_store(_store(active, actions_manifest(CAPTURE_DST)))
    assert llr == -6.0
    assert detail["basis_on_active"] is False
    reliability = reliability_for(
        validation_state="Trusted", llr=llr, basis_on_active=detail["basis_on_active"]
    )
    assert reliability == 0.0  # contribution 0 -> abstain, not a confident "human"


def test_claim_is_attributed_to_the_manifest_that_made_it() -> None:
    active = {"signature_info": {"issuer": "Adobe Inc."}, "assertions": []}
    ingredient = actions_manifest(CAPTURE_DST)
    ingredient["signature_info"] = {"issuer": "Evil Corp"}
    _, _, detail = classify_store(_store(active, ingredient))
    assert detail["basis_manifest"] == "urn:m1"  # not the active/trusted one


def test_end_to_end_ingredient_laundering_attack_is_blocked(tmp_path: Path) -> None:
    """The canonical C2PA laundering attack, signed for real with two distinct cert chains.

    An attacker self-signs an AI image as 'digitalCapture' with an off-list cert, then
    re-exports it through a trust-listed editor. The spoofed manifest survives as an ingredient
    and the store validates as *Trusted* on the editor's cert. Bedrock must still refuse to
    treat the ingredient's camera claim as vouched-for evidence.
    """
    from bedrock.ingest import context_from_path

    evil, editor = make_cert_chain("Evil Corp"), make_cert_chain("Trusted Editor")

    source = tmp_path / "src.png"
    source.write_bytes(make_tiny_png())
    spoofed = tmp_path / "spoofed.png"
    Builder(actions_manifest(CAPTURE_DST)).sign_file(source, spoofed, signer_from(evil))

    laundered = tmp_path / "laundered.png"
    # The trusted editor asserts nothing about origin — it merely re-exports.
    created = {"label": "c2pa.actions", "data": {"actions": [{"action": "c2pa.created"}]}}
    builder = Builder({
        "claim_generator_info": [{"name": "Trusted Editor App", "version": "2.0"}],
        "assertions": [created],
    })
    ingredient = {"title": "o.png", "relationship": "componentOf"}
    with spoofed.open("rb") as handle:
        builder.add_ingredient(ingredient, "image/png", handle)
    builder.sign_file(source, laundered, signer_from(editor))

    signal = C2paSignal(trust_anchors_pem=editor.ca_pem)
    # Check only the active signer's trust — c2pa's own posture, and exactly when this bug bit.
    signal._trust_settings["verify"]["check_ingredient_trust"] = False
    ev = signal.analyze(context_from_path(laundered))

    assert ev.detail["validation_state"] == "Trusted"  # the store really is trusted...
    assert ev.detail["basis_on_active"] is False  # ...but the claim came from the ingredient
    assert ev.llr == -6.0
    assert ev.reliability == 0.0  # so it is recorded and weighed at exactly nothing
    assert ev.contribution == 0.0
    assert ev.detail["signer"] == "Evil Corp"  # attribution names the true claimant
    assert ev.detail["active_signer"] == "Trusted Editor"

    verdict = fuse([ev], modality=Modality.IMAGE, engine_version="t")
    assert verdict.decision == "abstain"  # never a confident "human"
    assert verdict.p_ai == 0.5


def test_classify_ai_marker_outranks_capture_marker() -> None:
    # A real photo edited with GenAI contains AI content; the AI marker must win.
    llr, basis, _ = classify_store(
        _store(
            actions_manifest(CAPTURE_DST),
            actions_manifest(
                "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia"
            ),
        )
    )
    assert llr == 3.5
    assert basis is not None and "compositeWithTrainedAlgorithmicMedia" in basis


def test_classify_accepts_https_scheme_uris() -> None:
    # Real Adobe Firefly manifests use https:// IPTC URIs, not the canonical http://.
    llr, basis, _ = classify_store(
        _store(
            actions_manifest(
                "https://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
            )
        )
    )
    assert llr == 6.0
    assert basis == "digitalSourceType:trainedAlgorithmicMedia"


def test_classify_accepts_the_misspelled_composited_variant() -> None:
    llr, _, _ = classify_store(
        _store(
            actions_manifest(
                "http://cv.iptc.org/newscodes/digitalsourcetype/compositedWithTrainedAlgorithmicMedia"
            )
        )
    )
    assert llr == 3.5


def test_benign_tool_name_containing_a_pattern_substring_is_not_ai_evidence() -> None:
    # "Imagenomic Portraiture" contains "imagen"; "Gemini Photos" contains "gemini".
    # Bare substring matching would have flagged a real photo as AI-generated.
    for name in ("Imagenomic Portraiture 4.5", "Gemini Photos 2.1", "Sorativa Studio"):
        manifest = {
            "claim_generator_info": [{"name": name}],
            "assertions": [
                {"label": "c2pa.actions.v2", "data": {"actions": [{"action": "c2pa.edited"}]}}
            ],
        }
        llr, basis, _ = classify_store(_store(manifest))
        assert llr == 0.0, name
        assert basis is None, name


def test_explicit_capture_marker_outranks_the_generator_name_and_conflicts() -> None:
    # A signed digitalCapture marker beats the name heuristic — but naming an AI generator
    # alongside it is a genuine contradiction, so surface a conflict and let fusion abstain.
    manifest = actions_manifest(CAPTURE_DST)
    manifest["claim_generator_info"] = [{"name": "Midjourney"}]
    llr, basis, detail = classify_store(_store(manifest))
    assert llr == -6.0
    assert basis == "digitalSourceType:digitalCapture"
    assert "Midjourney" in detail["conflict"]


def test_generator_fallback_matches_separator_variants() -> None:
    # Real Firefly manifests carry "Adobe_Firefly/1.0 c2pa-adobe-js/0.16.0 c2pa-rs/0.25.1".
    manifest = {
        "claim_generator": "Adobe_Firefly/1.0 c2pa-adobe-js/0.16.0 c2pa-rs/0.25.1",
        "assertions": [],
    }
    llr, _, _ = classify_store(_store(manifest))
    assert llr == 4.0


def test_classify_falls_back_to_known_ai_generator() -> None:
    manifest = {
        "claim_generator_info": [{"name": "ChatGPT", "org.cai.c2pa_rs": "0.49.5"}],
        "assertions": [
            {"label": "c2pa.actions.v2", "data": {"actions": [{"action": "c2pa.created"}]}}
        ],
    }
    llr, basis, detail = classify_store(_store(manifest))
    assert llr == 4.0
    assert basis == "claim_generator:ChatGPT"
    assert detail["matched_generator"] == "ChatGPT"


def test_classify_reads_v1_softwareagent_strings() -> None:
    manifest = {
        "claim_generator": "OpenAI-API c2pa-rs/0.28.4",
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {"actions": [{"action": "c2pa.created", "softwareAgent": "DALL·E"}]},
            }
        ],
    }
    llr, basis, _ = classify_store(_store(manifest))
    assert llr == 4.0
    assert basis is not None and basis.startswith("claim_generator:")


def test_classify_neutral_source_types_carry_no_weight() -> None:
    llr, basis, _ = classify_store(
        _store(actions_manifest("http://cv.iptc.org/newscodes/digitalsourcetype/screenCapture"))
    )
    assert llr == 0.0
    assert basis is None


def test_classify_tolerates_malformed_store_shapes() -> None:
    # Manifest stores are attacker-controlled JSON; nothing here may raise.
    stores: tuple[dict[str, Any], ...] = (
        {},
        {"manifests": None},
        {"manifests": []},
        {"manifests": {"a": "not-a-dict"}},
        {"manifests": {"a": {"assertions": "not-a-list"}}},
        {"manifests": {"a": {"assertions": [{"label": "c2pa.actions", "data": None}]}}},
        {"manifests": {"a": {"assertions": [{"label": "c2pa.actions", "data": {"actions": [1]}}]}}},
        {"manifests": {"a": {"claim_generator_info": [{"name": 42}]}}},
        {"manifests": {"a": {"assertions": [{"label": "c2pa.actions", "data": {"actions": [
            {"action": "c2pa.created", "digitalSourceType": 99}]}}]}}},
    )
    for store in stores:
        llr, basis, detail = classify_store(store)
        assert llr == 0.0
        assert basis is None
        assert detail["basis_on_active"] is False


# --- trust-list fingerprinting (reproducibility constraint) --------------------------------


def test_trust_material_is_fingerprinted_into_the_signal_version() -> None:
    bundled = C2paSignal()
    custom = C2paSignal(trust_anchors_pem=ca_anchor_pem())
    other = C2paSignal(trust_anchors_pem=ca_anchor_pem() + "\n# different bytes\n")
    assert "trust/" in bundled.version
    assert bundled.version != custom.version != other.version
    assert custom.version != other.version
    assert "sha256:" in custom._trust_source
