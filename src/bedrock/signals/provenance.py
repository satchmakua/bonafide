"""C2PA / Content Credentials provenance signal (M1) — the first CONCLUSIVE-tier adapter.

Reads a C2PA manifest store via ``c2pa-python`` (Rust ``c2pa-rs`` binding), verifies the
signer against bundled trust lists, and maps what the manifest *claims* about the asset's
origin to log-likelihood evidence. The rules, from DESIGN.md and ADR-0003:

1. **Absence is not evidence.** No manifest → ``applicable=False`` (contributes exactly 0).
   Manifests are routinely stripped by platforms, and most generators never add one.
2. **A claim is only as good as the signer who vouched for it — per manifest, not per store.**
   ``validation_state`` describes the *active* manifest's signer only. A capture claim riding
   in an untrusted *ingredient* is not vouched for by the trusted editor that re-signed the
   asset, so it must not inherit that trust (otherwise: self-sign AI as "camera", round-trip
   through Photoshop, and the spoof launders itself into a trusted verdict).
3. **The discount is direction-aware.** An untrusted signer claiming "AI-generated" is a
   statement against interest (still strong); an untrusted signer claiming "camera capture"
   is exactly what a spoofer forges, so it counts for zero.
4. **A signed manifest without origin markers is NOT evidence of human origin** (attackers can
   validly sign AI content and simply omit the AI-origin assertion), and AI markers may live in
   *ingredient* manifests (an AI image edited in Photoshop), so we walk the whole store.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from importlib import resources
from typing import Any

import c2pa

from ..ingest import sniff_mime
from ..types import Evidence, InputContext, Modality, SignalTier

# --- digitalSourceType -> llr -------------------------------------------------------------
# IPTC digitalsourcetype vocabulary (verified against cv.iptc.org 2026-07-08), keyed by the
# URI's last path segment — real manifests use both http:// and https:// schemes (Adobe
# Firefly ships https://). Positive = AI, negative = real capture / human tooling.
# |6.0| = e^6 ~ 400:1 odds: conclusive, but not so saturated that a conflicting conclusive
# signal can't pull the verdict back to abstain.
_DST_LLR: dict[str, float] = {
    # AI generation
    "trainedAlgorithmicMedia": 6.0,  # the canonical GenAI marker (OpenAI, Firefly, Google)
    "compositeWithTrainedAlgorithmicMedia": 3.5,  # GenAI edit (inpainting / Generative Fill)
    "compositedWithTrainedAlgorithmicMedia": 3.5,  # early-guidance misspelling, in the wild
    "compositeSynthetic": 3.0,  # composite, at least one element GenAI
    "algorithmicMedia": 2.0,  # pure algorithm, not trained on data (CGI/procedural)
    "virtualRecording": 2.0,  # recording of a virtual/GenAI event
    "composite": 0.75,  # elements "may or may not" be GenAI — nearly uninformative
    # Real capture
    "digitalCapture": -6.0,  # a digital camera captured a real-life scene
    "computationalCapture": -5.0,  # multi-frame merge of real captures (HDR, night mode)
    "negativeFilm": -5.0,
    "positiveFilm": -5.0,
    "print": -4.0,
    "compositeCapture": -4.0,  # composite where all elements are real captures
    # Human, non-generative tooling — weakly toward human (self-asserted, easily mislabeled)
    "digitalCreation": -1.0,
    "digitalArt": -1.0,  # retired 2024-09, superseded by digitalCreation
    "softwareImage": -1.0,  # retired 2022-06, superseded by digitalCreation
    # Neutral — recorded, zero weight
    "humanEdits": 0.0,
    "minorHumanEdits": 0.0,  # retired 2024-09
    "algorithmicallyEnhanced": 0.0,
    "dataDrivenMedia": 0.0,
    "screenCapture": 0.0,
}

# Fallback for manifests that name a known AI generator but omit digitalSourceType. Matched on
# word boundaries against a separator-normalized name. Deliberately excludes ambiguous
# fragments ("imagen" matches *Imagenomic Portraiture*; "gemini" matches MacPaw's *Gemini
# Photos*; "sora"/"firefly"/"stability" are common words) — those tools emit
# trainedAlgorithmicMedia anyway, which is signed evidence rather than a name guess.
_AI_GENERATOR_PATTERNS: tuple[str, ...] = (
    "openai",
    "chatgpt",
    "dall-e",
    "dall·e",  # OpenAI's actual middle-dot spelling
    "gpt-4o",
    "gpt-image",
    "midjourney",
    "stable diffusion",
    "stability ai",
    "adobe firefly",
    "google c2pa core generator",
)
_AI_GENERATOR_RE = re.compile(
    "|".join(rf"(?<!\w){re.escape(p)}(?!\w)" for p in _AI_GENERATOR_PATTERNS)
)
_GENERATOR_FALLBACK_LLR = 4.0

# Direction-aware reliability (ADR-0003). "Vouched" = the trusted active manifest itself
# carries the claim; an ingredient's claim was never attested to by that signer.
_RELIABILITY_VOUCHED = 0.95
_RELIABILITY_UNVOUCHED_AI = 0.85  # self-declaration against interest
_RELIABILITY_UNVOUCHED_NONAI = 0.0  # spoofable in exactly the direction that matters

_UNVOUCHED_AI_NOTE = (
    "AI claim not vouched for by a trust-listed signer on the active manifest - "
    "discounted as a statement against interest (ADR-0003)"
)
_UNVOUCHED_CAPTURE_NOTE = (
    "capture/human claim not vouched for by a trust-listed signer on the active manifest - "
    "counted as zero (self-signed 'camera' claims are exactly how provenance is spoofed)"
)
_NO_MARKER_NOTE = (
    "manifest present but carries no origin markers - NOT evidence of human origin "
    "(AI-origin assertions can simply be omitted)"
)


def _dst_key(uri: str) -> str:
    """The vocabulary term of a digitalSourceType URI (its last path segment)."""
    return uri.rstrip("/").rsplit("/", 1)[-1]


def _normalize_generator(name: str) -> str:
    """Lowercase and treat underscores as spaces ("Adobe_Firefly/1.0" -> "adobe firefly/1.0")."""
    return name.lower().replace("_", " ")


def _bundled(name: str) -> str:
    return resources.files("bedrock.signals").joinpath(f"data/trust/{name}").read_text(
        encoding="utf-8"
    )


def _fingerprint(*parts: str) -> str:
    """Short content hash of the trust material, so a Verdict pins the lists that produced it."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:12]


def _collect_actions(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """All action dicts from every c2pa.actions(.v2) assertion in one manifest."""
    actions: list[dict[str, Any]] = []
    assertions = manifest.get("assertions")
    for assertion in assertions if isinstance(assertions, list) else []:
        if not isinstance(assertion, dict):
            continue
        if str(assertion.get("label", "")).startswith("c2pa.actions"):
            data = assertion.get("data")
            entries = data.get("actions") if isinstance(data, dict) else None
            if isinstance(entries, list):
                actions.extend(a for a in entries if isinstance(a, dict))
    return actions


def _collect_generators(manifest: dict[str, Any], actions: list[dict[str, Any]]) -> list[str]:
    """Every generator/agent name string in one manifest (v1 + v2 shapes)."""
    names: list[str] = []
    info_list = manifest.get("claim_generator_info")
    for info in info_list if isinstance(info_list, list) else []:
        if isinstance(info, dict) and isinstance(info.get("name"), str):
            names.append(info["name"])
    if isinstance(manifest.get("claim_generator"), str):  # v1 top-level string
        names.append(manifest["claim_generator"])
    for action in actions:
        agent = action.get("softwareAgent")
        if isinstance(agent, dict) and isinstance(agent.get("name"), str):  # v2 object
            names.append(agent["name"])
        elif isinstance(agent, str):  # v1 plain string
            names.append(agent)
    return names


def _issuer(manifests: dict[str, Any], manifest_id: object) -> str | None:
    manifest = manifests.get(str(manifest_id))
    if not isinstance(manifest, dict):
        return None
    info = manifest.get("signature_info")
    issuer = info.get("issuer") if isinstance(info, dict) else None
    return str(issuer) if issuer else None


def reliability_for(*, validation_state: str, llr: float, basis_on_active: bool) -> float:
    """The direction-aware trust discount (ADR-0003). Pure, so it can be tested exhaustively.

    A claim earns full reliability only when the trusted signer of the *active* manifest is the
    one making it. Otherwise an AI claim is still strong (statement against interest) while a
    capture/human claim counts for zero (the canonical C2PA spoof).
    """
    if validation_state == "Invalid":
        return 0.0
    if validation_state == "Trusted" and basis_on_active:
        return _RELIABILITY_VOUCHED
    if llr > 0:
        return _RELIABILITY_UNVOUCHED_AI
    return _RELIABILITY_UNVOUCHED_NONAI


def classify_store(store: dict[str, Any]) -> tuple[float, str | None, dict[str, Any]]:
    """Pure mapping: manifest store JSON -> (llr, basis, detail). No I/O — unit-testable.

    Walks EVERY manifest in the store (not just the active one) because AI-origin evidence
    often lives in an ingredient manifest after an edit — and records *which* manifest the
    winning claim came from, so ``analyze`` can tell vouched claims from laundered ones.
    """
    raw = store.get("manifests")
    manifests: dict[str, Any] = raw if isinstance(raw, dict) else {}
    active_id = str(store.get("active_manifest") or "")

    markers: list[tuple[float, str, str]] = []  # (llr, uri, manifest_id)
    generators: list[tuple[str, str]] = []  # (name, manifest_id)
    source_types: list[str] = []
    for manifest_id, manifest in manifests.items():
        if not isinstance(manifest, dict):
            continue
        actions = _collect_actions(manifest)
        for action in actions:
            uri = action.get("digitalSourceType")
            if isinstance(uri, str):
                source_types.append(uri)
                if _dst_key(uri) in _DST_LLR:
                    markers.append((_DST_LLR[_dst_key(uri)], uri, str(manifest_id)))
        for name in _collect_generators(manifest, actions):
            generators.append((name, str(manifest_id)))

    ai_markers = [m for m in markers if m[0] > 0]
    non_ai_markers = [m for m in markers if m[0] < 0]
    ai_generator = next(
        ((n, mid) for n, mid in generators if _AI_GENERATOR_RE.search(_normalize_generator(n))),
        None,
    )

    detail: dict[str, Any] = {
        "digital_source_types": sorted(set(source_types)),
        "generators": sorted({name for name, _ in generators}),
        "manifest_count": len(manifests),
        "basis_on_active": False,
    }

    def _marker_basis(entry: tuple[float, str, str]) -> tuple[float, str, dict[str, Any]]:
        llr, uri, manifest_id = entry
        detail["basis_manifest"] = manifest_id
        detail["basis_on_active"] = manifest_id == active_id
        return llr, f"digitalSourceType:{_dst_key(uri)}", detail

    # AI markers outrank capture markers: a real photo edited with GenAI *does* contain AI
    # content, and that mixed case is exactly what compositeWithTrainedAlgorithmicMedia means.
    if ai_markers:
        return _marker_basis(max(ai_markers, key=lambda m: m[0]))

    if non_ai_markers:
        # A signed capture marker outranks the generator-name heuristic — but if the manifest
        # ALSO names an AI generator, that is a genuine contradiction: surface it and abstain.
        if ai_generator is not None:
            detail["matched_generator"] = ai_generator[0]
            detail["conflict"] = (
                f"manifest asserts real capture but names AI generator {ai_generator[0]!r}"
            )
        return _marker_basis(min(non_ai_markers, key=lambda m: m[0]))

    if ai_generator is not None:  # no signed origin marker at all — fall back to the name
        name, manifest_id = ai_generator
        detail["matched_generator"] = name
        detail["basis_manifest"] = manifest_id
        detail["basis_on_active"] = manifest_id == active_id
        return _GENERATOR_FALLBACK_LLR, f"claim_generator:{name}", detail

    detail["note"] = _NO_MARKER_NOTE
    return 0.0, None, detail


class C2paSignal:
    """Verify Content Credentials and turn the *claims* + *signer trust* into Evidence."""

    id = "c2pa.manifest"
    modalities = frozenset({Modality.IMAGE})
    tier = SignalTier.CONCLUSIVE
    version: str

    def __init__(self, trust_anchors_pem: str | None = None) -> None:
        """``trust_anchors_pem=None`` loads the bundled CAI + C2PA-conformance lists;
        passing a PEM string replaces them (used in tests and by orgs with private anchors).

        The trust material is fingerprinted into ``version``, so a Verdict pins the exact
        lists that produced it — refreshing the bundled PEMs changes the recorded version.
        """
        if trust_anchors_pem is None:
            anchors = _bundled("anchors.pem") + "\n" + _bundled("c2pa-trust-list.pem")
            allowed, config = _bundled("allowed.sha256.txt"), _bundled("store.cfg")
            fingerprint = _fingerprint(anchors, allowed, config)
            self._trust_settings: dict[str, Any] = {
                "trust": {
                    "trust_anchors": anchors,
                    "allowed_list": allowed,
                    "trust_config": config,
                },
                "verify": {"verify_trust": True},
            }
            self._trust_source = f"bundled CAI+C2PA lists (sha256:{fingerprint})"
        else:
            fingerprint = _fingerprint(trust_anchors_pem)
            self._trust_settings = {
                "trust": {"trust_anchors": trust_anchors_pem},
                "verify": {"verify_trust": True},
            }
            self._trust_source = f"custom anchors (sha256:{fingerprint})"
        self.version = (
            f"c2pa-python/{c2pa.__version__}+sdk/{c2pa.sdk_version()}+trust/{fingerprint}"
        )

    def _not_applicable(self, note: str) -> Evidence:
        return Evidence(
            signal_id=self.id,
            applicable=False,
            tier=self.tier,
            model_version=self.version,
            detail={"note": note},
        )

    def _open_reader(self, ctx: InputContext) -> Any | None:
        settings = c2pa.Settings.from_dict(self._trust_settings)
        c2pa_ctx = c2pa.Context.builder().with_settings(settings).build()
        if ctx.data is not None:  # prefer in-memory bytes: ctx.path may be a hint, not a file
            mime = ctx.mime or sniff_mime(ctx.data) or "image/jpeg"
            return c2pa.Reader.try_create(mime, io.BytesIO(ctx.data), context=c2pa_ctx)
        if ctx.path is not None:
            return c2pa.Reader.try_create(ctx.path, context=c2pa_ctx)
        return None

    def applies_to(self, ctx: InputContext) -> bool:
        return ctx.modality in self.modalities and (ctx.path is not None or ctx.data is not None)

    def analyze(self, ctx: InputContext) -> Evidence:
        try:
            reader = self._open_reader(ctx)
            if reader is None:  # no JUMBF data in the asset — absence is not evidence
                return self._not_applicable("no C2PA manifest")
            with reader:
                store = json.loads(reader.json())
        except (c2pa.C2paError, OSError, ValueError) as exc:  # unsupported/malformed input
            return self._not_applicable(f"C2PA read failed: {type(exc).__name__}")

        state = str(store.get("validation_state") or "Valid")
        llr, basis, detail = classify_store(store)
        detail["validation_state"] = state
        detail["trust"] = self._trust_source
        if basis is not None:
            detail["basis"] = basis

        # Attribute the claim to the manifest that actually made it, not blindly to the active
        # one — otherwise a laundered ingredient claim reads as if the trusted editor asserted it.
        manifests = store.get("manifests")
        manifests = manifests if isinstance(manifests, dict) else {}
        active_id = store.get("active_manifest") or ""
        claim_signer = _issuer(manifests, detail.get("basis_manifest", active_id))
        active_signer = _issuer(manifests, active_id)
        if claim_signer:
            detail["signer"] = claim_signer
        if active_signer and active_signer != claim_signer:
            detail["active_signer"] = active_signer

        if state == "Invalid":
            # Tampered or malformed: the claims are unusable in EITHER direction. Contribute
            # nothing, but surface the broken chain as a conflict — it warrants a human look.
            detail["conflict"] = (
                "C2PA manifest present but INVALID (tampering or malformation) - "
                "provenance claims ignored"
            )
            return Evidence(
                signal_id=self.id,
                applicable=True,
                llr=0.0,
                reliability=0.0,
                tier=self.tier,
                model_version=self.version,
                detail=detail,
            )

        reliability = reliability_for(
            validation_state=state, llr=llr, basis_on_active=bool(detail["basis_on_active"])
        )
        if reliability < _RELIABILITY_VOUCHED:
            if llr > 0:
                detail["note"] = _UNVOUCHED_AI_NOTE
            elif llr < 0:
                detail["note"] = _UNVOUCHED_CAPTURE_NOTE

        return Evidence(
            signal_id=self.id,
            applicable=True,
            llr=llr,
            reliability=reliability,
            tier=self.tier,
            model_version=self.version,
            detail=detail,
        )
