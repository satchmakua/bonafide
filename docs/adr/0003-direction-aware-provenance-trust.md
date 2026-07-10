# 3. Direction-aware reliability for provenance claims

- **Status:** Accepted
- **Date:** 2026-07-10

## Context

A C2PA manifest is a *signed claim*, not ground truth. `c2pa-rs` reports three validation
states: `Trusted` (signer chains to a configured trust list), `Valid` (signature intact but
signer unknown — the default for ANY self-signed manifest), and `Invalid` (tampered/malformed).

The naive mapping — "valid manifest ⇒ believe its claims" — is exploitable in one specific,
well-documented direction: anyone can generate an AI image and self-sign a manifest claiming
`digitalSourceType: digitalCapture` ("a camera took this"). Nothing stops this; it is the
canonical C2PA spoof. The reverse claim — self-signing *your own* content as AI-generated —
is a statement against interest with no comparable attack economics (the marginal scenario,
framing someone else's human art as AI by re-signing it, produces a *new* manifest that
doesn't bind to the artist's identity, and Bedrock reports evidence + signer, never an
accusation).

## Decision

Reliability (the `w_i` weight in fusion, ADR-0002) depends on **the direction of the claim**
and on **whether the trusted signer actually made it** — not merely on the store's validation
state. `validation_state` describes only the **active** manifest's signer. Define a claim as
*vouched* when `validation_state == "Trusted"` **and** the claim sits on that active manifest:

| | AI-direction claim (llr > 0) | capture/human claim (llr < 0) |
|---|---|---|
| Vouched (trusted signer, active manifest) | 0.95 | 0.95 |
| Unvouched (off-list signer, **or** the claim rides in an ingredient) | 0.85 | **0.0** — recorded, never counted |
| `Invalid` (tampered/malformed) | 0.0 + a `conflict` note (forces abstain) | 0.0 + conflict |

The per-manifest rule closes the **ingredient-laundering attack**: self-sign an AI image as
`digitalCapture` with a throwaway cert, then re-export it through any trust-listed editor
(Photoshop preserves the original manifest as an ingredient). The store then validates as
`Trusted` on the *editor's* cert. Judging trust at store level would hand the attacker's
spoofed camera claim a 0.95 weight and return a confident `human` verdict for an AI image —
reproduced end-to-end, and now pinned by
`test_end_to_end_ingredient_laundering_attack_is_blocked`. Evidence therefore also records
`basis_manifest` / `basis_on_active`, and attributes `signer` to the manifest that actually
made the claim (with `active_signer` alongside), so the audit trail never credits a claim to a
signer who never made it.

Bedrock bundles the CAI interim trust anchors + the official C2PA Conformance trust list
(`signals/data/trust/`, fetch date recorded) so genuine vendor-signed assets (Adobe, Leica,
Truepic, …) actually reach `Trusted`. Off-list means "unknown signer", never "fake" —
legitimate vendors lag the lists.

Additional mapping rules (from the C2PA/IPTC research, verified 2026-07-08):
- Markers are read from **every** manifest in the store, not just the active one — AI origin
  evidence often lives in an *ingredient* after an edit (Photoshop-edited AI image).
- AI markers outrank capture markers when both appear: a real photo edited with generative
  fill *does* contain AI content, which is exactly what its marker asserts.
- A validly-signed manifest with **no** origin markers is llr 0, not negative — AI-origin
  assertions can simply be omitted ("authenticated contradictions" / semantic omission).
- URIs are matched by vocabulary term (last path segment): both `http://` and `https://`
  IPTC schemes occur in the wild (Adobe ships `https://`).
- **A signed `digitalSourceType` marker always outranks the generator-name heuristic.** The
  name list is a fallback for manifests that omit the marker entirely. It matches on word
  boundaries against a separator-normalized name, and deliberately excludes ambiguous
  fragments — bare `imagen` matches *Imagenomic Portraiture*, `gemini` matches MacPaw's
  *Gemini Photos*; those would brand a real photograph as AI-generated. A manifest that
  asserts capture *and* names an AI generator is a genuine contradiction: raise a `conflict`
  and abstain rather than pick a side.

## Consequences

- Provenance spoofing in the harmful direction (AI passed off as camera-real) is structurally
  neutralized: the claim is displayed in the evidence trail with an explanatory note but moves
  the verdict zero.
- An untrusted AI-direction claim still yields a confident "ai" verdict (6.0 × 0.85 = 5.1
  log-odds). Accepted residual risk: maliciously re-signing human art with an AI claim reads
  as "ai" until outweighed — mitigated by the evidence-trail-not-accusation output contract,
  the visible `signer` field, and (M3+) forensic signals that can contradict it into abstain.
- Genuine capture evidence requires the signer to be on a bundled trust list, so the lists
  must be kept fresh (update mechanism is M4 hardening work). The trust material is
  **content-fingerprinted** (`sha256:…`) into `Evidence.model_version` and `detail.trust`, so
  refreshing the PEMs changes the recorded version and two runs that disagree can never look
  identical in an audit.
- Genuine camera captures whose manifests are signed by an off-list vendor (Samsung lagged the
  list at S25 launch) yield `abstain`, not `human`. Accepted: silence beats a spoofable claim.
