# Bundled C2PA trust lists

Vendored copies of the public C2PA/CAI trust lists, used by `signals/provenance.py` to
decide whether a manifest's signer chains to a known root (`validation_state: "Trusted"`)
or is merely well-formed (`"Valid"`). Fetched **2026-07-10**:

| File | Source | What it is |
|---|---|---|
| `anchors.pem` | <https://contentcredentials.org/trust/anchors.pem> (301 → verify.contentauthenticity.org) | CAI **interim** CA anchors (27 certs; frozen, "will be deprecated" per its header — kept because major vendors like Adobe still chain here) |
| `allowed.sha256.txt` | <https://contentcredentials.org/trust/allowed.sha256.txt> | CAI interim end-entity allow-list (sha256 lines) |
| `store.cfg` | <https://contentcredentials.org/trust/store.cfg> | Allowed EKU OIDs for signing certs |
| `c2pa-trust-list.pem` | <https://raw.githubusercontent.com/c2pa-org/conformance-public/refs/heads/main/trust-list/C2PA-TRUST-LIST.pem> | The **official** C2PA Conformance Program trust list (28 certs) |

**Staleness:** these lists change as vendors are added. Re-fetch periodically (a proper
update mechanism is future work — see ROADMAP M4 hardening). An off-list signer means
"unknown signer", never "fake" — legitimate vendors lag the list (e.g. Samsung at S25 launch).
