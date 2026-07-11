# ROADMAP — Bedrock

The milestone checklist.

**Rules of the road:**
- Each milestone is an **independently runnable** slice — something actually testable end-to-end.
- Every milestone ends with explicit **Test** steps: what to do and what should happen. These are
  the acceptance criteria.
- Build **top-down**: a thin end-to-end slice first, then deepen. Counts and scopes are budgets,
  not promises — split a milestone if it grows too big.
- Check a box **only after its Test passes**.

---

## Phase 0 — Walking skeleton

- [x] **M0 — Skeleton & it runs.** The full spine runs end-to-end: ingest → modality router →
  `Signal` port → **real** log-odds fusion core → `Verdict`, exposed via a Typer CLI and a FastAPI
  service, with a placeholder text signal. Tooling (`uv` + hatchling), ruff, mypy(strict), and a
  passing test suite are wired. The fusion core (the moat) is real from day one; only the *signal*
  is a stub, so most inputs honestly `abstain`.
  **Test:** `uv run pytest` → green; `uv run bedrock detect --text "…"` → prints a verdict + evidence
  report; `uv run uvicorn bedrock.service.app:app` then open `http://localhost:8000/docs` → `/detect` works.

## Phase 1 — Real signals, one tier at a time

- [x] **M1 — Provenance-first images (C2PA).** Added the `c2pa-python` adapter behind the `Signal`
  port: validates a Content Credentials manifest against bundled CAI + C2PA-conformance trust
  lists, maps IPTC `digitalSourceType` markers (and, as a fallback, known AI generator names) to
  CONCLUSIVE evidence. Reliability is **direction-aware and per-manifest** (ADR-0003): untrusted
  AI claims are discounted, untrusted capture claims count zero, and a spoofed claim can't launder
  itself through a trust-listed re-export. Absent manifests → `applicable=False`; tampered ones →
  conflict → `abstain`. Image ingest wired (container brands checked, so WebP/HEIC/AVIF route right).
  **Test:** `bedrock detect firefly.jpg` (a real Adobe Firefly asset) → `100% AI-generated,
  decision: AI`, CONCLUSIVE evidence, `signer=Adobe Inc.`; an unsigned image → `abstain` with the
  C2PA signal marked not-applicable; a self-signed "camera" spoof → `abstain` (claim weighed zero).

- [x] **M2a — Calibration + conformal abstention + eval harness (the thesis core).** Replaced the
  M0 placeholders with real, fitted calibration (`PlattCalibrator`, `IsotonicCalibrator` via PAVA)
  and a **split-conformal** abstention gate with a class-conditional coverage guarantee — all pure
  stdlib, so the calibration core stays zero-dependency. Added the **evaluation harness** (ECE,
  Brier, AUROC, FPR with a subgroup slice, empirical coverage) and `bedrock fit` / `bedrock eval`
  CLI commands. Calibration ships as a versioned, fingerprinted `CalibrationArtifact` threaded
  through detection; the default stays honestly uncalibrated until an artifact is fit on real data.
  **Test:** `bedrock fit corpus.jsonl -o cal.json` then `bedrock eval corpus.jsonl -c cal.json` →
  reports ECE / FPR / coverage; on synthetic scores, isotonic cuts ECE 0.069 → 0.008 and empirical
  conformal coverage tracks the 1-α target (both pinned by tests).

- [ ] **M2b — Neural text detector ensemble.** Add Binoculars + Fast-DetectGPT (zero-shot, over an
  Apache-2.0 scoring LLM behind the `[text]` extra) and the SynthID-Text watermark adapter; fit the
  M2a calibrator + conformal gate on the RAID benchmark (incl. a non-native-English slice) and ship
  the trained artifact as the default. Retire the `lexical.py` placeholder.
  **Test:** `bedrock detect essay.txt` → a calibrated `p_ai` with a CI and honest `abstain` on
  borderline text; `bedrock eval raid.jsonl -c shipped.json` prints ECE < 0.05 and the target FPR.

- [ ] **M3 — Image ML ensemble + multi-signal fusion.** Add open image-forensics detector(s) so an
  image fuses provenance + watermark + ML; turn on conflict surfacing (C2PA "camera" vs. pixel tells).
  **Test:** run on a generated-but-metadata-stripped image → ML signals carry the verdict; run on a
  tampered image whose C2PA claim contradicts the pixels → verdict `abstain` with a conflict noted.

## Phase 2 — Production-credible

- [ ] **M4 — Explanation, ledger & hardening.** Claude-rendered evidence reports (`explain` extra,
  narrator only); a Postgres verdict ledger + replay (`GET /verdict/{id}`); tuned OOD guards; a
  Redis/Arq async queue for slow media; an opt-in Reality Defender adapter (free dev tier).
  **Test:** submit an input, get a verdict id back, re-fetch it and get a byte-identical verdict +
  a plain-English report; a large input is processed via the queue without blocking the API.

- [ ] **M5 — Thin hosted API + demo console.** Deploy the FastAPI service and a minimal Next.js
  console (upload → evidence card with the abstain state). The first public, visible Bedrock — the
  pivot toward the hosted product.
  **Test:** open the deployed console, upload an image/paste text, see the calibrated evidence card.

## Phase 3 — Planned, not yet scheduled

- [ ] **M-later — Audio & Video.** Audio: voice-clone + AI-music (Suno/Udio) adapters (MERT features,
  acoustic-signature ensemble). Video: frame-sampling reusing the image pipeline + audio track +
  temporal-consistency signal, over the async queue. Both are *new adapters* — no fusion-core change.
  **Test:** `bedrock detect clip.mp3` / `clip.mp4` returns a calibrated verdict through the same core.

---

**North star:** the one AI detector people actually trust — because it is calibrated, it explains
every verdict, and it says "I don't know" instead of ruining someone with a confident false positive.
