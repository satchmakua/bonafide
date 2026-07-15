# Bonafide — Design

> A calibrated, provenance-first framework for detecting AI-generated media — honest about what it knows, and about what it doesn't.

**Status:** Design draft · **Language:** Python 3.11+ (Rust-backed provenance) · **Stack target:** Engine/SDK + service, cross-platform, GPU-optional

Name: **Bonafide** *(formerly Bedrock)* — *bona fide*, "in good faith": the genuine article, verified. The verdict object is a **`Verdict`**; the fused truth-value is its **authenticity**.

---

<!-- IP / LEGAL / ETHICAL — read first, it constrains the whole design -->
> **Legal & ethical constraints (load-bearing, not footnotes).**
> 1. **A wrong "this is AI" is a real-world harm.** Independent studies put incumbent text detectors at **5–12% false-positive rates on non-native English writers**; OpenAI *retired* its own classifier for low accuracy, and universities (e.g. Vanderbilt) have **disabled** Turnitin's detector as unsafe. Bonafide therefore never emits a bare accusation. It emits *evidence + a calibrated probability + an explicit abstain state*, and the API is shaped so it **cannot** be used as an automated "this person cheated" oracle. This is both an ethical stance and a defamation/discrimination-liability shield.
> 2. **Third-party detector licensing.** Hive and Sensity are enterprise-contract-only; Reality Defender offers a free 50/mo dev tier. These are **opt-in adapters**, never hard dependencies — the open-core engine must be fully functional on open-weight models + open standards alone.
> 3. **Scoring-model licenses.** Zero-shot text detectors need a scoring LLM. Default to **Apache-2.0 models** (Falcon, Qwen) to avoid the Llama community-license >700M-MAU clause in a distributable open core.
> 4. **Regulatory tailwind.** **EU AI Act Article 50** transparency/labeling obligations take effect **2 Aug 2026** (fines to €15M / 3% of turnover). Bonafide is a *verification* enabler for that regime — a reason enterprises will need it, and a reason to get the provenance path exactly right.

*(Prior art, versions, and standards below verified via web research on **2026-07-08**.)*

---

## 1. Concept

Bonafide answers one question for any medium — **"was this made by a human or by AI?"** — but it refuses to answer it the way everyone else does. The incumbent pattern is a single model emitting a confident binary that is wrong often enough, and in biased enough ways, that its own customers are turning it off.

Bonafide treats detection as **evidence aggregation under uncertainty**. For a given input it gathers every signal it can:

- **Hard provenance** — a cryptographically-valid C2PA / Content Credentials manifest (now shipping in the Samsung Galaxy S25 camera, all of Adobe, OpenAI, and Google surfaces). Near-conclusive *when present*.
- **Watermarks** — SynthID and friends. High-trust for participating generators.
- **An ensemble of ML detectors** — zero-shot and supervised, our own and (opt-in) third-party.
- **Forensic & stylometric features** — perplexity curvature, frequency artifacts, metadata.

It then **fuses** these signals — which live on wildly different trust levels — into **one calibrated posterior probability with a confidence interval**, runs it through a **conformal-prediction gate**, and — crucially — **abstains** when the evidence doesn't support a confident call. The output is an **evidence report** you can read and audit, not a black-box number. When a valid C2PA signature says "camera-captured" but pixel forensics scream "generated," Bonafide surfaces *both* and flags the conflict rather than hiding it.

You should be able to picture it: `bonafide detect photo.jpg` → a card that says *"87% AI-generated (CI 81–91%), decision: AI"* with rows underneath — *C2PA: no manifest (n/a) · SynthID: watermark detected (+strong) · image-forensics: diffusion artifacts in 3 regions (+moderate)* — and, on a 40-word ambiguous paragraph, the honest answer: *"Insufficient evidence — abstain."*

**Engineering pillars — the 1–3 things that make or break this:**

1. **The Evidence Fusion & Calibration Engine (§4).** Combining a cryptographic signature and a noisy perplexity score into one *calibrated, honest* number — with principled abstention — is the intellectual core and the moat. Get this wrong and Bonafide is just another aggregator; get it right and it's the only detector you can actually trust.
2. **A modality-agnostic detector plugin architecture (§5, §6.1).** Ports-and-adapters so *any* detector for *any* medium plugs into the same pipeline. This is what makes "supports every kind of media eventually" true instead of aspirational — text and images in v1, audio and video as drop-in adapters later with **zero core changes**.
3. **Adversarial robustness & honesty (§6.6, §9).** Detectors lose ~50% accuracy in the wild and under attack. The system must *know when it's out of its depth* — via OOD guards, conflict detection, and conformal coverage — and say so, rather than emit confident garbage.

## 2. Goals / Non-goals

**Goals (v1 — testable):**
- One call — `bonafide.detect(input)` (SDK) and `POST /detect` (HTTP) — returns a `Verdict`: calibrated `p_ai` with a confidence interval, a `decision ∈ {ai, human, abstain}`, and an `evidence[]` list itemizing every signal's contribution.
- Fuse **≥3 signal types for images** (C2PA provenance, watermark check, ≥1 ML image detector) and **≥3 for text** (Binoculars, Fast-DetectGPT, SynthID-Text, plus stylometric features).
- **Calibrated, not just accurate:** on a held-out benchmark, reported confidence matches empirical outcome — **Expected Calibration Error (ECE) < 0.05** — and the human-text false-positive rate is *controllable to a configured operating point* (e.g. ≤1%), with borderline cases routed to `abstain` under a conformal coverage guarantee.
- **Reproducible & auditable:** the same input + same model versions yields the same verdict, byte-for-byte; every verdict is versioned and stored with the exact signal/calibrator versions that produced it.
- **Zero paid dependencies in the core:** runs locally on open-weight models + open standards; third-party detector APIs are opt-in adapters.

**Non-goals (v1) — deliberately out of scope (this section protects the build):**
- **Not** training our own SOTA foundation detectors from scratch. We orchestrate + calibrate *first*; proprietary models come once we have a labeled corpus and the eval harness to justify them (post-v1). The moat starts as the fusion layer, not the models.
- **Not** implementing audio or video in v1 — **but** the schema, plugin interface, and job queue are designed so both drop in later with no redesign (see M-later). This is a hard architectural requirement, per the product decision.
- **Not** a *marking / watermarking* product (helping creators stamp content as human/AI). Detection & verification only. (Marking is the natural adjacent product for EU AI Act *provider-side* compliance — a later company bet, not v1.)
- **Not** a "humanizer" / evasion tool, and **no** promise of adversarial-proof detection. Bonafide promises *honesty about uncertainty*, not invincibility.
- **Not** a polished consumer SaaS in v1. v1 is the **engine + SDK + a thin API and demo console**; the full hosted product (your "option 1") is the M5+ evolution.
- **Not** a plagiarism/similarity checker, identity/KYC, or content-moderation platform. Adjacent, not us.
- **No** per-person accusations, disciplinary automation, or "verdict: cheater." Bonafide outputs evidence; a human decides. Enforced by the type system, not just docs.
- **Not** real-time / streaming detection. Request-response and batch first (live-stream C2PA is a future signal).

## 3. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Core engine | **Python 3.11+** (dev on 3.11) | Every detector, model, and calibration tool lives in the Python ML ecosystem (PyTorch, `transformers`, scikit-learn). The core belongs where the signals are. `uv` manages the env; `hatchling` builds. |
| Provenance | **`c2pa-python` ≈0.36 (binding to Rust `c2pa-rs`)** | Official Content Authenticity Initiative reference implementation; validates C2PA 2.x manifests + the CA trust list; wide format support (JPEG/PNG/WebP/TIFF/AVIF/HEIF/MP4/MOV/PDF). Rust-backed = correct + fast. |
| Text watermark | **SynthID-Text via HF `transformers` ≥4.46** | The only open-source watermark detector (Weighted Mean, no training; Bayesian, trained). Image/audio/video SynthID detection is gated to Google's portal — handled as a context signal (§6.3). |
| Text detectors | **Binoculars + Fast-DetectGPT** (zero-shot) | No training data required to launch; Binoculars is known for very low FPR at a fixed threshold; both open-source and evaluated on the RAID robustness benchmark. |
| Scoring LLM | **Falcon-7B or Qwen2.5-7B (Apache-2.0), PyTorch** | Zero-shot detectors need an observer LLM; Apache-2.0 keeps the open core license-clean (avoids Llama's MAU clause). Runs on one mid GPU; CPU fallback for small inputs. |
| Image detectors | **Open HF image-forensics model(s)** + opt-in **Reality Defender / Hive** adapters | Seed with open detectors so the core stands alone; third-party APIs are adapters (RD has a free dev tier). |
| Calibration | **pure-Python: Platt (Newton) + isotonic (PAVA) + split-conformal** | Hand-verifiable and dependency-free, so the calibration core imports into `fusion` with zero heavy deps. Chosen over sklearn/MAPIE — ADR-0004. |
| Fusion meta-model | **Logistic regression / gradient-boosted stacker (scikit-learn)** | Interpretable stacked ensemble over signal features — the proven pattern (MOSAIC, authio's 12-model meta-classifier). Interpretability *is* a feature here. |
| Service API | **FastAPI + Pydantic v2 + Uvicorn** | Async, typed, OpenAPI for free; same language as the core, so the SDK and service share models. |
| Async media jobs | **Redis + Arq** | Image/video inference is slow and GPU-bound; a queue keeps `/detect` responsive and is the seam video needs later. |
| Storage | **Postgres 16** (verdict ledger + audit) + **S3-compatible object store** (media blobs) | Durable, queryable, auditable verdict history; blobs out of the DB. |
| Explanation | **Claude — `claude-sonnet-5` (cheap path `claude-haiku-4-5-20251001`)** | Renders the structured evidence into a plain-English report. **A narrator, never a detector** — it never sees a vote in the verdict. |
| SDK & CLI | **`bonafide` Python package** + a thin HTTP client mirroring it; **Typer** CLI | Engine-first, per the product decision: developers integrate the library; the CLI is the first "UI." |
| Console (M5+) | **Next.js + TypeScript** | The hosted-product evolution ("option 1"). |
| Tooling | **`uv` + Docker** (GPU containers for model workers) | Fast, reproducible envs; model inference isolated in GPU containers. |

## 4. The Evidence Fusion & Calibration model — get this exactly right

This is the heart. Everything else is plumbing around it.

**The problem it solves.** Signals disagree, come at different scales, and carry radically different trust. A valid C2PA signature from a trusted CA is near-conclusive. A perplexity score is a hint. Naive averaging (what a thin aggregator does) lets three weak hints outvote one cryptographic proof — or lets one loud detector dominate. We need a combination that (a) lets a genuinely conclusive signal win, (b) lets weak signals *accumulate* without pretending to certainty, (c) produces a **calibrated** probability, and (d) **abstains** when it should.

**The model: weighted evidence fusion in log-odds space, then calibrate, then gate.**

Every detector emits not a boolean but an **`Evidence`** record carrying a **log-likelihood-ratio (LLR)** toward AI, and a **reliability weight**. Fusion is Bayesian log-odds accumulation over *applicable* signals; a learned calibrator maps the fused score to a true probability; a conformal gate decides whether we're allowed to answer at all.

```
prior:        L0   = logit(P_prior(AI))                # configurable per domain
per signal i: ℓ_i  = w_i · s_i                         # only if signal is applicable
                     s_i = log[ p(evidence_i | AI) / p(evidence_i | human) ]   # the LLR
                     w_i ∈ [0,1]                        # reliability in THIS context (OOD ⇒ ~0)
fused:        L    = L0 + Σ_i ℓ_i                       # applicable signals only
calibrate:    p_ai = g(L)                              # g = isotonic / temperature, fit on labeled data
gate:         decision, coverage = conformal(L, α)     # {ai} / {human} / abstain, guaranteed 1−α coverage
```

**Why log-odds + reliability weights.** Independent evidence combines additively in log-odds (Bayes). A conclusive signal has a huge `|s_i|`, so it dominates *correctly* — but it's still expressed as evidence, so a *conflicting* conclusive signal (tampered manifest vs. forensic tells) doesn't get silently overridden; both land in the record and trip conflict detection. Weak signals have small `|s_i|` and accumulate honestly. `w_i` is where the OOD guard and per-signal trust live: flag a text as out-of-distribution (too short, heavy paraphrase) and its detectors' `w_i → 0`, so they *stop voting* instead of voting wrong.

**Signal tiers** (drive default reliability and conflict logic):

```python
class SignalTier(Enum):
    CONCLUSIVE = "conclusive"   # cryptographic provenance, verified watermark
    STRONG     = "strong"       # supervised detector, in-distribution
    WEAK        = "weak"        # zero-shot / heuristic / stylometric
    CONTEXT    = "context"      # metadata hints (EXIF, filename) — never decisive
```

**The two core types** (this is the contract the whole system is built around):

```python
@dataclass(frozen=True)
class Evidence:
    signal_id: str            # "c2pa.manifest", "synthid.text", "binoculars", ...
    applicable: bool          # did this signal apply to this input at all?
    llr: float                # log-likelihood ratio toward AI (>0 AI, <0 human); 0 if n/a
    reliability: float        # w_i ∈ [0,1] — trust in this signal for THIS input
    tier: SignalTier
    detail: dict              # signal-specific payload: cert chain, score, span heat-map, ...
    model_version: str        # exact version/hash — reproducibility

@dataclass(frozen=True)
class Verdict:
    p_ai: float                       # calibrated P(AI-generated) ∈ [0,1]
    ci: tuple[float, float]           # confidence interval on p_ai
    decision: Literal["ai","human","abstain"]
    coverage: float                   # conformal guarantee level used (1 − α)
    evidence: list[Evidence]          # EVERY signal, incl. non-applicable — full transparency
    conflicts: list[str]              # human-readable notes where signals disagreed
    modality: Modality
    engine_version: str               # bonafide + fused-model version — reproducible & auditable
```

**Fusion function** (pure, deterministic given its inputs):

```python
def fuse(evidence: list[Evidence], *, prior_ai: float,
         calibrator: Calibrator, conformal: ConformalGate,
         alpha: float = 0.05) -> Verdict:
    applicable = [e for e in evidence if e.applicable]
    if not applicable:
        return _abstain(evidence, reason="no applicable signals")
    L  = logit(prior_ai) + sum(e.reliability * e.llr for e in applicable)
    p  = calibrator.transform(L)                       # calibrated probability
    ci = calibrator.interval(L)                        # CI from the calibration set
    decision, coverage = conformal.decide(L, alpha)    # may return "abstain"
    conflicts = detect_conflicts(applicable)           # e.g. CONCLUSIVE signals of opposite sign
    if conflicts and decision != "abstain":
        decision = "abstain"                           # honesty beats a coin-flip
    return Verdict(p_ai=p, ci=ci, decision=decision, coverage=coverage,
                   evidence=evidence, conflicts=conflicts, ...)
```

**Invariants (the design's promises):**
- A signal with `applicable=False` contributes **exactly 0** to `L` — absence never penalizes (no C2PA manifest ≠ "human").
- **No signal may emit a raw boolean.** Everything is `(llr, reliability)` so fusion is uniform and one code path calibrates all modalities.
- `reliability → 0` under an OOD flag → the signal gracefully self-mutes.
- **Abstain when:** no applicable signals · conformal set is ambiguous at level α · OOD across the board · CONCLUSIVE signals conflict. Abstention is a first-class answer, tracked as a metric, not a failure.
- A `Verdict` is a **pure function** of `(input bytes, signal set + versions, calibrator version, prior, α)`. This is what makes it reproducible and auditable — and litigation-defensible.

**Calibration & abstention, concretely.** Fit `g` (isotonic regression; temperature scaling as the low-data fallback) on a labeled calibration set so predicted probability matches empirical frequency (target ECE < 0.05). For abstention use **split conformal prediction**: compute nonconformity scores on the calibration set; at target error α, form a prediction set; emit the label only if the set is the singleton `{ai}` or `{human}`, else abstain. This yields a **distribution-free guarantee** that the true label falls in the emitted set ≥ 1−α of the time — the rigorous version of "know what you don't know." The FPR operating point (e.g. ≤1% on human text) is chosen on the calibration curve and enforced at the gate.

## 5. Architecture

**Pattern: Hexagonal (ports & adapters).** The core — router + fusion + calibration — depends only on the abstract `Signal` **port**. Every detector, provenance checker, and watermark reader is an **adapter** behind that port. A new modality or a new detector is a new adapter and a registry entry; **the core never changes.** This is precisely what turns "supports all media eventually" from a slogan into a property of the code.

```
                          ┌──────────────────────────────────────────────┐
  input bytes  ─────────▶ │                 BONAFIDE CORE                   │
  (+ optional context)    │  ┌───────────┐    ┌──────────────────────┐    │
                          │  │ Ingest &  │    │   Signal Registry     │    │
                          │  │ Modality  │──▶ │   (signals per        │    │
                          │  │ Router    │    │    modality)          │    │
                          │  └───────────┘    └──────────┬───────────┘    │
                          │                      fan-out │ → Evidence[]    │
                          │                              ▼                 │
                          │  ┌────────────────────────────────────────┐   │
                          │  │  FUSION · CALIBRATION · CONFORMAL GATE   │   │ ◀── §4, the moat
                          │  │              → Verdict                   │   │
                          │  └───────────────────┬────────────────────┘   │
                          │            ┌─────────┴──────────┐              │
                          │            ▼                    ▼              │
                          │      Explanation           Verdict Ledger      │
                          │   (Claude, optional)      (Postgres · audit)    │
                          └──────────────┬──────────────────┬─────────────┘
             PORT: Signal ───────────────┴──────────────────┴──────────
   ┌───────────────┬──────────────────┬───────────────────┬──────────────────┐
   ▼               ▼                  ▼                   ▼                  ▼
 ADAPTERS:     Provenance          Watermark          ML Detectors       3rd-party APIs
   C2PA verify   (SynthID-Text)     (Binoculars,       (Reality Defender,
   (c2pa-rs)                         Fast-DetectGPT,    Hive) — opt-in
                                     image-forensics)
   ─────────────────────────────────────────────────────────────────────────
 DELIVERY:   Python SDK   │   Typer CLI   │   FastAPI service   │   Next.js console (M5+)
```

Data flow: bytes → router picks the modality + its registered signals → signals run (concurrently; slow ones via the queue) and each returns `Evidence` → fusion produces a `Verdict` → optionally narrated by Claude and persisted to the ledger → returned via SDK/CLI/API.

## 6. Core systems

### 6.1 The `Signal` port (plugin interface)
Every detector implements one tiny protocol. This is the extension point the whole "any media" promise rests on.

```python
class Signal(Protocol):
    id: str
    modalities: frozenset[Modality]          # {TEXT}, {IMAGE}, {AUDIO}, {VIDEO}
    tier: SignalTier
    def applies_to(self, ctx: InputContext) -> bool: ...     # cheap gate
    def analyze(self, ctx: InputContext) -> Evidence: ...    # the work → Evidence

class SignalRegistry:
    def register(self, signal: Signal) -> None: ...
    def for_modality(self, m: Modality) -> list[Signal]: ...
```
Registration is declarative (entry-points/decorator), so third parties ship signals as pip packages.

### 6.2 Provenance adapter — C2PA (`c2pa-python`)
Validates the manifest, checks the signature against the bundled CAI + C2PA-conformance trust lists, and inspects `c2pa.actions` / `claim_generator` across **every** manifest in the store (AI-origin evidence often hides in an *ingredient* after an edit). Maps to `Evidence` — and because a manifest is a *signed claim*, not ground truth, reliability is **direction-aware and per-manifest** (ADR-0003):
- Trust-listed signer, and the claim sits on the **active** manifest it vouches for → CONCLUSIVE at full reliability: a known AI generator / `trainedAlgorithmicMedia` → large `+llr`; a hardware-camera capture with intact hash bindings → large `−llr`.
- Untrusted signer (or a claim laundered in via an untrusted ingredient) → an **AI** claim is still strong (a statement against interest, discounted to 0.85); a **capture/human** claim counts for **exactly zero** — self-signing "a camera took this" is the canonical C2PA spoof, so it is recorded in the evidence trail and weighed at nothing.
- No manifest → `applicable=False` (contributes 0). **Never** read as "human."
- Manifest present but signature invalid / hash-mismatch (tampering or malformation) → contributes 0 and `detail` feeds `conflicts`, forcing `abstain`.
- Manifest valid but carrying **no** origin markers → `llr = 0`. A signed manifest is not evidence of human origin; AI-origin assertions can simply be omitted.

### 6.3 Watermark adapter — SynthID-Text
Weighted-Mean detector (no training) + Bayesian detector (trained) over supported models → CONCLUSIVE-tier `+llr` when a watermark is found; `applicable=False` when the text is too short or the model family isn't covered. Image/audio/video SynthID detection is currently gated to Google's portal, so for those modalities Bonafide emits a **CONTEXT** signal ("SynthID check recommended — verify at Google's SynthID Detector") until a programmatic API exists. Tracked in §9.

### 6.4 Text detector ensemble
Binoculars (cross-perplexity between two observer LLMs) + Fast-DetectGPT (conditional-probability curvature) share one loaded scoring LLM. Each converts its score to an `llr` via a fitted logistic link; features (perplexity, cross-perplexity, curvature) also feed the meta-stacker. WEAK/STRONG tier. Paraphrase/adversarial edits raise the OOD flag (§6.6) → reliability drops.

### 6.5 Image detector ensemble
Open forensic model(s) (frequency-domain + diffusion-artifact detectors) as STRONG-tier signals, plus opt-in Reality Defender / Hive adapters. Returns `llr` + a region heat-map in `detail` for the evidence report.

### 6.6 Modality router & OOD guard
Sniffs the type (magic bytes / MIME), routes to that modality's registered signals, and runs an **out-of-distribution guard** that sets `reliability → 0` when an input is outside a signal's competence (text under N tokens, unsupported image codec, heavy re-encoding). The guard is why Bonafide degrades to *abstain* instead of to *confidently wrong* — the single biggest lesson from the ~50% real-world accuracy drop the incumbents hide.

### 6.7 Fusion engine
Implements §4 exactly. Pure, deterministic, unit-tested against hand-computed log-odds. Owns conflict detection and the conformal gate.

### 6.8 Calibration & evaluation harness *(this is an ML product — evaluation is a first-class subsystem)*
- **Benchmarks:** RAID (text robustness, incl. adversarial/paraphrase splits) + a curated image set spanning generators (Midjourney/Imagen/SD/Flux) and real photography.
- **Metrics:** ECE (calibration), FPR at the chosen operating point (esp. a *non-native-English* slice — the harm we refuse to cause), abstention rate, and conformal coverage vs. target.
- **Artifacts:** a fitted `CalibrationArtifact` (calibrator + conformal gate + fit-time prior), content-fingerprinted and stamped into every `Verdict`. The prior is locked at inference so a caller can't silently re-scale a calibrated model.
- **Coverage is the gate's, not a knob:** a `Verdict`'s reported coverage comes from the conformal gate that actually decided (`1-α_fit`), never from a runtime `α` — reporting the latter would *overstate* the guarantee, the precise dishonesty this project rejects.
- **Train/test discipline:** the coverage guarantee and calibration assume the calibration set is exchangeable with, and not reused as, the evaluation/production data. Held-out evaluation is the honest path (ADR-0004).

### 6.9 Verdict ledger & reproducibility
Every verdict persists to Postgres with the input hash, the full `evidence[]`, and the exact versions of every signal + calibrator. Any verdict can be **replayed** and explained months later — essential for audits, disputes, and the EU AI Act paper trail.

### 6.10 API / SDK surface
```python
# SDK
from bonafide import detect
v = detect("essay.txt", prior_ai=0.5, alpha=0.05)   # -> Verdict
print(v.decision, v.p_ai, v.ci)
for e in v.evidence: print(e.signal_id, e.llr, e.reliability)
```
```
# HTTP
POST /detect            multipart (file|text) + params  -> Verdict JSON
GET  /verdict/{id}      replay a stored verdict (audit)
GET  /signals           list registered signals + versions
GET  /healthz
```
```bash
# CLI
bonafide detect photo.jpg --explain      # verdict + Claude-rendered evidence report
```

## 7. Interface & UX

v1 is engine-first, so the primary "interface" is the **SDK + CLI + API** above. The design principle carries into the eventual console (M5): the one loop the product lives on is **transparency** — *upload/paste → an evidence card*, where the headline is the calibrated verdict and every signal is a readable row (*what it is · what it found · how far it moved the needle*), with a dignified, unmistakable **"Insufficient evidence — abstain"** state. Users never get a naked number; they always get the *why*. That honesty is the brand.

## 8. Milestones

Top-down and independently runnable — each one you can open and test.

- **M0 — Skeleton & it runs.** `detect()` flows bytes → router → a single stub `Signal` → `fuse()` → `Verdict`, exposed via the Typer CLI and a FastAPI `/detect`. Proves the whole spine end-to-end. Verdict is honest-but-dumb (one signal, usually abstains).
- **M1 — Provenance-first images.** Real C2PA adapter (`c2pa-python`): validate manifest + trust list, map to CONCLUSIVE evidence; image ingest. Bonafide now gives a *correct, high-trust* verdict for any Content-Credentials image and abstains otherwise. Proves the fusion core with a real conclusive signal — a low-risk, high-credibility first capability given C2PA's ubiquity.
- **M2 — Text ensemble + calibration (the thesis).** Binoculars + Fast-DetectGPT + SynthID-Text over a scoring LLM; fit the isotonic calibrator + conformal gate on RAID; the eval harness reports ECE / FPR / coverage. Delivers calibrated, abstaining text verdicts — the actual differentiator — with the discipline to prove it.
- **M3 — Image ML ensemble + multi-signal fusion.** Add open image-forensics detector(s) so images fuse *provenance + watermark + ML*, and turn on conflict surfacing (C2PA vs. pixels). Proves cross-tier fusion — the full showcase.
- **M4 — Explanation, ledger & hardening.** Claude-rendered evidence reports; Postgres verdict ledger + reproducibility; OOD guards tuned; Redis/Arq async queue for slow media; opt-in Reality Defender adapter. Proves auditability, honesty, and production extensibility.
- **M5 — Thin hosted API + demo console.** Deploy the FastAPI service and a minimal Next.js console (upload → evidence card). The first *public, visible* Bonafide — the pivot toward the hosted product ("option 1").
- **M-later (planned, not v1) — Audio & Video.** **Audio:** voice-clone + AI-music (Suno/Udio) adapters using MERT features and an acoustic-signature ensemble — reuses fusion/calibration unchanged. **Video:** frame-sampling that reuses the image pipeline + the audio-track pipeline + a temporal-consistency signal, orchestrated over the Arq queue; C2PA 2.3 live-stream support. The plugin architecture + queue mean these are *new adapters*, not a rewrite — this is the whole point of §5.

## 9. Risks / open questions

- **Text false positives harm real people (ethical + legal).** *The* headline risk — it's what's killing the incumbents. → Calibration + conformal abstention + an "evidence, not accusation" API contract + a configurable FPR operating point + an explicit non-native-English eval slice + no automated decisions. This risk *is* the product strategy.
- **Detectors lose ~50% accuracy in the wild / under attack.** → OOD guards zero out unreliable signals; abstain rather than guess; lean on hard provenance where present; publish honest benchmark numbers (credibility as a moat).
- **Distribution shift — new generators weekly.** → Versioned calibrators + a scheduled re-calibration pipeline; conformal guarantees hold under exchangeability; drift monitoring on the ledger.
- **Provenance is strippable and partially spoofable.** → Absence ⇒ `applicable=False` (never "human"); always validate the trust list + hash bindings; surface *tampered* manifests as `conflicts`, and give *untrusted* claims direction-aware weight (untrusted capture claims count zero — ADR-0003) — never silent truth. Trust is judged per-manifest, so a spoofed claim cannot launder itself into credibility by riding as an ingredient inside a trust-listed re-export.
- **Third-party API licensing & cost.** Hive/Sensity are enterprise-only. → Opt-in adapters only; the OSS core is fully functional without any of them.
- **SynthID detection for image/audio/video is gated to Google.** → Text via the OSS detector now; other modalities emit a CONTEXT "verify at SynthID Detector" signal until a programmatic API appears. Track Google's rollout.
- **Scoring-LLM license (Llama MAU clause).** → Default to Apache-2.0 observer models (Falcon/Qwen); document the choice; verify each detector repo's own license at integration.
- **"Number one in the world" is a trust + distribution problem, not only a tech one.** → Open-core for developer adoption and credibility (publish the eval harness), *honesty* as the brand, and EU AI Act Article 50 (effective 2 Aug 2026) as the enterprise wedge. (Strategy, not a v1 deliverable — noted so the build stays aimed at it.)
- **Open questions to resolve during M2–M4:** default `P_prior(AI)` per domain, and whether per-signal reliability weights are expert-configured or learned. → Start expert-configured; learn from labeled data as the ledger accumulates.

## 10. References

*Prior art, standards, and libraries — verified 2026-07-08.*

**Standards & provenance**
- C2PA / Content Credentials — spec 2.x (2.3/2.4), CAI: <https://c2pa.org/> · `c2pa-python` (Rust `c2pa-rs` binding): <https://github.com/contentauth/c2pa-python>, <https://pypi.org/project/c2pa-python/>
- SynthID (DeepMind) — detector portal: <https://blog.google/innovation-and-ai/products/google-synthid-ai-content-detector/> · OSS text detector: <https://github.com/google-deepmind/synthid-text> (in HF `transformers` ≥4.46)
- EU AI Act **Article 50** transparency obligations (effective 2 Aug 2026): <https://artificialintelligenceact.eu/article/50/>

**Detectors & benchmarks**
- Binoculars (zero-shot): <https://arxiv.org/pdf/2401.12070> · Fast-DetectGPT: <https://github.com/baoguangsheng/fast-detect-gpt> · RAID benchmark (robust eval) · MOSAIC ensemble: <https://arxiv.org/pdf/2409.07615>
- Calibration & abstention — isotonic/Platt/temperature scaling; split conformal prediction: <https://arxiv.org/abs/2402.05806>

**Competitive landscape (what to beat / borrow)**
- Text: Pangram (SOTA FPR ~0.01%), GPTZero, Originality.ai, Turnitin (being *disabled* by universities), OpenAI classifier (*retired*).
- Media: Reality Defender (broadest, free 50/mo dev tier), Sensity, Hive (enterprise-only), Pindrop (audio), Intel FakeCatcher.
- Audio/music: ACRCloud, authio (12-model meta-classifier), letssubmit bAbI v2 (MERT+LogReg).
- **Aggregator prior art / the gap:** Eden AI proxies multiple detectors but only *compares* scores — nobody does calibrated multi-signal *fusion* with provenance + abstention. **That gap is Bonafide.**

**Models (for the Claude explanation layer)**
- `claude-sonnet-5` (reports) · `claude-haiku-4-5-20251001` (cheap path). Narrator only — never a detector.
