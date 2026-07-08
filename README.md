# Bedrock

**A calibrated, provenance-first framework for detecting AI-generated media — honest about what it knows, and about what it doesn't.**

In an age of deepfakes and synthetic everything, the question "was this made by a human or by AI?" matters more each day. The existing detectors answer it with confident binaries that are wrong often enough — and in biased enough ways — that their own customers are turning them off.

Bedrock takes a different bet. It treats detection as **evidence aggregation under uncertainty**: it fuses hard provenance (C2PA / Content Credentials), watermarks (SynthID), and an ensemble of ML detectors into **one calibrated probability with a confidence interval and a readable evidence report** — and it **abstains** when the evidence doesn't support a confident call. Evidence, never an accusation.

- **The moat:** a Bayesian evidence-fusion + calibration engine that combines cryptographic proof and noisy ML scores into one honest, auditable verdict.
- **The architecture:** ports-and-adapters, so any detector for any medium plugs in — **text and images** first, **audio and video** as drop-in adapters later, with zero core changes.
- **The wedge:** the only detector you can actually trust — plus a regulatory tailwind (EU AI Act Article 50, effective 2 Aug 2026).

See [DESIGN.md](DESIGN.md) for the full technical design.

**Status:** M0 scaffold — the engine spine runs end-to-end (CLI + API + fusion core) with a
placeholder detector. See [ROADMAP.md](ROADMAP.md) for the plan.

---

## Run it

**Prerequisites:** Python ≥ 3.11 and [`uv`](https://docs.astral.sh/uv/) (check: `python --version`, `uv --version`).

```bash
uv venv                          # create .venv
uv pip install -e ".[dev]"       # install (core is pure-Python; ML/provenance are extras)

# CLI — analyze text or a file
uv run bedrock detect --text "The quick brown fox jumps over the lazy dog near the bridge."
uv run bedrock detect ./some-file.txt --json

# API — starts on http://localhost:8000 (interactive docs at /docs)
uv run uvicorn bedrock.service.app:app --reload
```

Today every verdict is honest-but-shallow: only a placeholder text heuristic is wired, so most
inputs correctly return **abstain**. Real signals (C2PA, then the text ensemble) arrive in M1–M2.

### Commands

| Command | What it does |
|---|---|
| `uv run bedrock detect <src>` | Run a detection and print the evidence report |
| `uv run uvicorn bedrock.service.app:app --reload` | Run the HTTP API |
| `uv run pytest` | Run the tests |
| `uv run ruff check . && uv run mypy src tests` | Lint + typecheck |
| `uv build` | Build the wheel/sdist |

## Optional extras (installed per milestone)

| Extra | Adds | Milestone |
|---|---|---|
| `.[provenance]` | C2PA / Content Credentials verification (`c2pa-python`) | M1 |
| `.[calibrate]` | isotonic calibration + conformal abstention | M2 |
| `.[text]` | zero-shot text detectors + SynthID-Text (`torch`, `transformers`) | M2 |
| `.[image]` | image forensic detectors | M3 |
| `.[explain]` | Claude-rendered evidence reports | M4 |
| `.[all]` | everything above | — |

## Project docs

| Doc | What's in it |
|---|---|
| [DESIGN.md](DESIGN.md) | The full design and rationale — the single source of truth. |
| [ROADMAP.md](ROADMAP.md) | The milestone checklist (the plan + what's done). |
| [docs/adr/](docs/adr/) | Architecture Decision Records. |

## Tech stack

Python 3.11+ · Pydantic v2 · FastAPI + Uvicorn · Typer CLI · `uv` + hatchling. ML/provenance
(`c2pa-python`, `transformers`, `torch`, scikit-learn, conformal) load as optional extras.

## License

MIT — see [LICENSE](LICENSE).
