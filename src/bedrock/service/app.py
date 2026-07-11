"""FastAPI service exposing the engine over HTTP.

Run: ``uv run uvicorn bedrock.service.app:app --reload``. Interactive docs at ``/docs``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .._version import __version__
from ..engine import default_registry, detect_bytes, detect_text, engine_version
from ..types import Verdict

app = FastAPI(
    title="Bedrock",
    version=__version__,
    description="Calibrated, provenance-first detection of AI-generated media.",
)

# Uvicorn/Starlette impose no body-size limit; an unbounded read() of an attacker-sized
# upload exhausts memory. Generous for any single image or text asset.
MAX_UPLOAD_BYTES = 64 * 2**20


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "engine": engine_version()}  # uncalibrated default


@app.get("/signals")
def signals() -> list[dict[str, object]]:
    """List the registered detection signals and their versions."""
    return [
        {
            "id": s.id,
            "tier": s.tier.value,
            "modalities": sorted(m.value for m in s.modalities),
            "version": s.version,
        }
        for s in default_registry().signals
    ]


@app.post("/detect")
async def detect_endpoint(
    text: Annotated[str | None, Form()] = None,
    file: Annotated[UploadFile | None, File()] = None,
    prior: Annotated[float, Form()] = 0.5,
    alpha: Annotated[float, Form()] = 0.05,
) -> Verdict:
    """Analyze either a ``text`` field or an uploaded ``file`` and return a Verdict."""
    if file is not None:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds the {MAX_UPLOAD_BYTES // 2**20} MiB upload limit.",
            )
        return detect_bytes(data, path=file.filename, prior_ai=prior, alpha=alpha)
    if text is not None:
        return detect_text(text, prior_ai=prior, alpha=alpha)
    raise HTTPException(status_code=422, detail="Provide either a 'text' field or a 'file' upload.")
