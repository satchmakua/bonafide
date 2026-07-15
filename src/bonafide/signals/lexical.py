"""M0 placeholder text signal — exists to exercise the pipeline end-to-end, nothing more.

It computes a crude type-token ratio (lexical diversity): very repetitive text leans *very
weakly* toward AI. It is intentionally ``WEAK`` tier with low reliability so the engine
abstains rather than over-claims. It gets retired in M2 when the real ensemble (Binoculars,
Fast-DetectGPT, SynthID-Text) arrives. Do not take its verdicts seriously.
"""

from __future__ import annotations

import re

from ..types import Evidence, InputContext, Modality, SignalTier

_WORD = re.compile(r"[A-Za-z']+")
MIN_TOKENS = 20  # below this the sample is out of distribution → not applicable


class LexicalHeuristicSignal:
    id = "lexical-heuristic"
    modalities = frozenset({Modality.TEXT})
    tier = SignalTier.WEAK
    version = "ttr-v0"

    def _tokens(self, text: str) -> list[str]:
        return [t.lower() for t in _WORD.findall(text)]

    def applies_to(self, ctx: InputContext) -> bool:
        return ctx.text is not None and len(self._tokens(ctx.text)) >= MIN_TOKENS

    def analyze(self, ctx: InputContext) -> Evidence:
        text = ctx.text or ""
        tokens = self._tokens(text)
        ttr = len(set(tokens)) / len(tokens)
        # Low diversity -> mild +llr (AI-ish); high diversity -> mild -llr. Small magnitude.
        llr = (0.5 - ttr) * 2.0
        return Evidence(
            signal_id=self.id,
            applicable=True,
            llr=llr,
            reliability=0.25,
            tier=self.tier,
            model_version=self.version,
            detail={"type_token_ratio": round(ttr, 4), "tokens": len(tokens)},
        )
