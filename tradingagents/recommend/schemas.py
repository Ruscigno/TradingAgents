"""Pydantic schemas for LLM structured output across cascade stages.

Each LLM-driven stage (2 through 6) prompts the model to return one of these
schemas. Stage names match the keys in ``StageName`` from
:mod:`tradingagents.recommend.types`.

We intentionally use a small, shared shape (verdict + confidence + narrative)
across stages so the orchestrator can treat them uniformly. Stage-specific
extras (e.g. ``direction`` for market analysis) are added per schema.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Verdict = Literal["PASS", "NO_PASS"]
Direction = Literal["long", "short", "none"]


class MarketVerdict(BaseModel):
    """Etapa 2 — Market Analyst output.

    The LLM judges whether the technical setup justifies entering a trade today.
    """

    verdict: Verdict = Field(
        description=(
            "PASS if the technical setup justifies further analysis; "
            "NO_PASS otherwise."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Your certainty in the verdict, between 0 (uncertain) and 1 (very certain).",
    )
    direction: Direction = Field(
        description=(
            "Trade direction implied by the technical setup. "
            "Use 'none' if no clear direction or if NO_PASS."
        ),
    )
    narrative: str = Field(
        description="One short paragraph explaining the verdict.",
        max_length=1500,
    )


class _AnalystVerdict(BaseModel):
    """Base shape for stage-3 analyst verdicts.

    Sentiment / News / Fundamentals all use the same fields. Separate classes
    let the LLM see schema names that hint at the analyst role, and let tests
    assert on type identity.
    """

    verdict: Verdict = Field(
        description="PASS if your analysis supports advancing this ticker; NO_PASS otherwise.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Your certainty in the verdict, 0 (coin-flip) to 1 (very certain).",
    )
    narrative: str = Field(
        description="One short paragraph (max ~3 sentences) explaining the verdict.",
        max_length=1500,
    )


class SentimentVerdict(_AnalystVerdict):
    """Etapa 3 — Sentiment / social-media analyst output.

    Judges whether the *tone* of recent news + market chatter is constructive.
    Uses the same source data as :class:`NewsVerdict` but a sentiment-focused
    prompt.
    """


class NewsVerdict(_AnalystVerdict):
    """Etapa 3 — News analyst output.

    Judges whether recent news (specific to the ticker + relevant macro)
    is supportive of the trade thesis.
    """


class FundamentalsVerdict(_AnalystVerdict):
    """Etapa 3 — Fundamentals analyst output.

    Judges whether the company's financial profile (valuation, margins,
    growth, debt) supports the trade thesis.
    """
