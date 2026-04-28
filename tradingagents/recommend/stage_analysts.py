"""Etapa 3 — Sentiment + News + Fundamentals analysts + composite agg.

Para cada candidata vinda da Etapa 2:

1. **Fetch de dados** (sem LLM, barato):
   - Notícias recentes da ticker via yfinance (compartilhado entre sentiment + news)
   - Fundamentos via yfinance

2. **Três chamadas LLM em paralelo** (Kimi K2.6, structured output):
   - Sentiment analyst — tom das notícias + chatter
   - News analyst — impacto factual das notícias
   - Fundamentals analyst — valuation / saúde financeira

3. **Composite + veto** (determinístico):
   - ``composite = w_market * c_market + w_sentiment * c_sentiment + w_news * c_news + w_fundamentals * c_fundamentals``
   - PASS exige composite ≥ 0.65 **e** nenhum NO_PASS com conf ≥ 0.85 (veto).

Pesos default (estudo 01 D2): market 0.35, sentiment 0.15, news 0.20, fundamentals 0.30.

Custo estimado: ~$0.015 por ticker (3 calls × ~$0.005 + yfinance grátis).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

from pydantic import BaseModel

from tradingagents.agents.utils.structured import bind_structured
from tradingagents.recommend.llm import LLMConfig, build_llm
from tradingagents.recommend.schemas import (
    FundamentalsVerdict,
    NewsVerdict,
    SentimentVerdict,
)
from tradingagents.recommend.types import Candidate, Elimination, StageOutcome


log = logging.getLogger(__name__)
STAGE = "analysts"

DEFAULT_COMPOSITE_MIN = 0.65
DEFAULT_VETO_CONFIDENCE = 0.85
DEFAULT_NEWS_LOOKBACK_DAYS = 7

# Decisão D2 do estudo 01.
DEFAULT_WEIGHTS: dict[str, float] = {
    "market": 0.35,
    "sentiment": 0.15,
    "news": 0.20,
    "fundamentals": 0.30,
}


# ── Public API ──────────────────────────────────────────────────────────────


def run(
    candidates: list[Candidate],
    *,
    as_of_date: str,
    llm: Any | None = None,
    llm_config: LLMConfig | None = None,
    weights: dict[str, float] | None = None,
    composite_min: float = DEFAULT_COMPOSITE_MIN,
    veto_confidence: float = DEFAULT_VETO_CONFIDENCE,
    news_lookback_days: int = DEFAULT_NEWS_LOOKBACK_DAYS,
    max_candidate_workers: int = 4,
    news_fetcher: Callable[[str, str, str], str] | None = None,
    fundamentals_fetcher: Callable[[str, str], str] | None = None,
) -> StageOutcome:
    """Etapa 3 — três analistas em paralelo + composite agregado.

    Args:
        candidates: sobreviventes da Etapa 2.
        as_of_date: data ``YYYY-MM-DD``; usada pra delimitar a janela de notícias.
        llm: LangChain chat client. Se None, instancia via ``build_llm``.
        llm_config: usado quando ``llm`` é None.
        weights: pesos do composite. Default :data:`DEFAULT_WEIGHTS`.
        composite_min: threshold de PASS para o composite.
        veto_confidence: NO_PASS com conf ≥ veto_confidence elimina a candidata
            mesmo que o composite passe.
        news_lookback_days: janela de notícias passada pro yfinance.
        max_candidate_workers: paralelismo entre tickers.
        news_fetcher / fundamentals_fetcher: injeção pra testes — assinaturas
            compatíveis com :func:`get_news_yfinance` e :func:`get_fundamentals`.
    """
    t0 = time.monotonic()

    if not candidates:
        return StageOutcome(stage_name=STAGE, candidates=[], duration_s=0.0)

    if llm is None:
        llm = build_llm(llm_config)

    structured_clients = _build_structured_clients(llm)
    if structured_clients is None:
        log.error(
            "stage_analysts: LLM provider does not support structured output; "
            "marking all %d candidates as NO_PASS",
            len(candidates),
        )
        out = [
            _clone_with_elimination(c, "structured output unsupported by LLM provider", 0.0)
            for c in candidates
        ]
        return StageOutcome(stage_name=STAGE, candidates=out, duration_s=time.monotonic() - t0)

    weights = weights or DEFAULT_WEIGHTS
    news_fn = news_fetcher or _default_news_fetcher
    fund_fn = fundamentals_fetcher or _default_fundamentals_fetcher

    def _process(cand: Candidate) -> Candidate:
        return _judge_one(
            cand,
            as_of_date=as_of_date,
            structured_clients=structured_clients,
            weights=weights,
            composite_min=composite_min,
            veto_confidence=veto_confidence,
            news_lookback_days=news_lookback_days,
            news_fn=news_fn,
            fund_fn=fund_fn,
        )

    out_by_ticker: dict[str, Candidate] = {}
    with ThreadPoolExecutor(max_workers=max_candidate_workers) as pool:
        futures = {pool.submit(_process, c): c for c in candidates}
        for fut in as_completed(futures):
            original = futures[fut]
            try:
                out_by_ticker[original.ticker] = fut.result()
            except Exception as exc:
                log.warning(
                    "stage_analysts: %s — unhandled error: %s; marking NO_PASS",
                    original.ticker, exc,
                )
                out_by_ticker[original.ticker] = _clone_with_elimination(
                    original, f"analyst pipeline failed: {type(exc).__name__}", 0.0,
                )

    out = [out_by_ticker[c.ticker] for c in candidates]

    duration = time.monotonic() - t0
    passed = sum(1 for c in out if c.elimination is None)
    log.info(
        "stage_analysts: %d in, %d kept, %d eliminated, %.1fs",
        len(candidates), passed, len(candidates) - passed, duration,
    )
    return StageOutcome(
        stage_name=STAGE,
        candidates=out,
        duration_s=duration,
        cost_usd=0.0,
    )


# ── Composite math ──────────────────────────────────────────────────────────


def composite_confidence(
    confidences: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    """Weighted average over whatever stage confidences are present.

    Pesos para stages ausentes são redistribuídos proporcionalmente. Útil
    quando algum analista falhou e queremos ainda ter um sinal das demais
    dimensões em vez de descartar tudo. Stages explicitly-zero ainda contam
    e tendem a derrubar a média.
    """
    weights = weights or DEFAULT_WEIGHTS
    relevant = {k: w for k, w in weights.items() if k in confidences}
    total_weight = sum(relevant.values())
    if total_weight == 0:
        return 0.0
    return sum(confidences[k] * (w / total_weight) for k, w in relevant.items())


# ── Per-candidate work ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class _StructuredClients:
    sentiment: Any
    news: Any
    fundamentals: Any


def _build_structured_clients(llm: Any) -> _StructuredClients | None:
    """Bind ``with_structured_output`` for each schema once, reuse across all candidates."""
    sentiment = bind_structured(llm, SentimentVerdict, "recommend.sentiment")
    news = bind_structured(llm, NewsVerdict, "recommend.news")
    fundamentals = bind_structured(llm, FundamentalsVerdict, "recommend.fundamentals")
    if any(c is None for c in (sentiment, news, fundamentals)):
        return None
    return _StructuredClients(sentiment=sentiment, news=news, fundamentals=fundamentals)


def _judge_one(
    cand: Candidate,
    *,
    as_of_date: str,
    structured_clients: _StructuredClients,
    weights: dict[str, float],
    composite_min: float,
    veto_confidence: float,
    news_lookback_days: int,
    news_fn: Callable[[str, str, str], str],
    fund_fn: Callable[[str, str], str],
) -> Candidate:
    # 1. Fetch external data (deterministic, no LLM).
    end_dt = _parse_iso_date(as_of_date)
    start_dt = end_dt - timedelta(days=news_lookback_days)
    start_str, end_str = start_dt.strftime("%Y-%m-%d"), as_of_date

    try:
        news_text = news_fn(cand.ticker, start_str, end_str)
    except Exception as exc:
        return _clone_with_elimination(cand, f"news fetch failed: {exc}", 0.0)
    try:
        fundamentals_text = fund_fn(cand.ticker, as_of_date)
    except Exception as exc:
        return _clone_with_elimination(cand, f"fundamentals fetch failed: {exc}", 0.0)

    # 2. Three structured LLM calls in parallel for this candidate.
    analyst_jobs: dict[str, tuple[Any, type[BaseModel], list[dict[str, str]]]] = {
        "sentiment": (
            structured_clients.sentiment,
            SentimentVerdict,
            _build_sentiment_prompt(cand, as_of_date, news_text),
        ),
        "news": (
            structured_clients.news,
            NewsVerdict,
            _build_news_prompt(cand, as_of_date, news_text),
        ),
        "fundamentals": (
            structured_clients.fundamentals,
            FundamentalsVerdict,
            _build_fundamentals_prompt(cand, as_of_date, fundamentals_text),
        ),
    }

    verdicts: dict[str, BaseModel] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=3) as inner_pool:
        future_to_role = {
            inner_pool.submit(_invoke_with_retry, client, prompt, role): role
            for role, (client, _schema, prompt) in analyst_jobs.items()
        }
        for fut in as_completed(future_to_role):
            role = future_to_role[fut]
            try:
                verdicts[role] = fut.result()
            except Exception as exc:
                errors[role] = f"{type(exc).__name__}: {exc}"

    if errors and len(errors) == 3:
        # Tudo falhou — não há sinal nenhum, descarta com razão clara.
        joined = "; ".join(f"{r}={msg}" for r, msg in errors.items())
        return _clone_with_elimination(cand, f"all analysts failed ({joined})", 0.0)

    # 3. Compose verdict & confidence.
    confidences = dict(cand.confidence_history)  # already has "screener", "market"
    no_pass_with_high_conf: list[tuple[str, float, str]] = []
    for role, verdict in verdicts.items():
        confidences[role] = float(verdict.confidence)
        if verdict.verdict == "NO_PASS" and verdict.confidence >= veto_confidence:
            no_pass_with_high_conf.append((role, verdict.confidence, verdict.narrative))

    composite = composite_confidence(confidences, weights)

    # 4. Apply veto rule: any analyst NO_PASS with conf >= veto_confidence eliminates.
    if no_pass_with_high_conf:
        role, conf, why = no_pass_with_high_conf[0]
        reason = f"veto by {role} (conf={conf:.2f}): {why}"
        new = _clone(cand)
        new.confidence_history = confidences
        new.elimination = Elimination(stage=STAGE, reason=reason, confidence=composite)
        return new

    if composite < composite_min:
        new = _clone(cand)
        new.confidence_history = confidences
        new.elimination = Elimination(
            stage=STAGE,
            reason=f"composite {composite:.3f} below threshold {composite_min}",
            confidence=composite,
        )
        return new

    # 5. PASS — store composite as the stage's confidence, keep individual analyst entries too.
    new = _clone(cand)
    new.confidence_history = confidences
    new.confidence_history[STAGE] = composite
    new.reports[STAGE] = _format_pass_report(verdicts, composite, errors)
    return new


def _invoke_with_retry(structured_llm: Any, prompt: list[dict[str, str]], role: str) -> BaseModel:
    """Invoke once, with a single retry on the first exception.

    Three analysts × N candidates → some transient failures expected.
    A simple retry catches most without compounding latency too much.
    """
    try:
        return structured_llm.invoke(prompt)
    except Exception as exc:
        log.info("stage_analysts: %s first attempt failed (%s); retrying", role, exc)
        return structured_llm.invoke(prompt)


# ── Prompts ─────────────────────────────────────────────────────────────────


_SENTIMENT_SYSTEM = """You are a market sentiment analyst. Read the recent news \
and chatter about a single ticker and judge whether the *tone* is supportive of \
opening a position today. Focus on tone, not facts: are headlines bullish, \
bearish, mixed, or noise? Reply with the structured schema.

Be conservative. Reply NO_PASS when the picture is mixed, when there is no \
clear sentiment signal, or when the tone is plainly negative."""


_NEWS_SYSTEM = """You are a news analyst. Read the recent news about a single \
ticker and judge whether the *content* (events, announcements, macro impact) \
supports advancing this ticker for further analysis. Focus on facts, not tone: \
material announcements, regulatory news, sector dynamics. Reply with the \
structured schema.

Be conservative. NO_PASS when there is no material news, or when the news \
suggests headwinds outweighing tailwinds."""


_FUNDAMENTALS_SYSTEM = """You are a fundamentals analyst. Read the company \
financial profile (valuation, margins, growth, leverage) and judge whether \
the fundamentals support advancing this ticker. Reply with the structured schema.

Be conservative. NO_PASS when the financial profile is weak (overvalued, low \
margins, high debt without growth), or when key data is missing."""


def _build_sentiment_prompt(cand: Candidate, as_of: str, news_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _SENTIMENT_SYSTEM},
        {"role": "user", "content": (
            f"Ticker: {cand.ticker}\nAs-of date: {as_of}\n\n"
            f"Recent news (last 7 days):\n{news_text}\n\n"
            f"Question: is the *sentiment* around {cand.ticker} constructive enough to "
            f"justify advancing this ticker in our cascade? Reply with the structured schema."
        )},
    ]


def _build_news_prompt(cand: Candidate, as_of: str, news_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _NEWS_SYSTEM},
        {"role": "user", "content": (
            f"Ticker: {cand.ticker}\nAs-of date: {as_of}\n\n"
            f"Recent news (last 7 days):\n{news_text}\n\n"
            f"Question: do the *facts* of recent news support advancing {cand.ticker} "
            f"in our cascade? Reply with the structured schema."
        )},
    ]


def _build_fundamentals_prompt(cand: Candidate, as_of: str, fund_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _FUNDAMENTALS_SYSTEM},
        {"role": "user", "content": (
            f"Ticker: {cand.ticker}\nAs-of date: {as_of}\n\n"
            f"Fundamentals snapshot:\n{fund_text}\n\n"
            f"Question: do the fundamentals support advancing {cand.ticker} "
            f"in our cascade? Reply with the structured schema."
        )},
    ]


def _format_pass_report(
    verdicts: dict[str, BaseModel],
    composite: float,
    errors: dict[str, str],
) -> str:
    parts = [f"Composite confidence: {composite:.3f}"]
    for role, v in verdicts.items():
        parts.append(
            f"  - {role}: {v.verdict} (conf={v.confidence:.2f}) — {v.narrative}"
        )
    for role, err in errors.items():
        parts.append(f"  - {role}: ERROR — {err}")
    return "\n".join(parts)


# ── Default data fetchers ───────────────────────────────────────────────────


def _default_news_fetcher(ticker: str, start_date: str, end_date: str) -> str:
    from tradingagents.dataflows.yfinance_news import get_news_yfinance
    return get_news_yfinance(ticker, start_date, end_date)


def _default_fundamentals_fetcher(ticker: str, curr_date: str) -> str:
    from tradingagents.dataflows.y_finance import get_fundamentals
    return get_fundamentals(ticker, curr_date)


def _parse_iso_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


# ── Candidate immutability ──────────────────────────────────────────────────


def _clone(cand: Candidate) -> Candidate:
    return Candidate(
        ticker=cand.ticker,
        confidence_history=dict(cand.confidence_history),
        reports=dict(cand.reports),
        elimination=cand.elimination,
        final_score=cand.final_score,
    )


def _clone_with_elimination(cand: Candidate, reason: str, confidence: float) -> Candidate:
    new = _clone(cand)
    new.elimination = Elimination(stage=STAGE, reason=reason, confidence=confidence)
    return new
