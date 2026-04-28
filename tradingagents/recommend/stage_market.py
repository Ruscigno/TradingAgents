"""Etapa 2 — Market Analyst isolado com structured output (Kimi K2.6).

Recebe candidatas que sobreviveram à Etapa 1 (cada uma já tem
``reports['screener']`` com RSI/VolOsc/DistSMA50). Para cada ticker:

1. Compõe um prompt focado em apenas indicadores técnicos.
2. Chama o LLM com structured output (:class:`MarketVerdict`).
3. Carimba a candidata: PASS → ``confidence_history['market']``;
   NO_PASS → ``elimination``.

Custo estimado por ticker: ~$0.005 (Kimi K2.6 via OpenRouter, prompt ~1.5k
tokens, output ~300 tokens).

Diferente da etapa 3+ que vão usar o ``create_market_analyst`` completo
com tools, a etapa 2 é deliberadamente magra: usa só o que o screener já
calculou, sem fetchar dados extras nem fazer tool-calls. Filtro barato,
proporcional à etapa.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from tradingagents.agents.utils.structured import bind_structured
from tradingagents.recommend.llm import LLMConfig, build_llm
from tradingagents.recommend.schemas import MarketVerdict
from tradingagents.recommend.types import Candidate, Elimination, StageOutcome


log = logging.getLogger(__name__)
STAGE = "market"

# Confidence default mínima para PASS na etapa 2 (estudo 01 D2: thresh ≥ 0.6).
DEFAULT_CONFIDENCE_MIN = 0.6


_SYSTEM_PROMPT = """You are a market technical analyst. Decide whether a single \
ticker shows a setup worth analysing further today, based on the technical \
indicators provided. You must reply with the structured schema.

Be conservative — return PASS only if the indicators clearly suggest a tradable \
setup. If the picture is muddled or ambiguous, return NO_PASS. Use the \
``confidence`` field to express how sure you are: 0 means coin-flip, 1 means \
very high conviction. The threshold for advancing in the cascade is around \
0.6, so use the full range thoughtfully.

Direction:
- "long"  if you would consider a long entry,
- "short" if you would consider a short entry,
- "none"  if no clear direction or NO_PASS.

Keep ``narrative`` to one short paragraph. Reference the specific indicator \
values you weighed."""


def run(
    candidates: list[Candidate],
    *,
    as_of_date: str,
    llm: Any | None = None,
    llm_config: LLMConfig | None = None,
    confidence_min: float = DEFAULT_CONFIDENCE_MIN,
    max_workers: int = 4,
) -> StageOutcome:
    """Etapa 2 — judgar cada candidata via Market Analyst (LLM estruturado).

    Args:
        candidates: sobreviventes da etapa 1.
        as_of_date: data de referência ``YYYY-MM-DD`` (entra no prompt).
        llm: LangChain chat client. Se None, instancia via :func:`build_llm`.
        llm_config: config caso ``llm`` seja None.
        confidence_min: confidence ≥ este valor é necessário pra PASS.
            Veredict ``NO_PASS`` retornado pelo modelo sempre elimina.
        max_workers: paralelismo entre tickers (decisão D6 — default 4).

    Returns:
        :class:`StageOutcome` com candidatas atualizadas (sempre clonadas — não
        muta as de entrada).
    """
    t0 = time.monotonic()

    if not candidates:
        return StageOutcome(stage_name=STAGE, candidates=[], duration_s=0.0)

    if llm is None:
        llm = build_llm(llm_config)

    structured = bind_structured(llm, MarketVerdict, agent_name="recommend.market")
    if structured is None:
        # Provider sem structured output — abortar a etapa, deixar candidatas passarem
        # como NO_PASS com razão clara, em vez de tentar parsing manual frágil.
        log.error(
            "stage_market: LLM provider does not support structured output; "
            "marking all %d candidates as NO_PASS",
            len(candidates),
        )
        out = [_clone_with_elimination(c, "structured output unsupported by LLM provider", 0.0)
               for c in candidates]
        return StageOutcome(stage_name=STAGE, candidates=out, duration_s=time.monotonic() - t0)

    # ── Paralelizar entre tickers ────────────────────────────────────────────
    out_by_ticker: dict[str, Candidate] = {}

    def _process(cand: Candidate) -> Candidate:
        return _judge_one(cand, as_of_date, structured, confidence_min)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process, c): c for c in candidates}
        for fut in as_completed(futures):
            original = futures[fut]
            try:
                out_by_ticker[original.ticker] = fut.result()
            except Exception as exc:
                log.warning(
                    "stage_market: %s — LLM call raised %s; marking NO_PASS",
                    original.ticker, exc,
                )
                out_by_ticker[original.ticker] = _clone_with_elimination(
                    original, f"LLM call failed: {type(exc).__name__}", 0.0,
                )

    # Preserve input order for reproducibility / nicer logs.
    out = [out_by_ticker[c.ticker] for c in candidates]

    duration = time.monotonic() - t0
    passed = sum(1 for c in out if c.elimination is None)
    log.info(
        "stage_market: %d in, %d kept, %d eliminated, %.1fs",
        len(candidates), passed, len(candidates) - passed, duration,
    )

    return StageOutcome(
        stage_name=STAGE,
        candidates=out,
        duration_s=duration,
        cost_usd=0.0,  # token-level cost tracking is a future improvement
    )


# ── Per-ticker work ──────────────────────────────────────────────────────────


def _judge_one(
    cand: Candidate,
    as_of_date: str,
    structured_llm: Any,
    confidence_min: float,
) -> Candidate:
    prompt = _build_prompt(cand, as_of_date)

    try:
        verdict: MarketVerdict = structured_llm.invoke(prompt)
    except Exception as exc:
        log.warning("stage_market: %s — structured invoke failed: %s", cand.ticker, exc)
        return _clone_with_elimination(
            cand, f"LLM structured invoke failed: {type(exc).__name__}", 0.0,
        )

    if verdict.verdict == "NO_PASS":
        return _clone_with_elimination(cand, verdict.narrative, verdict.confidence)

    if verdict.confidence < confidence_min:
        reason = (
            f"PASS verdict but confidence {verdict.confidence:.2f} below "
            f"threshold {confidence_min}"
        )
        return _clone_with_elimination(cand, reason, verdict.confidence)

    new = _clone(cand)
    new.confidence_history[STAGE] = verdict.confidence
    new.reports[STAGE] = (
        f"Market PASS ({verdict.direction}, confidence={verdict.confidence:.2f}): "
        f"{verdict.narrative}"
    )
    return new


def _build_prompt(cand: Candidate, as_of_date: str) -> list[dict[str, str]]:
    """Compose the user prompt feeding the model the screener's findings."""
    screener_report = cand.reports.get("screener", "no prior technical report")
    user = (
        f"Ticker: {cand.ticker}\n"
        f"As-of date: {as_of_date}\n\n"
        f"Pre-filter technical context (passed earlier deterministic screen):\n"
        f"  {screener_report}\n\n"
        f"Question: based on the technical setup as of {as_of_date}, does "
        f"{cand.ticker} have a tradable setup today? Reply with the structured schema."
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ── Candidate immutability helpers ───────────────────────────────────────────


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
