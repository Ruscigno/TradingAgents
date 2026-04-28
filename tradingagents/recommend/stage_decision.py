"""Etapa 6 — Trader + Risk debate + Portfolio Manager: decisão final.

A última etapa da cascata. Para cada sobrevivente da Etapa 5:

1. **Risk debate** (3 perspectivas em paralelo): Aggressive / Neutral / Conservative
   produzem cada um um ponto-de-vista textual sobre o investment_plan.
2. **Portfolio Manager** consolida tudo numa :class:`TradeDecision` estruturada
   (action BUY/SELL/HOLD, stop_loss, take_profit, rationale).

Position sizing é fixo (decisão D4) e vem da config; o LLM **não** decide tamanho.

Custo estimado: ~$0.031/ticker (3 risk + 1 portfolio = 4 calls, contexto grande).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from tradingagents.agents.utils.structured import bind_structured
from tradingagents.recommend.llm import LLMConfig, build_llm
from tradingagents.recommend.schemas import TradeDecision
from tradingagents.recommend.types import Candidate, Elimination, StageOutcome


log = logging.getLogger(__name__)
STAGE = "decision"

DEFAULT_CONFIDENCE_MIN = 0.7
DEFAULT_POSITION_SIZE_PCT = 2.0
DEFAULT_MAX_CONCURRENT_TRADES = 5


@dataclass(frozen=True)
class PositionSizing:
    """Position sizing estática vinda da config (decisão D4)."""

    size_pct: float = DEFAULT_POSITION_SIZE_PCT
    max_concurrent_trades: int = DEFAULT_MAX_CONCURRENT_TRADES


_AGGRESSIVE_SYSTEM = """You are an aggressive risk perspective. Argue for taking \
the trade with the strongest position the plan allows. Prioritize upside; tolerate \
volatility. Reply in 2-3 sentences. Cite specific numbers from the plan when possible."""

_NEUTRAL_SYSTEM = """You are a balanced risk perspective. Weigh upside and downside \
even-handedly. Suggest the most defensible execution given current market regime. \
Reply in 2-3 sentences."""

_CONSERVATIVE_SYSTEM = """You are a conservative risk perspective. Argue for the \
trade only if downside is well-defined and small. Prefer no trade over a fragile one. \
Reply in 2-3 sentences."""


_PORTFOLIO_SYSTEM = """You are a Portfolio Manager. Make the final go/no-go call \
on a candidate trade given:
  (a) the consolidated investment plan,
  (b) three risk-perspective takes (aggressive / neutral / conservative).

Reply with the structured TradeDecision schema. Rules:
- ``action`` must be BUY (long entry), SELL (short entry), or HOLD (no trade today).
- HOLD must come with verdict=NO_PASS — HOLDs are not actionable trades.
- Set ``stop_loss`` and ``take_profit`` to concrete dollar prices when action != HOLD.
  Use 0.0 for HOLD.
- Position sizing is FIXED elsewhere — do **not** include it in your decision.
- Use the full confidence range; the cascade threshold for advancing this stage is around 0.7."""


def run(
    candidates: list[Candidate],
    *,
    as_of_date: str,
    llm: Any | None = None,
    llm_config: LLMConfig | None = None,
    position: PositionSizing | None = None,
    confidence_min: float = DEFAULT_CONFIDENCE_MIN,
    max_candidate_workers: int = 4,
) -> StageOutcome:
    """Etapa 6 — risk debate + portfolio manager final decision."""
    t0 = time.monotonic()

    if not candidates:
        return StageOutcome(stage_name=STAGE, candidates=[], duration_s=0.0)

    pos = position or PositionSizing()

    if llm is None:
        llm = build_llm(llm_config)
    pm_client = bind_structured(llm, TradeDecision, "recommend.portfolio")
    if pm_client is None:
        out = [
            _clone_with_elim(c, "structured output unsupported by LLM provider", 0.0, pos)
            for c in candidates
        ]
        return StageOutcome(stage_name=STAGE, candidates=out, duration_s=time.monotonic() - t0)

    def _process(cand: Candidate) -> Candidate:
        return _judge_one(
            cand,
            as_of_date=as_of_date,
            llm=llm,
            pm_client=pm_client,
            confidence_min=confidence_min,
            position=pos,
        )

    out_by_ticker: dict[str, Candidate] = {}
    with ThreadPoolExecutor(max_workers=max_candidate_workers) as pool:
        futures = {pool.submit(_process, c): c for c in candidates}
        for fut in as_completed(futures):
            original = futures[fut]
            try:
                out_by_ticker[original.ticker] = fut.result()
            except Exception as exc:
                log.warning("stage_decision: %s — %s", original.ticker, exc)
                out_by_ticker[original.ticker] = _clone_with_elim(
                    original, f"decision pipeline failed: {type(exc).__name__}", 0.0, pos,
                )

    out = [out_by_ticker[c.ticker] for c in candidates]
    duration = time.monotonic() - t0
    passed = sum(1 for c in out if c.elimination is None)
    log.info(
        "stage_decision: %d in, %d kept, %d eliminated, %.1fs",
        len(candidates), passed, len(candidates) - passed, duration,
    )
    return StageOutcome(stage_name=STAGE, candidates=out, duration_s=duration, cost_usd=0.0)


# ── Per-candidate work ──────────────────────────────────────────────────────


def _judge_one(
    cand: Candidate,
    *,
    as_of_date: str,
    llm: Any,
    pm_client: Any,
    confidence_min: float,
    position: PositionSizing,
) -> Candidate:
    context = _format_full_context(cand, as_of_date)

    # 1. Three risk perspectives in parallel (free-text, NOT structured).
    def _ask(system: str) -> str:
        try:
            response = llm.invoke([
                {"role": "system", "content": system},
                {"role": "user", "content": context},
            ])
            return getattr(response, "content", str(response))
        except Exception as exc:
            return f"(perspective LLM call failed: {exc})"

    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {
            "aggressive": pool.submit(_ask, _AGGRESSIVE_SYSTEM),
            "neutral": pool.submit(_ask, _NEUTRAL_SYSTEM),
            "conservative": pool.submit(_ask, _CONSERVATIVE_SYSTEM),
        }
        risk_takes = {role: f.result() for role, f in futs.items()}

    # 2. Portfolio manager final call.
    pm_user = (
        f"{context}\n\n"
        f"---\nRisk debate:\n"
        f"  [Aggressive] {risk_takes['aggressive']}\n"
        f"  [Neutral]    {risk_takes['neutral']}\n"
        f"  [Conservative] {risk_takes['conservative']}\n\n"
        f"---\nDecide. Reply with the structured TradeDecision."
    )
    try:
        decision: TradeDecision = pm_client.invoke([
            {"role": "system", "content": _PORTFOLIO_SYSTEM},
            {"role": "user", "content": pm_user},
        ])
    except Exception as exc:
        return _clone_with_elim(cand, f"portfolio manager call failed: {exc}", 0.0, position)

    if decision.action == "HOLD" or decision.verdict == "NO_PASS":
        new = _clone(cand)
        new.elimination = Elimination(
            stage=STAGE,
            reason=f"portfolio manager: {decision.action} — {decision.rationale[:300]}",
            confidence=decision.confidence,
        )
        return new
    if decision.confidence < confidence_min:
        new = _clone(cand)
        new.elimination = Elimination(
            stage=STAGE,
            reason=f"PASS but confidence {decision.confidence:.2f} < threshold {confidence_min}",
            confidence=decision.confidence,
        )
        return new

    new = _clone(cand)
    new.confidence_history[STAGE] = decision.confidence
    new.reports[STAGE] = (
        f"FINAL DECISION: {decision.action} (confidence={decision.confidence:.2f})\n"
        f"  stop_loss={decision.stop_loss}  take_profit={decision.take_profit}\n"
        f"  position_size={position.size_pct}% (fixed via config)\n"
        f"  rationale: {decision.rationale}"
    )
    # Stash the parsed decision attributes on reports so persistence layer can pull them.
    new.reports["_decision_action"] = decision.action
    new.reports["_decision_stop_loss"] = str(decision.stop_loss)
    new.reports["_decision_take_profit"] = str(decision.take_profit)
    new.reports["_decision_position_size_pct"] = str(position.size_pct)
    return new


def _format_full_context(cand: Candidate, as_of_date: str) -> str:
    parts = [
        f"Ticker: {cand.ticker}",
        f"As-of date: {as_of_date}",
        "",
        "All cascade reports so far:",
    ]
    for stage, report in cand.reports.items():
        if stage.startswith("_"):
            continue
        parts.append(f"\n[{stage}]")
        parts.append(report)
    return "\n".join(parts)


def _clone(cand: Candidate) -> Candidate:
    return Candidate(
        ticker=cand.ticker,
        confidence_history=dict(cand.confidence_history),
        reports=dict(cand.reports),
        elimination=cand.elimination,
        final_score=cand.final_score,
    )


def _clone_with_elim(
    cand: Candidate,
    reason: str,
    confidence: float,
    _position: PositionSizing,
) -> Candidate:
    new = _clone(cand)
    new.elimination = Elimination(stage=STAGE, reason=reason, confidence=confidence)
    return new
