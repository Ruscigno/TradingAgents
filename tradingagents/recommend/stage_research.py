"""Etapa 5 — Research Manager consolida tudo num investment plan.

Para cada sobrevivente da Etapa 4, faz **uma** chamada LLM com todo o contexto
das etapas anteriores e gera um plano de investimento estruturado
(:class:`ResearchPlan`). O Research Manager pode ainda emitir NO_PASS se
ao revisar a totalidade dos sinais ele identificar contradição que as etapas
isoladas não viram.

Custo estimado: ~$0.018/ticker (1 chamada com contexto grande, ~10k input + 2k output).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from tradingagents.agents.utils.structured import bind_structured
from tradingagents.recommend.llm import LLMConfig, build_llm
from tradingagents.recommend.schemas import ResearchPlan
from tradingagents.recommend.types import Candidate, Elimination, StageOutcome


log = logging.getLogger(__name__)
STAGE = "research"

DEFAULT_CONFIDENCE_MIN = 0.7


_SYSTEM = """You are a senior research manager consolidating multiple analyst \
reports into a single coherent investment plan. The cascade has already \
filtered for technical setup, multi-analyst signal, and a passed bull/bear \
debate. Your role is the integrator: synthesise everything, spot contradictions \
the earlier stages might have missed, and emit a structured investment plan.

Reply with the ResearchPlan schema. Use NO_PASS only if your synthesis reveals \
material conflicts between earlier signals (e.g. technical bullish but \
fundamentals deteriorating, or news catalysts that contradict the debate \
verdict). Use PASS otherwise, with a thoughtful confidence and a concrete \
investment_plan in markdown."""


def run(
    candidates: list[Candidate],
    *,
    as_of_date: str,
    llm: Any | None = None,
    llm_config: LLMConfig | None = None,
    confidence_min: float = DEFAULT_CONFIDENCE_MIN,
    max_candidate_workers: int = 4,
) -> StageOutcome:
    """Etapa 5 — Research Manager."""
    t0 = time.monotonic()

    if not candidates:
        return StageOutcome(stage_name=STAGE, candidates=[], duration_s=0.0)

    if llm is None:
        llm = build_llm(llm_config)
    structured = bind_structured(llm, ResearchPlan, "recommend.research")
    if structured is None:
        out = [
            _clone_with_elim(c, "structured output unsupported by LLM provider", 0.0)
            for c in candidates
        ]
        return StageOutcome(stage_name=STAGE, candidates=out, duration_s=time.monotonic() - t0)

    def _process(cand: Candidate) -> Candidate:
        return _judge_one(cand, as_of_date, structured, confidence_min)

    out_by_ticker: dict[str, Candidate] = {}
    with ThreadPoolExecutor(max_workers=max_candidate_workers) as pool:
        futures = {pool.submit(_process, c): c for c in candidates}
        for fut in as_completed(futures):
            original = futures[fut]
            try:
                out_by_ticker[original.ticker] = fut.result()
            except Exception as exc:
                log.warning("stage_research: %s — %s", original.ticker, exc)
                out_by_ticker[original.ticker] = _clone_with_elim(
                    original, f"research pipeline failed: {type(exc).__name__}", 0.0,
                )

    out = [out_by_ticker[c.ticker] for c in candidates]
    duration = time.monotonic() - t0
    passed = sum(1 for c in out if c.elimination is None)
    log.info(
        "stage_research: %d in, %d kept, %d eliminated, %.1fs",
        len(candidates), passed, len(candidates) - passed, duration,
    )
    return StageOutcome(stage_name=STAGE, candidates=out, duration_s=duration, cost_usd=0.0)


def _judge_one(
    cand: Candidate,
    as_of_date: str,
    structured: Any,
    confidence_min: float,
) -> Candidate:
    context = _format_full_context(cand, as_of_date)
    try:
        plan: ResearchPlan = structured.invoke([
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": context + "\n\nProduce the structured ResearchPlan."},
        ])
    except Exception as exc:
        return _clone_with_elim(cand, f"research LLM call failed: {exc}", 0.0)

    if plan.verdict == "NO_PASS":
        new = _clone(cand)
        new.elimination = Elimination(
            stage=STAGE,
            reason=f"research manager rejected: {plan.investment_plan[:300]}",
            confidence=plan.confidence,
        )
        return new
    if plan.confidence < confidence_min:
        new = _clone(cand)
        new.elimination = Elimination(
            stage=STAGE,
            reason=f"PASS but confidence {plan.confidence:.2f} < threshold {confidence_min}",
            confidence=plan.confidence,
        )
        return new

    new = _clone(cand)
    new.confidence_history[STAGE] = plan.confidence
    new.reports[STAGE] = (
        f"Research plan ({plan.direction}, confidence={plan.confidence:.2f}):\n"
        f"{plan.investment_plan}\n\n"
        f"Key risks:\n" + "\n".join(f"  - {r}" for r in plan.key_risks)
    )
    return new


def _format_full_context(cand: Candidate, as_of_date: str) -> str:
    parts = [
        f"Ticker: {cand.ticker}",
        f"As-of date: {as_of_date}",
        "",
        "All cascade reports so far:",
    ]
    for stage, report in cand.reports.items():
        parts.append(f"\n[{stage}]")
        parts.append(report)
    parts.append(
        "\nConfidence history so far: "
        + ", ".join(f"{k}={v:.2f}" for k, v in cand.confidence_history.items())
    )
    return "\n".join(parts)


def _clone(cand: Candidate) -> Candidate:
    return Candidate(
        ticker=cand.ticker,
        confidence_history=dict(cand.confidence_history),
        reports=dict(cand.reports),
        elimination=cand.elimination,
        final_score=cand.final_score,
    )


def _clone_with_elim(cand: Candidate, reason: str, confidence: float) -> Candidate:
    new = _clone(cand)
    new.elimination = Elimination(stage=STAGE, reason=reason, confidence=confidence)
    return new
