"""Etapa 4 — Bull × Bear researcher debate + judge.

Para cada candidata vinda da Etapa 3:

1. Bull e Bear LLM são chamados **em paralelo** com todos os reports anteriores
   no contexto. Cada um produz uma tese + argumentos (1 round de filtro).
2. O Judge recebe os dois lados e decide: PASS/NO_PASS, direção, confidence.
3. Threshold default: confidence ≥ 0.7 para PASS.

Custo estimado: ~$0.014/ticker (3 LLM calls × ~$0.005, prompts ~6k input + 2k output).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from tradingagents.agents.utils.structured import bind_structured
from tradingagents.recommend.llm import LLMConfig, build_llm
from tradingagents.recommend.schemas import BearCase, BullCase, DebateVerdict
from tradingagents.recommend.types import Candidate, Elimination, StageOutcome


log = logging.getLogger(__name__)
STAGE = "debate"

DEFAULT_CONFIDENCE_MIN = 0.7


_BULL_SYSTEM = """You are a bullish equity researcher. Argue for opening a long \
position in the ticker, given all reports already accumulated by the cascade. \
Reply with the structured BullCase schema.

Be specific — anchor your arguments to the actual data in the reports. If the \
data does not support a long thesis, say so plainly with a thin thesis (e.g. \
"setup is mediocre but not negative") rather than fabricating arguments."""


_BEAR_SYSTEM = """You are a bearish equity researcher. Argue for skipping or \
shorting the ticker, given all reports already accumulated by the cascade. \
Reply with the structured BearCase schema.

Be specific — anchor your arguments to the actual data in the reports. If you \
see no real bear case, present the strongest skeptical reading you can, but \
keep the thesis honest."""


_JUDGE_SYSTEM = """You are an experienced research manager judging a single \
round of debate between Bull and Bear researchers. Decide whether the ticker \
advances in the cascade.

Reply with the structured DebateVerdict schema. Use the full confidence range \
thoughtfully (around 0.7 is the threshold to advance). Choose direction based \
on which side won the debate ('none' if the picture is unclear or NO_PASS)."""


def run(
    candidates: list[Candidate],
    *,
    as_of_date: str,
    llm: Any | None = None,
    llm_config: LLMConfig | None = None,
    confidence_min: float = DEFAULT_CONFIDENCE_MIN,
    max_candidate_workers: int = 4,
) -> StageOutcome:
    """Etapa 4 — Bull/Bear debate + judge."""
    t0 = time.monotonic()

    if not candidates:
        return StageOutcome(stage_name=STAGE, candidates=[], duration_s=0.0)

    if llm is None:
        llm = build_llm(llm_config)

    bull_client = bind_structured(llm, BullCase, "recommend.bull")
    bear_client = bind_structured(llm, BearCase, "recommend.bear")
    judge_client = bind_structured(llm, DebateVerdict, "recommend.judge")

    if any(c is None for c in (bull_client, bear_client, judge_client)):
        log.error("stage_debate: structured output unsupported by LLM provider")
        out = [
            _clone_with_elimination(c, "structured output unsupported by LLM provider", 0.0)
            for c in candidates
        ]
        return StageOutcome(stage_name=STAGE, candidates=out, duration_s=time.monotonic() - t0)

    def _process(cand: Candidate) -> Candidate:
        return _judge_one(
            cand,
            as_of_date=as_of_date,
            bull_client=bull_client,
            bear_client=bear_client,
            judge_client=judge_client,
            confidence_min=confidence_min,
        )

    out_by_ticker: dict[str, Candidate] = {}
    with ThreadPoolExecutor(max_workers=max_candidate_workers) as pool:
        futures = {pool.submit(_process, c): c for c in candidates}
        for fut in as_completed(futures):
            original = futures[fut]
            try:
                out_by_ticker[original.ticker] = fut.result()
            except Exception as exc:
                log.warning("stage_debate: %s — %s", original.ticker, exc)
                out_by_ticker[original.ticker] = _clone_with_elimination(
                    original, f"debate pipeline failed: {type(exc).__name__}", 0.0,
                )

    out = [out_by_ticker[c.ticker] for c in candidates]
    duration = time.monotonic() - t0
    passed = sum(1 for c in out if c.elimination is None)
    log.info(
        "stage_debate: %d in, %d kept, %d eliminated, %.1fs",
        len(candidates), passed, len(candidates) - passed, duration,
    )
    return StageOutcome(stage_name=STAGE, candidates=out, duration_s=duration, cost_usd=0.0)


# ── Per-candidate work ──────────────────────────────────────────────────────


def _judge_one(
    cand: Candidate,
    *,
    as_of_date: str,
    bull_client: Any,
    bear_client: Any,
    judge_client: Any,
    confidence_min: float,
) -> Candidate:
    context = _format_prior_context(cand, as_of_date)

    # Run bull + bear in parallel.
    with ThreadPoolExecutor(max_workers=2) as pool:
        bull_future = pool.submit(
            bull_client.invoke,
            [
                {"role": "system", "content": _BULL_SYSTEM},
                {"role": "user", "content": context + "\n\nProduce the structured BullCase."},
            ],
        )
        bear_future = pool.submit(
            bear_client.invoke,
            [
                {"role": "system", "content": _BEAR_SYSTEM},
                {"role": "user", "content": context + "\n\nProduce the structured BearCase."},
            ],
        )
        try:
            bull: BullCase = bull_future.result()
            bear: BearCase = bear_future.result()
        except Exception as exc:
            return _clone_with_elimination(cand, f"bull/bear call failed: {exc}", 0.0)

    # Judge step (sequential, depends on both).
    judge_user = (
        f"{context}\n\n"
        f"---\nBull case:\nThesis: {bull.thesis}\n"
        + "\n".join(f"  - {a}" for a in bull.key_arguments)
        + f"\n\n---\nBear case:\nThesis: {bear.thesis}\n"
        + "\n".join(f"  - {a}" for a in bear.key_arguments)
        + "\n\n---\nDecide. Reply with the structured DebateVerdict."
    )
    try:
        verdict: DebateVerdict = judge_client.invoke(
            [
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": judge_user},
            ],
        )
    except Exception as exc:
        return _clone_with_elimination(cand, f"judge call failed: {exc}", 0.0)

    if verdict.verdict == "NO_PASS":
        new = _clone(cand)
        new.elimination = Elimination(stage=STAGE, reason=verdict.rationale, confidence=verdict.confidence)
        return new
    if verdict.confidence < confidence_min:
        new = _clone(cand)
        new.elimination = Elimination(
            stage=STAGE,
            reason=f"PASS but confidence {verdict.confidence:.2f} below threshold {confidence_min}",
            confidence=verdict.confidence,
        )
        return new

    new = _clone(cand)
    new.confidence_history[STAGE] = verdict.confidence
    new.reports[STAGE] = (
        f"Debate verdict PASS ({verdict.direction}, confidence={verdict.confidence:.2f}):\n"
        f"  Bull: {bull.thesis}\n"
        f"  Bear: {bear.thesis}\n"
        f"  Judge: {verdict.rationale}"
    )
    return new


def _format_prior_context(cand: Candidate, as_of_date: str) -> str:
    parts = [f"Ticker: {cand.ticker}", f"As-of date: {as_of_date}", "", "Reports from earlier cascade stages:"]
    for stage, report in cand.reports.items():
        parts.append(f"\n[{stage}]")
        parts.append(report)
    parts.append("\nConfidence history so far: " +
                 ", ".join(f"{k}={v:.2f}" for k, v in cand.confidence_history.items()))
    return "\n".join(parts)


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
