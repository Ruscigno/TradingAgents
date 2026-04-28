"""Persiste cada run da cascata como JSON em ``eval_results/_recommend_runs/``.

Schema do snapshot está documentado no estudo 01 (seção "Persistência").
Inclui: outcome de cada etapa que rodou, lista de eliminações com razão,
decisões finais ordenadas por ``final_score``, custo total e timing.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tradingagents.recommend.types import Candidate, StageOutcome


log = logging.getLogger(__name__)


@dataclass
class RunSnapshot:
    """Estrutura serializável de um run do recommend."""

    run_id: str
    started_at: str        # ISO-8601 UTC
    as_of_date: str        # YYYY-MM-DD
    skipped_reason: str | None
    universe_source: str   # ex: "mcp:market-data-service"
    universe_size: int
    llm_provider: str
    llm_model: str
    stages: list[dict[str, Any]]
    skipped_stages: list[str]
    decisions: list[dict[str, Any]]
    total_cost_usd: float
    total_duration_s: float


def build_snapshot(
    *,
    as_of_date: str,
    universe_source: str,
    llm_provider: str,
    llm_model: str,
    stage_outcomes: list[StageOutcome],
    final: list[Candidate],
    total_duration_s: float,
    skipped_reason: str | None = None,
    skipped_stages: list[str] | None = None,
    started_at: datetime | None = None,
) -> RunSnapshot:
    """Build a serializable snapshot from in-memory cascade results."""
    started = started_at or datetime.now(tz=timezone.utc)
    universe_size = stage_outcomes[0].candidates if stage_outcomes else []
    return RunSnapshot(
        run_id=uuid.uuid4().hex[:12],
        started_at=started.isoformat(),
        as_of_date=as_of_date,
        skipped_reason=skipped_reason,
        universe_source=universe_source,
        universe_size=len(universe_size),
        llm_provider=llm_provider,
        llm_model=llm_model,
        stages=[_stage_to_dict(o) for o in stage_outcomes],
        skipped_stages=list(skipped_stages or []),
        decisions=[_decision_to_dict(c) for c in final],
        total_cost_usd=sum(o.cost_usd for o in stage_outcomes),
        total_duration_s=total_duration_s,
    )


def write_snapshot(
    snapshot: RunSnapshot,
    base_dir: str | Path | None = None,
) -> Path:
    """Write snapshot to ``{base_dir}/{date}_{run_id}.json``.

    Default base_dir is ``$TRADINGAGENTS_RESULTS_DIR/_recommend_runs`` or
    ``./eval_results/_recommend_runs``.
    """
    if base_dir is None:
        results_dir = os.environ.get("TRADINGAGENTS_RESULTS_DIR", "./eval_results")
        base_dir = Path(results_dir) / "_recommend_runs"
    base_path = Path(base_dir).expanduser().resolve()
    base_path.mkdir(parents=True, exist_ok=True)

    fname = f"{snapshot.as_of_date}_{snapshot.run_id}.json"
    out = base_path / fname
    with out.open("w", encoding="utf-8") as f:
        json.dump(_asdict(snapshot), f, indent=2, default=str)

    log.info("recommend run persisted to %s", out)
    return out


# ── Serialization helpers ────────────────────────────────────────────────────


def _stage_to_dict(outcome: StageOutcome) -> dict[str, Any]:
    return {
        "name": outcome.stage_name,
        "before": len(outcome.candidates),
        "passed": len(outcome.passed),
        "eliminated": [
            {
                "ticker": c.ticker,
                "reason": (c.elimination.reason if c.elimination else ""),
                "confidence": (c.elimination.confidence if c.elimination else 0.0),
            }
            for c in outcome.eliminated
        ],
        "duration_s": round(outcome.duration_s, 3),
        "cost_usd": round(outcome.cost_usd, 4),
    }


def _decision_to_dict(c: Candidate) -> dict[str, Any]:
    """Convert final passed Candidate to a decision-shaped dict.

    Pulls structured fields stashed by stage_decision (action, stop, target,
    position_size_pct) from ``reports``. If they are missing (i.e. the run
    stopped before stage 6), still returns the base candidate info.
    """
    rep = c.reports
    return {
        "ticker": c.ticker,
        "final_score": c.final_score,
        "action": rep.get("_decision_action"),
        "stop_loss": _safe_float(rep.get("_decision_stop_loss")),
        "take_profit": _safe_float(rep.get("_decision_take_profit")),
        "position_size_pct": _safe_float(rep.get("_decision_position_size_pct")),
        "rationale": rep.get("decision", "")[:2000],
        "confidence_history": dict(c.confidence_history),
    }


def _safe_float(s: str | None) -> float | None:
    if s is None:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _asdict(snapshot: RunSnapshot) -> dict[str, Any]:
    """Plain dict for JSON dump (avoid `dataclasses.asdict` to keep our shape)."""
    return {
        "run_id": snapshot.run_id,
        "started_at": snapshot.started_at,
        "as_of_date": snapshot.as_of_date,
        "skipped_reason": snapshot.skipped_reason,
        "universe_source": snapshot.universe_source,
        "universe_size": snapshot.universe_size,
        "llm_provider": snapshot.llm_provider,
        "llm_model": snapshot.llm_model,
        "stages": snapshot.stages,
        "skipped_stages": snapshot.skipped_stages,
        "decisions": snapshot.decisions,
        "total_cost_usd": round(snapshot.total_cost_usd, 4),
        "total_duration_s": round(snapshot.total_duration_s, 3),
    }
