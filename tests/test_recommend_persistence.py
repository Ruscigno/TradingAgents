"""Tests for persistence (run snapshot to JSON)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from tradingagents.recommend.persistence import build_snapshot, write_snapshot
from tradingagents.recommend.types import Candidate, Elimination, StageOutcome


def _outcome(stage: str, candidates: list[Candidate], duration: float = 1.0) -> StageOutcome:
    return StageOutcome(stage_name=stage, candidates=candidates, duration_s=duration)


def _passed(ticker: str, score: float = 0.85, with_decision: bool = False) -> Candidate:
    reports = {"market": "ok"}
    if with_decision:
        reports.update({
            "decision": "FINAL DECISION: BUY",
            "_decision_action": "BUY",
            "_decision_stop_loss": "178.5",
            "_decision_take_profit": "195.0",
            "_decision_position_size_pct": "2.0",
        })
    return Candidate(
        ticker=ticker,
        confidence_history={"screener": 0.6, "market": 0.8},
        reports=reports,
        final_score=score,
    )


def _eliminated(ticker: str, stage: str, reason: str) -> Candidate:
    return Candidate(
        ticker=ticker,
        elimination=Elimination(stage=stage, reason=reason, confidence=0.5),
    )


class TestBuildSnapshot:
    def test_basic_shape(self):
        outcome = _outcome("universe", [_passed("AAPL"), _passed("MSFT")])
        snap = build_snapshot(
            as_of_date="2026-04-24",
            universe_source="mcp:market-data-service",
            llm_provider="openrouter",
            llm_model="moonshotai/kimi-k2.6",
            stage_outcomes=[outcome],
            final=[_passed("AAPL", with_decision=True)],
            total_duration_s=42.5,
            started_at=datetime(2026, 4, 27, 18, 0, tzinfo=timezone.utc),
        )
        assert snap.as_of_date == "2026-04-24"
        assert snap.universe_size == 2
        assert snap.llm_model == "moonshotai/kimi-k2.6"
        assert len(snap.stages) == 1
        assert snap.stages[0]["name"] == "universe"
        assert snap.stages[0]["passed"] == 2
        assert len(snap.decisions) == 1
        assert snap.decisions[0]["ticker"] == "AAPL"
        assert snap.decisions[0]["action"] == "BUY"
        assert snap.decisions[0]["stop_loss"] == 178.5
        assert snap.decisions[0]["position_size_pct"] == 2.0

    def test_skipped_run(self):
        snap = build_snapshot(
            as_of_date="2026-01-03",   # Saturday
            universe_source="mcp:market-data-service",
            llm_provider="openrouter",
            llm_model="moonshotai/kimi-k2.6",
            stage_outcomes=[],
            final=[],
            total_duration_s=0.1,
            skipped_reason="not a trading day",
        )
        assert snap.skipped_reason == "not a trading day"
        assert snap.universe_size == 0
        assert snap.decisions == []

    def test_eliminations_captured(self):
        outcome = _outcome("screener", [
            _passed("AAPL"),
            _eliminated("BAD", stage="screener", reason="RSI=82 (>70)"),
        ])
        snap = build_snapshot(
            as_of_date="2026-04-24",
            universe_source="mcp:test",
            llm_provider="openrouter",
            llm_model="kimi",
            stage_outcomes=[outcome],
            final=[_passed("AAPL")],
            total_duration_s=1.0,
        )
        elim = snap.stages[0]["eliminated"]
        assert len(elim) == 1
        assert elim[0]["ticker"] == "BAD"
        assert "RSI" in elim[0]["reason"]


class TestWriteSnapshot:
    def test_writes_to_disk_and_returns_path(self, tmp_path: Path):
        snap = build_snapshot(
            as_of_date="2026-04-24",
            universe_source="mcp:test",
            llm_provider="openrouter",
            llm_model="kimi",
            stage_outcomes=[_outcome("universe", [_passed("AAPL")])],
            final=[_passed("AAPL")],
            total_duration_s=1.0,
        )
        out = write_snapshot(snap, base_dir=tmp_path)
        assert out.exists()
        assert out.parent == tmp_path.resolve()

        with out.open() as f:
            data = json.load(f)
        assert data["as_of_date"] == "2026-04-24"
        assert data["llm_model"] == "kimi"
        assert data["universe_size"] == 1
        assert data["run_id"] in out.name

    def test_uses_env_var_when_no_base_dir(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("TRADINGAGENTS_RESULTS_DIR", str(tmp_path))
        snap = build_snapshot(
            as_of_date="2026-04-24",
            universe_source="x", llm_provider="x", llm_model="x",
            stage_outcomes=[], final=[], total_duration_s=0.0,
        )
        out = write_snapshot(snap)
        assert (tmp_path / "_recommend_runs").exists()
        assert out.parent == (tmp_path / "_recommend_runs").resolve()
