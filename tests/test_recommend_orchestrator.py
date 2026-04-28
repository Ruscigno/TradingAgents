"""Tests para o orquestrador dry-run da cascata."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.recommend.orchestrator import DryRunResult, run_dry_run
from tradingagents.recommend.types import Candidate, Elimination, StageOutcome


# Reference dates from XNYS calendar — Mon 2026-01-05 trading; Sat 2026-01-03 not.
TRADING_DAY = datetime(2026, 1, 5, 16, 0, tzinfo=timezone.utc)
WEEKEND = datetime(2026, 1, 3, 16, 0, tzinfo=timezone.utc)
NEW_YEARS = datetime(2026, 1, 1, 16, 0, tzinfo=timezone.utc)


def _universe_outcome(candidates: list[Candidate]) -> StageOutcome:
    return StageOutcome(stage_name="universe", candidates=candidates, duration_s=1.0)


def _screener_outcome(candidates: list[Candidate]) -> StageOutcome:
    return StageOutcome(stage_name="screener", candidates=candidates, duration_s=2.0)


class TestRunDryRun:
    def test_skips_on_weekend(self):
        result = run_dry_run(
            mds_client=MagicMock(),
            as_of_date="2026-01-03",
            now=WEEKEND,
        )
        assert result.skipped_reason is not None
        assert "trading day" in result.skipped_reason.lower()
        assert result.stage_outcomes == []
        assert result.final == []

    def test_skips_on_holiday(self):
        result = run_dry_run(
            mds_client=MagicMock(),
            as_of_date="2026-01-01",
            now=NEW_YEARS,
        )
        assert result.skipped_reason is not None

    def test_runs_when_skip_calendar_check(self):
        # Even on a weekend, if user passes --skip-calendar-check the gate is OFF.
        with patch(
            "tradingagents.recommend.orchestrator.run_stage_universe",
            return_value=_universe_outcome([]),
        ) as mock_uni:
            result = run_dry_run(
                mds_client=MagicMock(),
                as_of_date="2026-01-03",
                now=WEEKEND,
                skip_calendar_check=True,
            )
        assert result.skipped_reason is None
        mock_uni.assert_called_once()

    def test_short_circuits_on_empty_universe(self):
        with patch(
            "tradingagents.recommend.orchestrator.run_stage_universe",
            return_value=_universe_outcome([]),
        ), patch(
            "tradingagents.recommend.orchestrator.run_stage_screener"
        ) as mock_screener:
            result = run_dry_run(
                mds_client=MagicMock(),
                as_of_date="2026-01-05",
                now=TRADING_DAY,
            )
        assert result.skipped_reason is None
        assert len(result.stage_outcomes) == 1
        assert result.final == []
        mock_screener.assert_not_called()

    def test_full_flow_orders_by_confidence_desc(self):
        cand_lo = Candidate(
            ticker="LOW", confidence_history={"screener": 0.3}
        )
        cand_hi = Candidate(
            ticker="HIGH", confidence_history={"screener": 0.9}
        )
        cand_mid = Candidate(
            ticker="MID", confidence_history={"screener": 0.6}
        )
        cand_cut = Candidate(
            ticker="CUT",
            elimination=Elimination(stage="screener", reason="RSI=85", confidence=0.0),
        )

        with patch(
            "tradingagents.recommend.orchestrator.run_stage_universe",
            return_value=_universe_outcome([cand_lo, cand_hi, cand_mid, cand_cut]),
        ), patch(
            "tradingagents.recommend.orchestrator.run_stage_screener",
            return_value=_screener_outcome([cand_lo, cand_hi, cand_mid, cand_cut]),
        ):
            result = run_dry_run(
                mds_client=MagicMock(),
                as_of_date="2026-01-05",
                now=TRADING_DAY,
            )

        assert [c.ticker for c in result.final] == ["HIGH", "MID", "LOW"]
        assert len(result.stage_outcomes) == 2
        assert result.stage_outcomes[0].stage_name == "universe"
        assert result.stage_outcomes[1].stage_name == "screener"

    def test_total_duration_recorded(self):
        with patch(
            "tradingagents.recommend.orchestrator.run_stage_universe",
            return_value=_universe_outcome([]),
        ):
            result = run_dry_run(
                mds_client=MagicMock(),
                as_of_date="2026-01-05",
                now=TRADING_DAY,
            )
        assert result.total_duration_s >= 0.0
