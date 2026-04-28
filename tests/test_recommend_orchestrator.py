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


# ── run_full ─────────────────────────────────────────────────────────────────


from tradingagents.recommend.orchestrator import _final_score, run_full


class TestFinalScore:
    def test_all_stages_present(self):
        c = Candidate(
            ticker="X",
            confidence_history={
                "screener": 0.5, "market": 0.6, "analysts": 0.7,
                "debate":   0.8, "research": 0.9, "decision": 0.85,
            },
        )
        # 0.05*0.5 + 0.10*0.6 + 0.15*0.7 + 0.20*0.8 + 0.25*0.9 + 0.25*0.85
        # = 0.025 + 0.06 + 0.105 + 0.16 + 0.225 + 0.2125 = 0.7875
        assert _final_score(c) == pytest.approx(0.7875, abs=1e-3)

    def test_renormalizes_when_stages_missing(self):
        # Only screener + market (cascade stopped early). Weights renormalize.
        c = Candidate(
            ticker="X",
            confidence_history={"screener": 1.0, "market": 0.0},
        )
        # weights: screener=0.05, market=0.10. total=0.15.
        # final = 1.0 * 0.05/0.15 + 0.0 * 0.10/0.15 = 0.3333
        assert _final_score(c) == pytest.approx(1.0/3, abs=1e-3)

    def test_empty_history_zero(self):
        assert _final_score(Candidate(ticker="X")) == 0.0


class TestRunFull:
    def test_skips_on_weekend(self):
        result = run_full(
            mds_client=MagicMock(),
            as_of_date="2026-01-03",
            now=WEEKEND,
        )
        assert result.skipped_reason is not None
        assert result.stage_outcomes == []
        assert result.final == []

    def test_short_circuits_when_universe_empty(self):
        with patch(
            "tradingagents.recommend.orchestrator.run_stage_universe",
            return_value=_universe_outcome([]),
        ):
            result = run_full(
                mds_client=MagicMock(),
                as_of_date="2026-01-05",
                now=TRADING_DAY,
            )
        assert len(result.stage_outcomes) == 1
        assert "screener" in result.skipped_stages
        assert "decision" in result.skipped_stages
        assert result.final == []

    def test_skip_on_zero_after_screener(self):
        # Universe returns candidates, screener eliminates all.
        cand_in = Candidate(ticker="AAPL")
        cand_out = Candidate(
            ticker="AAPL",
            elimination=Elimination(stage="screener", reason="RSI=80", confidence=0.0),
        )
        with patch(
            "tradingagents.recommend.orchestrator.run_stage_universe",
            return_value=_universe_outcome([cand_in]),
        ), patch(
            "tradingagents.recommend.orchestrator.run_stage_screener",
            return_value=StageOutcome(stage_name="screener", candidates=[cand_out]),
        ), patch(
            "tradingagents.recommend.orchestrator.build_llm",
        ) as mock_build_llm:
            result = run_full(
                mds_client=MagicMock(),
                as_of_date="2026-01-05",
                now=TRADING_DAY,
            )
        assert len(result.stage_outcomes) == 2  # universe + screener only
        assert set(result.skipped_stages) == {
            "market", "analysts", "debate", "research", "decision",
        }
        assert result.final == []
        mock_build_llm.assert_not_called()  # no LLM stages reached

    def test_full_flow_assigns_final_score_and_orders(self):
        # Candidate that passes everything has 6 confidences in history.
        winner = Candidate(
            ticker="WINNER",
            confidence_history={
                "screener": 0.7, "market": 0.8, "analysts": 0.85,
                "debate":   0.85, "research": 0.9, "decision": 0.9,
            },
            reports={"_decision_action": "BUY"},
        )
        runner = Candidate(
            ticker="RUNNER",
            confidence_history={
                "screener": 0.6, "market": 0.7, "analysts": 0.75,
                "debate":   0.8, "research": 0.8, "decision": 0.8,
            },
        )
        with patch(
            "tradingagents.recommend.orchestrator.run_stage_universe",
            return_value=_universe_outcome([winner, runner]),
        ), patch(
            "tradingagents.recommend.orchestrator.run_stage_screener",
            return_value=StageOutcome(stage_name="screener", candidates=[winner, runner]),
        ), patch(
            "tradingagents.recommend.orchestrator.run_stage_market",
            return_value=StageOutcome(stage_name="market", candidates=[winner, runner]),
        ), patch(
            "tradingagents.recommend.orchestrator.run_stage_analysts",
            return_value=StageOutcome(stage_name="analysts", candidates=[winner, runner]),
        ), patch(
            "tradingagents.recommend.orchestrator.run_stage_debate",
            return_value=StageOutcome(stage_name="debate", candidates=[winner, runner]),
        ), patch(
            "tradingagents.recommend.orchestrator.run_stage_research",
            return_value=StageOutcome(stage_name="research", candidates=[winner, runner]),
        ), patch(
            "tradingagents.recommend.orchestrator.run_stage_decision",
            return_value=StageOutcome(stage_name="decision", candidates=[winner, runner]),
        ), patch(
            "tradingagents.recommend.orchestrator.build_llm",
            return_value=MagicMock(),
        ):
            result = run_full(
                mds_client=MagicMock(),
                as_of_date="2026-01-05",
                now=TRADING_DAY,
            )
        assert len(result.stage_outcomes) == 7    # all 7 ran
        assert result.skipped_stages == []
        assert [c.ticker for c in result.final] == ["WINNER", "RUNNER"]
        assert result.final[0].final_score > result.final[1].final_score
