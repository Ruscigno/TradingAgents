"""Compact tests for stages 4 (debate), 5 (research), 6 (decision).

The full pattern (mocked structured LLM, retries, parallelism) was
exhaustively tested for stages 2 and 3. These tests cover only the
essentials: schema wiring, PASS/NO_PASS branches, and structured-output
unsupported fallback.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tradingagents.recommend.schemas import (
    BearCase,
    BullCase,
    DebateVerdict,
    ResearchPlan,
    TradeDecision,
)
from tradingagents.recommend.stage_debate import run as stage_debate
from tradingagents.recommend.stage_decision import PositionSizing
from tradingagents.recommend.stage_decision import run as stage_decision
from tradingagents.recommend.stage_research import run as stage_research
from tradingagents.recommend.types import Candidate


def _seed(ticker: str = "AAPL") -> Candidate:
    return Candidate(
        ticker=ticker,
        confidence_history={"screener": 0.5, "market": 0.7, "analysts": 0.75},
        reports={
            "screener": "RSI=60",
            "market": "Stage 2 PASS",
            "analysts": "Composite 0.75 (PASS in all 3)",
        },
    )


# ── Stage 4 — debate ─────────────────────────────────────────────────────────


class TestStageDebate:
    def test_pass_path_records_confidence_and_report(self):
        bull = MagicMock()
        bear = MagicMock()
        judge = MagicMock()
        bull.invoke.return_value = BullCase(thesis="Long thesis", key_arguments=["a", "b"])
        bear.invoke.return_value = BearCase(thesis="Bear thesis", key_arguments=["x"])
        judge.invoke.return_value = DebateVerdict(
            verdict="PASS", confidence=0.8, direction="long", rationale="bull case wins",
        )

        with patch(
            "tradingagents.recommend.stage_debate.bind_structured",
            side_effect=[bull, bear, judge],
        ):
            outcome = stage_debate(
                [_seed()],
                as_of_date="2026-04-24",
                llm=MagicMock(),
                max_candidate_workers=1,
            )

        c = outcome.candidates[0]
        assert c.verdict == "PASS"
        assert c.confidence_history["debate"] == 0.8
        assert "Bull:" in c.reports["debate"]
        assert "Bear:" in c.reports["debate"]
        assert "long" in c.reports["debate"].lower()

    def test_no_pass_eliminates(self):
        bull = MagicMock()
        bear = MagicMock()
        judge = MagicMock()
        bull.invoke.return_value = BullCase(thesis="weak", key_arguments=["a"])
        bear.invoke.return_value = BearCase(thesis="strong", key_arguments=["x", "y"])
        judge.invoke.return_value = DebateVerdict(
            verdict="NO_PASS", confidence=0.85, direction="none", rationale="bear wins",
        )

        with patch(
            "tradingagents.recommend.stage_debate.bind_structured",
            side_effect=[bull, bear, judge],
        ):
            outcome = stage_debate([_seed()], as_of_date="2026-04-24", llm=MagicMock())

        c = outcome.candidates[0]
        assert c.verdict == "NO_PASS"
        assert "bear wins" in c.elimination.reason

    def test_low_confidence_eliminates(self):
        bull = MagicMock(); bear = MagicMock(); judge = MagicMock()
        bull.invoke.return_value = BullCase(thesis="ok", key_arguments=["a"])
        bear.invoke.return_value = BearCase(thesis="ok", key_arguments=["a"])
        judge.invoke.return_value = DebateVerdict(
            verdict="PASS", confidence=0.4, direction="long", rationale="hesitant",
        )

        with patch(
            "tradingagents.recommend.stage_debate.bind_structured",
            side_effect=[bull, bear, judge],
        ):
            outcome = stage_debate([_seed()], as_of_date="2026-04-24", llm=MagicMock())

        assert outcome.candidates[0].verdict == "NO_PASS"

    def test_unsupported_provider_marks_all_no_pass(self):
        with patch(
            "tradingagents.recommend.stage_debate.bind_structured",
            return_value=None,
        ):
            outcome = stage_debate([_seed()], as_of_date="2026-04-24", llm=MagicMock())
        assert outcome.candidates[0].verdict == "NO_PASS"
        assert "structured output unsupported" in outcome.candidates[0].elimination.reason


# ── Stage 5 — research manager ───────────────────────────────────────────────


class TestStageResearch:
    def test_pass_path(self):
        structured = MagicMock()
        structured.invoke.return_value = ResearchPlan(
            verdict="PASS", confidence=0.85, direction="long",
            investment_plan="## Plan\nBuy on dip.",
            key_risks=["risk one"],
        )
        with patch(
            "tradingagents.recommend.stage_research.bind_structured",
            return_value=structured,
        ):
            outcome = stage_research(
                [_seed()],
                as_of_date="2026-04-24",
                llm=MagicMock(),
                max_candidate_workers=1,
            )
        c = outcome.candidates[0]
        assert c.verdict == "PASS"
        assert c.confidence_history["research"] == 0.85
        assert "Buy on dip" in c.reports["research"]
        assert "risk one" in c.reports["research"]

    def test_no_pass_path(self):
        structured = MagicMock()
        structured.invoke.return_value = ResearchPlan(
            verdict="NO_PASS", confidence=0.6, direction="none",
            investment_plan="conflict between technical and fundamentals",
            key_risks=["regulatory"],
        )
        with patch(
            "tradingagents.recommend.stage_research.bind_structured",
            return_value=structured,
        ):
            outcome = stage_research([_seed()], as_of_date="2026-04-24", llm=MagicMock())
        c = outcome.candidates[0]
        assert c.verdict == "NO_PASS"
        assert "conflict" in c.elimination.reason


# ── Stage 6 — decision ───────────────────────────────────────────────────────


class TestStageDecision:
    def test_buy_decision_path(self):
        pm = MagicMock()
        pm.invoke.return_value = TradeDecision(
            verdict="PASS", confidence=0.85, action="BUY",
            stop_loss=178.5, take_profit=195.0,
            rationale="Conviction trade, defined risk.",
        )
        # Free-text LLM for risk debate
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="risk perspective text")

        with patch(
            "tradingagents.recommend.stage_decision.bind_structured",
            return_value=pm,
        ):
            outcome = stage_decision(
                [_seed()],
                as_of_date="2026-04-24",
                llm=llm,
                max_candidate_workers=1,
            )
        c = outcome.candidates[0]
        assert c.verdict == "PASS"
        assert c.confidence_history["decision"] == 0.85
        assert "BUY" in c.reports["decision"]
        assert c.reports["_decision_action"] == "BUY"
        assert c.reports["_decision_stop_loss"] == "178.5"
        assert c.reports["_decision_take_profit"] == "195.0"
        assert c.reports["_decision_position_size_pct"] == "2.0"

    def test_hold_eliminates(self):
        pm = MagicMock()
        pm.invoke.return_value = TradeDecision(
            verdict="NO_PASS", confidence=0.7, action="HOLD",
            stop_loss=0.0, take_profit=0.0,
            rationale="Wait for clearer signal.",
        )
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="x")

        with patch(
            "tradingagents.recommend.stage_decision.bind_structured",
            return_value=pm,
        ):
            outcome = stage_decision([_seed()], as_of_date="2026-04-24", llm=llm)
        c = outcome.candidates[0]
        assert c.verdict == "NO_PASS"
        assert "HOLD" in c.elimination.reason

    def test_position_sizing_from_config(self):
        pm = MagicMock()
        pm.invoke.return_value = TradeDecision(
            verdict="PASS", confidence=0.85, action="BUY",
            stop_loss=100.0, take_profit=110.0, rationale="ok",
        )
        llm = MagicMock(); llm.invoke.return_value = MagicMock(content="x")

        with patch(
            "tradingagents.recommend.stage_decision.bind_structured",
            return_value=pm,
        ):
            outcome = stage_decision(
                [_seed()],
                as_of_date="2026-04-24",
                llm=llm,
                position=PositionSizing(size_pct=5.0, max_concurrent_trades=10),
            )
        c = outcome.candidates[0]
        assert c.reports["_decision_position_size_pct"] == "5.0"
