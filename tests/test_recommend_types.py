"""Schema do contrato entre etapas — sanity tests."""

from __future__ import annotations

from tradingagents.recommend.types import Candidate, Elimination, StageOutcome


class TestCandidateVerdict:
    def test_default_passes(self):
        c = Candidate(ticker="AAPL")
        assert c.verdict == "PASS"

    def test_eliminated_fails(self):
        c = Candidate(
            ticker="AAPL",
            elimination=Elimination(stage="screener", reason="RSI=82", confidence=0.0),
        )
        assert c.verdict == "NO_PASS"


class TestStageOutcome:
    def test_partition_into_passed_and_eliminated(self):
        passed_one = Candidate(ticker="AAPL")
        eliminated_one = Candidate(
            ticker="XYZ",
            elimination=Elimination(stage="screener", reason="vol osc < 0", confidence=0.0),
        )
        outcome = StageOutcome(
            stage_name="screener",
            candidates=[passed_one, eliminated_one],
        )
        assert outcome.passed == [passed_one]
        assert outcome.eliminated == [eliminated_one]

    def test_empty_outcome(self):
        outcome = StageOutcome(stage_name="universe", candidates=[])
        assert outcome.passed == []
        assert outcome.eliminated == []
