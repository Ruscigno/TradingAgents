"""Tests para a etapa 1 (technical screener) da cascata de recomendações."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from tradingagents.recommend.stage_screener import (
    _compute_confidence,
    _normalize_one_sided,
    _normalize_two_sided,
    run,
)
from tradingagents.recommend.types import Candidate
from tradingagents.screener.technical_screener import (
    ScreenerConfig,
    ScreenerResult,
)


def _stub_screener(results_by_ticker: dict[str, ScreenerResult]):
    """Build a duck-typed screener that returns canned results."""
    stub = MagicMock()
    stub.screen.side_effect = lambda tickers, as_of_date: [
        results_by_ticker[t] for t in tickers if t in results_by_ticker
    ]
    return stub


# ── End-to-end run ───────────────────────────────────────────────────────────


class TestStageScreenerRun:
    def test_passing_candidate_gets_confidence_in_history(self):
        results = {
            "AAPL": ScreenerResult(
                ticker="AAPL",
                passed=True,
                rsi=60.0,         # exact center of [50, 70] → c_rsi = 1.0
                vol_osc=15.0,     # half of reference strength 30 → c_vol = 0.5
                dist_ma_pct=0.0,  # center of [-5, 5] → c_sma = 1.0
                reason="PASS",
            ),
        }
        candidates = [Candidate(ticker="AAPL")]

        outcome = run(
            candidates,
            "2026-04-27",
            screener=_stub_screener(results),
        )

        assert outcome.stage_name == "screener"
        assert len(outcome.candidates) == 1
        c = outcome.candidates[0]
        assert c.verdict == "PASS"
        assert c.elimination is None
        # mean of (1.0, 1.0, 0.5) = 0.833...
        assert c.confidence_history["screener"] == pytest.approx(0.8333, abs=1e-3)
        assert "RSI=60.0" in c.reports["screener"]

    def test_failing_candidate_records_elimination(self):
        results = {
            "BAD": ScreenerResult(
                ticker="BAD",
                passed=False,
                rsi=85.0,
                vol_osc=10.0,
                dist_ma_pct=2.0,
                reason="RSI 85.0 outside [50.0, 70.0]",
            ),
        }
        candidates = [Candidate(ticker="BAD")]

        outcome = run(
            candidates,
            "2026-04-27",
            screener=_stub_screener(results),
        )

        c = outcome.candidates[0]
        assert c.verdict == "NO_PASS"
        assert c.elimination is not None
        assert c.elimination.stage == "screener"
        assert "RSI 85.0" in c.elimination.reason
        assert "screener" not in c.confidence_history

    def test_partition_into_passed_and_eliminated(self):
        results = {
            "AAPL": ScreenerResult(
                ticker="AAPL", passed=True, rsi=60.0, vol_osc=15.0,
                dist_ma_pct=0.0, reason="PASS",
            ),
            "BAD": ScreenerResult(
                ticker="BAD", passed=False, rsi=85.0, vol_osc=10.0,
                dist_ma_pct=2.0, reason="RSI 85.0 outside [50.0, 70.0]",
            ),
        }
        candidates = [Candidate(ticker="AAPL"), Candidate(ticker="BAD")]

        outcome = run(
            candidates,
            "2026-04-27",
            screener=_stub_screener(results),
        )

        assert [c.ticker for c in outcome.passed] == ["AAPL"]
        assert [c.ticker for c in outcome.eliminated] == ["BAD"]

    def test_empty_candidates_short_circuits(self):
        outcome = run([], "2026-04-27", screener=MagicMock())
        assert outcome.candidates == []
        assert outcome.stage_name == "screener"

    def test_missing_screener_result_eliminates(self):
        # Screener returned nothing for "GHOST" — must not crash, must mark NO_PASS.
        results = {"AAPL": ScreenerResult(
            ticker="AAPL", passed=True, rsi=60.0, vol_osc=15.0,
            dist_ma_pct=0.0, reason="PASS",
        )}
        candidates = [Candidate(ticker="AAPL"), Candidate(ticker="GHOST")]

        outcome = run(
            candidates,
            "2026-04-27",
            screener=_stub_screener(results),
        )

        ghost = next(c for c in outcome.candidates if c.ticker == "GHOST")
        assert ghost.verdict == "NO_PASS"
        assert "no screener result" in ghost.elimination.reason

    def test_requires_screener_or_mds_client(self):
        with pytest.raises(ValueError, match="screener.*mds_client"):
            run(
                [Candidate(ticker="AAPL")],
                "2026-04-27",
                # no screener, no mds_client
            )


# ── Confidence math ──────────────────────────────────────────────────────────


class TestNormalizeTwoSided:
    @pytest.mark.parametrize(
        "value,lo,hi,expected",
        [
            (60.0, 50.0, 70.0, 1.0),       # exact center
            (50.0, 50.0, 70.0, 0.0),       # left edge
            (70.0, 50.0, 70.0, 0.0),       # right edge
            (55.0, 50.0, 70.0, 0.5),       # quarter-way in
            (40.0, 50.0, 70.0, 0.0),       # outside, clamped
            (80.0, 50.0, 70.0, 0.0),       # outside, clamped
            (None, 50.0, 70.0, 0.0),       # missing
            (60.0, 70.0, 50.0, 0.0),       # malformed range (hi <= lo)
        ],
    )
    def test_known_points(self, value, lo, hi, expected):
        assert _normalize_two_sided(value, lo, hi) == pytest.approx(expected, abs=1e-9)


class TestNormalizeOneSided:
    @pytest.mark.parametrize(
        "value,threshold,reference,expected",
        [
            (0.0, 0.0, 30.0, 0.0),         # at threshold
            (15.0, 0.0, 30.0, 0.5),        # half ramp
            (30.0, 0.0, 30.0, 1.0),        # saturated
            (60.0, 0.0, 30.0, 1.0),        # past saturation, still 1.0
            (-10.0, 0.0, 30.0, 0.0),       # below threshold
            (None, 0.0, 30.0, 0.0),
            (15.0, 0.0, 0.0, 0.0),         # malformed reference
        ],
    )
    def test_known_points(self, value, threshold, reference, expected):
        assert _normalize_one_sided(value, threshold, reference) == pytest.approx(
            expected, abs=1e-9
        )


class TestComputeConfidence:
    def test_strong_pass_near_one(self):
        result = ScreenerResult(
            ticker="X", passed=True, rsi=60.0, vol_osc=30.0, dist_ma_pct=0.0,
            reason="PASS",
        )
        # all three indicators saturated → confidence = 1.0
        assert _compute_confidence(result, ScreenerConfig()) == pytest.approx(1.0)

    def test_borderline_pass_near_zero(self):
        result = ScreenerResult(
            ticker="X", passed=True, rsi=50.0, vol_osc=0.001, dist_ma_pct=4.99,
            reason="PASS",
        )
        # all three indicators basically at boundary → confidence ~ 0
        assert _compute_confidence(result, ScreenerConfig()) < 0.01

    def test_handles_missing_indicators(self):
        result = ScreenerResult(
            ticker="X", passed=False, rsi=None, vol_osc=None, dist_ma_pct=None,
            reason="No data",
        )
        assert _compute_confidence(result, ScreenerConfig()) == 0.0


# ── Live integration ─────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RECOMMEND_LIVE_MDS") is None,
    reason="set RECOMMEND_LIVE_MDS=1 (and have MDS at localhost:8080) to run",
)
class TestStageScreenerIntegration:
    """Bate no MDS real (REST API). Pula sem RECOMMEND_LIVE_MDS=1."""

    def test_live_run_a_few_tickers(self):
        from tradingagents.dataflows.mds_client import MDSClient

        client = MDSClient()
        candidates = [Candidate(ticker=t) for t in ["AAPL", "MSFT", "VTI"]]
        outcome = run(
            candidates,
            "2026-04-25",  # Sat — uses last trading day's data
            mds_client=client,
        )

        assert outcome.stage_name == "screener"
        assert len(outcome.candidates) == 3
        for c in outcome.candidates:
            if c.elimination is None:
                assert 0.0 <= c.confidence_history["screener"] <= 1.0
                assert "RSI=" in c.reports["screener"]
            else:
                assert c.elimination.stage == "screener"
