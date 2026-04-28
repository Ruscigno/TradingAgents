"""Tests para a etapa 3 (Sentiment + News + Fundamentals + composite)."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.recommend.schemas import (
    FundamentalsVerdict,
    NewsVerdict,
    SentimentVerdict,
)
from tradingagents.recommend.stage_analysts import (
    DEFAULT_WEIGHTS,
    composite_confidence,
    run,
)
from tradingagents.recommend.types import Candidate


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_candidate(
    ticker: str,
    market_conf: float = 0.7,
    screener_conf: float = 0.5,
) -> Candidate:
    return Candidate(
        ticker=ticker,
        confidence_history={"screener": screener_conf, "market": market_conf},
        reports={"market": "PASS via Stage 2 (synthetic for tests)"},
    )


def _verdict(cls, verdict: str, conf: float, narrative: str = "x") -> Any:
    return cls(verdict=verdict, confidence=conf, narrative=narrative)


def _stub_clients(per_role: dict[str, Any]):
    """Build three structured-LLM stubs, each returning the same canned verdict.

    `per_role` maps role name → callable(ticker) → verdict OR a fixed verdict.
    """
    def _maker(role: str, role_default: Any):
        client = MagicMock()
        def _invoke(prompt):
            user = next(m for m in prompt if m["role"] == "user")
            ticker = user["content"].split("\n", 1)[0].split(":", 1)[1].strip()
            entry = per_role.get(role, role_default)
            if callable(entry):
                return entry(ticker)
            return entry
        client.invoke.side_effect = _invoke
        return client

    return _maker("sentiment", per_role["sentiment_default"]), \
           _maker("news", per_role["news_default"]), \
           _maker("fundamentals", per_role["fundamentals_default"])


def _patch_bind_with_clients(s_client, n_client, f_client):
    """Patch bind_structured to return our three pre-canned clients in order."""
    return patch(
        "tradingagents.recommend.stage_analysts.bind_structured",
        side_effect=[s_client, n_client, f_client],
    )


def _no_data_fetchers():
    """Return identity stubs that yield trivial strings."""
    return (
        lambda t, s, e: f"news for {t} {s}..{e}",
        lambda t, d: f"fundamentals for {t} as of {d}",
    )


# ── composite_confidence ─────────────────────────────────────────────────────


class TestCompositeConfidence:
    def test_all_present_default_weights(self):
        result = composite_confidence({
            "market": 1.0, "sentiment": 1.0, "news": 1.0, "fundamentals": 1.0,
        })
        assert result == pytest.approx(1.0)

    def test_all_zero(self):
        result = composite_confidence({
            "market": 0.0, "sentiment": 0.0, "news": 0.0, "fundamentals": 0.0,
        })
        assert result == pytest.approx(0.0)

    def test_known_mix_default_weights(self):
        # weights: 0.35 m, 0.15 s, 0.20 n, 0.30 f
        result = composite_confidence({
            "market": 1.0, "sentiment": 0.0, "news": 1.0, "fundamentals": 0.0,
        })
        assert result == pytest.approx(0.55, abs=1e-9)

    def test_redistributes_when_stage_missing(self):
        # only market present → composite is just market_conf
        result = composite_confidence({"market": 0.8})
        assert result == pytest.approx(0.8)

    def test_redistributes_partial(self):
        # market 0.35 + news 0.20 = 0.55 weight; renormalize
        result = composite_confidence({"market": 1.0, "news": 0.0})
        # market gets 0.35/0.55 ≈ 0.636, news gets 0.20/0.55 ≈ 0.364
        assert result == pytest.approx(0.35 / 0.55, abs=1e-3)

    def test_empty_returns_zero(self):
        assert composite_confidence({}) == 0.0


# ── End-to-end run ──────────────────────────────────────────────────────────


class TestStageAnalystsRun:
    def test_strong_pass_advances_with_composite(self):
        sent = _verdict(SentimentVerdict, "PASS", 0.8, "bullish tone")
        news = _verdict(NewsVerdict, "PASS", 0.8, "supportive news")
        fund = _verdict(FundamentalsVerdict, "PASS", 0.8, "strong fundamentals")
        s, n, f = _stub_clients({
            "sentiment_default": sent,
            "news_default": news,
            "fundamentals_default": fund,
        })
        news_fn, fund_fn = _no_data_fetchers()

        with _patch_bind_with_clients(s, n, f):
            outcome = run(
                [_make_candidate("AAPL")],
                as_of_date="2026-04-24",
                llm=MagicMock(),
                news_fetcher=news_fn,
                fundamentals_fetcher=fund_fn,
                max_candidate_workers=1,
            )

        c = outcome.candidates[0]
        assert c.verdict == "PASS"
        assert "analysts" in c.confidence_history
        # Composite of (m=0.7, s=0.8, n=0.8, f=0.8) = 0.35*0.7 + 0.15*0.8 + 0.20*0.8 + 0.30*0.8
        # = 0.245 + 0.12 + 0.16 + 0.24 = 0.765
        assert c.confidence_history["analysts"] == pytest.approx(0.765, abs=1e-3)
        # Individual analyst confidences also stored
        assert c.confidence_history["sentiment"] == 0.8
        assert c.confidence_history["news"] == 0.8
        assert c.confidence_history["fundamentals"] == 0.8

    def test_low_composite_eliminates(self):
        sent = _verdict(SentimentVerdict, "PASS", 0.3, "weak")
        news = _verdict(NewsVerdict, "PASS", 0.3, "weak")
        fund = _verdict(FundamentalsVerdict, "PASS", 0.3, "weak")
        s, n, f = _stub_clients({
            "sentiment_default": sent, "news_default": news, "fundamentals_default": fund,
        })
        news_fn, fund_fn = _no_data_fetchers()
        cand = _make_candidate("X", market_conf=0.3)

        with _patch_bind_with_clients(s, n, f):
            outcome = run(
                [cand],
                as_of_date="2026-04-24",
                llm=MagicMock(),
                news_fetcher=news_fn,
                fundamentals_fetcher=fund_fn,
            )

        c = outcome.candidates[0]
        assert c.verdict == "NO_PASS"
        assert "below threshold" in c.elimination.reason
        # Composite = 0.3 (all dims at 0.3)
        assert c.elimination.confidence == pytest.approx(0.3, abs=1e-3)

    def test_veto_eliminates_even_with_high_composite(self):
        # Three at 0.9 PASS → composite high. But news returns NO_PASS conf=0.95 → veto.
        sent = _verdict(SentimentVerdict, "PASS", 0.9)
        news = _verdict(NewsVerdict, "NO_PASS", 0.95, "regulatory disaster announced")
        fund = _verdict(FundamentalsVerdict, "PASS", 0.9)
        s, n, f = _stub_clients({
            "sentiment_default": sent, "news_default": news, "fundamentals_default": fund,
        })
        news_fn, fund_fn = _no_data_fetchers()

        with _patch_bind_with_clients(s, n, f):
            outcome = run(
                [_make_candidate("AAPL", market_conf=0.9)],
                as_of_date="2026-04-24",
                llm=MagicMock(),
                news_fetcher=news_fn,
                fundamentals_fetcher=fund_fn,
            )

        c = outcome.candidates[0]
        assert c.verdict == "NO_PASS"
        assert "veto by news" in c.elimination.reason
        assert "regulatory disaster" in c.elimination.reason

    def test_high_conf_no_pass_below_veto_does_not_veto(self):
        # NO_PASS at conf 0.84 (just below default veto 0.85) — no veto.
        # Composite still must clear 0.65 to PASS.
        sent = _verdict(SentimentVerdict, "PASS", 0.9)
        news = _verdict(NewsVerdict, "NO_PASS", 0.84, "mild concern")
        fund = _verdict(FundamentalsVerdict, "PASS", 0.9)
        s, n, f = _stub_clients({
            "sentiment_default": sent, "news_default": news, "fundamentals_default": fund,
        })
        news_fn, fund_fn = _no_data_fetchers()

        with _patch_bind_with_clients(s, n, f):
            outcome = run(
                [_make_candidate("AAPL", market_conf=0.9)],
                as_of_date="2026-04-24",
                llm=MagicMock(),
                news_fetcher=news_fn,
                fundamentals_fetcher=fund_fn,
            )

        c = outcome.candidates[0]
        # Composite = 0.35*0.9 + 0.15*0.9 + 0.20*0.84 + 0.30*0.9 = 0.888
        assert c.verdict == "PASS"
        assert c.confidence_history["analysts"] == pytest.approx(0.888, abs=1e-3)

    def test_news_fetch_failure_eliminates(self):
        sent = _verdict(SentimentVerdict, "PASS", 0.9)
        news = _verdict(NewsVerdict, "PASS", 0.9)
        fund = _verdict(FundamentalsVerdict, "PASS", 0.9)
        s, n, f = _stub_clients({
            "sentiment_default": sent, "news_default": news, "fundamentals_default": fund,
        })
        bad_news = MagicMock(side_effect=RuntimeError("yfinance down"))
        good_fund = lambda t, d: "ok"

        with _patch_bind_with_clients(s, n, f):
            outcome = run(
                [_make_candidate("AAPL")],
                as_of_date="2026-04-24",
                llm=MagicMock(),
                news_fetcher=bad_news,
                fundamentals_fetcher=good_fund,
            )

        c = outcome.candidates[0]
        assert c.verdict == "NO_PASS"
        assert "news fetch failed" in c.elimination.reason

    def test_one_analyst_failure_falls_back_to_other_two(self):
        # Sentiment errors. News + fundamentals at 0.8 → composite redistributes.
        sent = MagicMock()
        sent.invoke.side_effect = RuntimeError("sentiment LLM error")
        news = _verdict(NewsVerdict, "PASS", 0.8)
        fund = _verdict(FundamentalsVerdict, "PASS", 0.8)
        s, n, f = _stub_clients({
            "sentiment_default": sent, "news_default": news, "fundamentals_default": fund,
        })
        # override sentiment client with a MagicMock that fails (twice — handles retry)
        s = MagicMock()
        s.invoke.side_effect = RuntimeError("sentiment LLM error")
        news_fn, fund_fn = _no_data_fetchers()

        with _patch_bind_with_clients(s, n, f):
            outcome = run(
                [_make_candidate("AAPL", market_conf=0.8)],
                as_of_date="2026-04-24",
                llm=MagicMock(),
                news_fetcher=news_fn,
                fundamentals_fetcher=fund_fn,
            )

        c = outcome.candidates[0]
        # market 0.8 + news 0.8 + fund 0.8 (sentiment skipped) — redistribute weights.
        # weight totals without sentiment = 0.35+0.20+0.30 = 0.85. All three at 0.8 → composite 0.8.
        assert c.verdict == "PASS"
        assert "sentiment" not in c.confidence_history  # sentiment failed
        assert c.confidence_history["analysts"] == pytest.approx(0.8, abs=1e-3)

    def test_provider_without_structured_output_marks_all_no_pass(self):
        with patch(
            "tradingagents.recommend.stage_analysts.bind_structured",
            return_value=None,
        ):
            outcome = run(
                [_make_candidate("AAPL")],
                as_of_date="2026-04-24",
                llm=MagicMock(),
            )
        c = outcome.candidates[0]
        assert c.verdict == "NO_PASS"
        assert "structured output unsupported" in c.elimination.reason

    def test_empty_candidates_short_circuits(self):
        outcome = run([], as_of_date="2026-04-24", llm=MagicMock())
        assert outcome.candidates == []

    def test_does_not_mutate_input(self):
        sent = _verdict(SentimentVerdict, "PASS", 0.9)
        news = _verdict(NewsVerdict, "PASS", 0.9)
        fund = _verdict(FundamentalsVerdict, "PASS", 0.9)
        s, n, f = _stub_clients({
            "sentiment_default": sent, "news_default": news, "fundamentals_default": fund,
        })
        news_fn, fund_fn = _no_data_fetchers()
        original = _make_candidate("AAPL")
        original_history = dict(original.confidence_history)
        original_reports = dict(original.reports)

        with _patch_bind_with_clients(s, n, f):
            run(
                [original],
                as_of_date="2026-04-24",
                llm=MagicMock(),
                news_fetcher=news_fn,
                fundamentals_fetcher=fund_fn,
            )

        assert original.confidence_history == original_history
        assert original.reports == original_reports


# ── Live integration ────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RECOMMEND_LIVE_LLM") is None
    or os.environ.get("OPENROUTER_API_KEY") is None,
    reason="set RECOMMEND_LIVE_LLM=1 (with OPENROUTER_API_KEY) for live OpenRouter+yfinance",
)
class TestStageAnalystsLive:
    def test_live_full_pipeline_for_single_ticker(self):
        """Full Stage 3 with real Kimi + yfinance for a single liquid ticker."""
        from tradingagents.recommend.llm import LLMConfig

        cand = _make_candidate("AAPL", market_conf=0.7)
        outcome = run(
            [cand],
            as_of_date="2026-04-24",
            llm_config=LLMConfig(),
            max_candidate_workers=1,
        )
        c = outcome.candidates[0]
        # We don't assert PASS or NO_PASS — just that the pipeline returns a coherent
        # result with all three analysts contributing.
        assert outcome.stage_name == "analysts"
        if c.verdict == "PASS":
            assert "sentiment" in c.confidence_history
            assert "news" in c.confidence_history
            assert "fundamentals" in c.confidence_history
            assert "analysts" in c.confidence_history
        else:
            assert c.elimination.stage == "analysts"
