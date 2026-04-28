"""Tests para a etapa 2 (Market Analyst com structured output)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.recommend.schemas import MarketVerdict
from tradingagents.recommend.stage_market import (
    DEFAULT_CONFIDENCE_MIN,
    _build_prompt,
    run,
)
from tradingagents.recommend.types import Candidate


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_candidate(ticker: str, screener_conf: float = 0.5) -> Candidate:
    return Candidate(
        ticker=ticker,
        confidence_history={"screener": screener_conf},
        reports={
            "screener": f"Technical screener PASS: RSI=60.0, VolOsc=+15.00%, DistSMA50=+1.00%",
        },
    )


def _stub_structured_llm(verdicts: dict[str, MarketVerdict]):
    """Mock para `structured_llm` que devolve um veredito por ticker.

    Inspeciona o prompt user-message pra extrair o ticker e devolve o
    veredito programado. Se um ticker não está no dict, raise to simulate
    LLM failure.
    """
    structured = MagicMock()

    def _invoke(prompt):
        user_msg = next(m for m in prompt if m["role"] == "user")
        # extrai o ticker de "Ticker: AAPL\n..."
        ticker = user_msg["content"].split("\n", 1)[0].split(":", 1)[1].strip()
        if ticker in verdicts:
            return verdicts[ticker]
        raise RuntimeError(f"no canned verdict for {ticker}")

    structured.invoke.side_effect = _invoke
    return structured


def _patch_bind(structured):
    """Substitui bind_structured do stage_market por um stub que devolve `structured`."""
    return patch(
        "tradingagents.recommend.stage_market.bind_structured",
        return_value=structured,
    )


# ── Run end-to-end ───────────────────────────────────────────────────────────


class TestStageMarketRun:
    def test_pass_above_threshold_advances(self):
        verdict = MarketVerdict(
            verdict="PASS", confidence=0.8, direction="long",
            narrative="Strong momentum.",
        )
        structured = _stub_structured_llm({"AAPL": verdict})
        candidates = [_make_candidate("AAPL")]

        with _patch_bind(structured):
            outcome = run(candidates, as_of_date="2026-04-24", llm=MagicMock())

        c = outcome.candidates[0]
        assert c.verdict == "PASS"
        assert c.confidence_history["market"] == 0.8
        assert "Strong momentum" in c.reports["market"]
        # screener confidence preserved
        assert c.confidence_history["screener"] == 0.5

    def test_no_pass_verdict_eliminates(self):
        verdict = MarketVerdict(
            verdict="NO_PASS", confidence=0.7, direction="none",
            narrative="Indicators conflict.",
        )
        structured = _stub_structured_llm({"AAPL": verdict})

        with _patch_bind(structured):
            outcome = run([_make_candidate("AAPL")], as_of_date="2026-04-24", llm=MagicMock())

        c = outcome.candidates[0]
        assert c.verdict == "NO_PASS"
        assert c.elimination.stage == "market"
        assert "conflict" in c.elimination.reason
        assert c.elimination.confidence == 0.7

    def test_pass_below_threshold_eliminates(self):
        verdict = MarketVerdict(
            verdict="PASS", confidence=0.4, direction="long",
            narrative="Weak signal.",
        )
        structured = _stub_structured_llm({"AAPL": verdict})

        with _patch_bind(structured):
            outcome = run([_make_candidate("AAPL")], as_of_date="2026-04-24", llm=MagicMock())

        c = outcome.candidates[0]
        assert c.verdict == "NO_PASS"
        assert "below threshold" in c.elimination.reason

    def test_custom_confidence_threshold(self):
        verdict = MarketVerdict(
            verdict="PASS", confidence=0.55, direction="long",
            narrative="Marginal.",
        )
        structured = _stub_structured_llm({"AAPL": verdict})

        # threshold=0.5 should let it through
        with _patch_bind(structured):
            outcome = run(
                [_make_candidate("AAPL")],
                as_of_date="2026-04-24",
                llm=MagicMock(),
                confidence_min=0.5,
            )
        assert outcome.candidates[0].verdict == "PASS"

    def test_llm_exception_marks_no_pass(self):
        # Stub returns no canned verdict for "GHOST" → raises inside _invoke
        structured = _stub_structured_llm({"AAPL": MarketVerdict(
            verdict="PASS", confidence=0.8, direction="long", narrative="ok",
        )})
        candidates = [_make_candidate("AAPL"), _make_candidate("GHOST")]

        with _patch_bind(structured):
            outcome = run(candidates, as_of_date="2026-04-24", llm=MagicMock(),
                          max_workers=2)

        ghost = next(c for c in outcome.candidates if c.ticker == "GHOST")
        aapl = next(c for c in outcome.candidates if c.ticker == "AAPL")
        assert ghost.verdict == "NO_PASS"
        assert "RuntimeError" in ghost.elimination.reason or "failed" in ghost.elimination.reason.lower()
        assert aapl.verdict == "PASS"

    def test_provider_without_structured_output_marks_all_no_pass(self):
        with patch(
            "tradingagents.recommend.stage_market.bind_structured",
            return_value=None,
        ):
            outcome = run(
                [_make_candidate("AAPL"), _make_candidate("MSFT")],
                as_of_date="2026-04-24",
                llm=MagicMock(),
            )
        for c in outcome.candidates:
            assert c.verdict == "NO_PASS"
            assert "structured output unsupported" in c.elimination.reason

    def test_empty_candidates_short_circuits(self):
        # Should not even try to build the LLM
        outcome = run([], as_of_date="2026-04-24", llm=MagicMock())
        assert outcome.candidates == []
        assert outcome.stage_name == "market"

    def test_preserves_input_order_after_parallel_execution(self):
        verdicts = {
            t: MarketVerdict(verdict="PASS", confidence=0.9, direction="long", narrative=t)
            for t in ["AAPL", "MSFT", "GOOG", "TSLA"]
        }
        structured = _stub_structured_llm(verdicts)
        candidates = [_make_candidate(t) for t in ["AAPL", "MSFT", "GOOG", "TSLA"]]

        with _patch_bind(structured):
            outcome = run(candidates, as_of_date="2026-04-24", llm=MagicMock(),
                          max_workers=4)

        assert [c.ticker for c in outcome.candidates] == ["AAPL", "MSFT", "GOOG", "TSLA"]

    def test_does_not_mutate_input_candidates(self):
        verdict = MarketVerdict(
            verdict="PASS", confidence=0.8, direction="long", narrative="ok",
        )
        structured = _stub_structured_llm({"AAPL": verdict})
        original = _make_candidate("AAPL")
        snapshot = (
            dict(original.confidence_history),
            dict(original.reports),
            original.elimination,
        )

        with _patch_bind(structured):
            run([original], as_of_date="2026-04-24", llm=MagicMock())

        assert original.confidence_history == snapshot[0]
        assert original.reports == snapshot[1]
        assert original.elimination == snapshot[2]


# ── Prompt building ──────────────────────────────────────────────────────────


class TestBuildPrompt:
    def test_includes_ticker_and_screener_report(self):
        cand = _make_candidate("AAPL")
        prompt = _build_prompt(cand, "2026-04-24")
        assert prompt[0]["role"] == "system"
        assert prompt[1]["role"] == "user"
        assert "AAPL" in prompt[1]["content"]
        assert "2026-04-24" in prompt[1]["content"]
        assert "Technical screener PASS" in prompt[1]["content"]

    def test_handles_missing_screener_report(self):
        cand = Candidate(ticker="X")
        prompt = _build_prompt(cand, "2026-04-24")
        assert "no prior technical report" in prompt[1]["content"]


# ── Live integration (opt-in) ────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RECOMMEND_LIVE_LLM") is None
    or os.environ.get("OPENROUTER_API_KEY") is None,
    reason=(
        "set RECOMMEND_LIVE_LLM=1 (with OPENROUTER_API_KEY in env) "
        "to call OpenRouter+Kimi for real"
    ),
)
class TestStageMarketLiveLLM:
    """Faz uma chamada real ao Kimi via OpenRouter. Custo ~$0.005 por ticker."""

    def test_live_kimi_call_returns_valid_verdict(self):
        from tradingagents.recommend.llm import LLMConfig

        candidates = [_make_candidate("AAPL")]
        outcome = run(
            candidates,
            as_of_date="2026-04-24",
            llm_config=LLMConfig(),
            max_workers=1,
        )
        assert len(outcome.candidates) == 1
        c = outcome.candidates[0]
        # Either accepted or rejected, but the schema is satisfied:
        if c.verdict == "PASS":
            assert 0.0 <= c.confidence_history["market"] <= 1.0
            assert "market" in c.reports
        else:
            assert c.elimination.stage == "market"
