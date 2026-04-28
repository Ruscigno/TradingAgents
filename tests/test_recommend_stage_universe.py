"""Tests para a etapa 0 (universe selection via MCP).

Testes unitários usam um ``mcp_caller`` injetado para isolar do servidor
real. Há também um teste de integração marcado com ``@pytest.mark.integration``
que bate no MCP do market-data-service em ``http://localhost:8080/mcp`` e
só roda se o serviço estiver no ar.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from tradingagents.recommend.stage_universe import run


NOW = datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc)
FRESH = (NOW - timedelta(days=2)).isoformat().replace("+00:00", "Z")
STALE = (NOW - timedelta(days=10)).isoformat().replace("+00:00", "Z")


def _fake_caller(payload: dict[str, Any]):
    """Build a stub `call_tool_sync` that returns `payload` unchanged."""
    def _caller(url: str, name: str, args: dict[str, Any] | None = None) -> dict:
        assert name == "list_tickers", f"unexpected MCP call: {name}"
        return payload
    return _caller


class TestStageUniverse:
    def test_keeps_fresh_tickers(self):
        outcome = run(
            mcp_url="http://test/mcp",
            now=NOW,
            mcp_caller=_fake_caller({
                "count": 1,
                "tickers": [{"symbol": "AAPL", "last_fetch_1d": FRESH}],
            }),
        )
        assert outcome.stage_name == "universe"
        assert len(outcome.candidates) == 1
        assert outcome.candidates[0].ticker == "AAPL"
        assert outcome.candidates[0].verdict == "PASS"
        assert outcome.cost_usd == 0.0

    def test_drops_stale_tickers(self):
        outcome = run(
            mcp_url="http://test/mcp",
            now=NOW,
            mcp_caller=_fake_caller({
                "tickers": [
                    {"symbol": "AAPL", "last_fetch_1d": FRESH},
                    {"symbol": "OLDCO", "last_fetch_1d": STALE},
                ],
            }),
        )
        kept = [c.ticker for c in outcome.candidates]
        assert kept == ["AAPL"]

    def test_drops_tickers_without_last_fetch(self):
        outcome = run(
            mcp_url="http://test/mcp",
            now=NOW,
            mcp_caller=_fake_caller({
                "tickers": [
                    {"symbol": "GOOD", "last_fetch_1d": FRESH},
                    {"symbol": "NODATA", "last_fetch_1d": None},
                    {"symbol": "ALSONODATA"},  # missing field entirely
                ],
            }),
        )
        kept = [c.ticker for c in outcome.candidates]
        assert kept == ["GOOD"]

    def test_normalizes_ticker_to_upper(self):
        outcome = run(
            mcp_url="http://test/mcp",
            now=NOW,
            mcp_caller=_fake_caller({
                "tickers": [{"symbol": "aapl", "last_fetch_1d": FRESH}],
            }),
        )
        assert outcome.candidates[0].ticker == "AAPL"

    def test_empty_universe(self):
        outcome = run(
            mcp_url="http://test/mcp",
            now=NOW,
            mcp_caller=_fake_caller({"tickers": []}),
        )
        assert outcome.candidates == []
        assert outcome.stage_name == "universe"

    def test_missing_tickers_field(self):
        # market-data-service should always return `tickers`, but defend against
        # an empty/malformed response by treating it as an empty universe.
        outcome = run(
            mcp_url="http://test/mcp",
            now=NOW,
            mcp_caller=_fake_caller({}),
        )
        assert outcome.candidates == []

    def test_custom_staleness_window(self):
        # 7-day window should now keep the "stale" ticker (10 days old still fails,
        # but a 6-day-old one would pass).
        six_days_old = (NOW - timedelta(days=6)).isoformat().replace("+00:00", "Z")
        outcome = run(
            mcp_url="http://test/mcp",
            max_staleness_days=7,
            now=NOW,
            mcp_caller=_fake_caller({
                "tickers": [{"symbol": "MIDDLECO", "last_fetch_1d": six_days_old}],
            }),
        )
        assert [c.ticker for c in outcome.candidates] == ["MIDDLECO"]

    def test_unparseable_timestamp_drops_ticker(self):
        outcome = run(
            mcp_url="http://test/mcp",
            now=NOW,
            mcp_caller=_fake_caller({
                "tickers": [
                    {"symbol": "GOOD", "last_fetch_1d": FRESH},
                    {"symbol": "BAD", "last_fetch_1d": "not-a-date"},
                ],
            }),
        )
        assert [c.ticker for c in outcome.candidates] == ["GOOD"]


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RECOMMEND_MCP_URL") is None,
    reason="set RECOMMEND_MCP_URL=http://localhost:8080/mcp to run live MCP test",
)
class TestStageUniverseIntegration:
    """Bate no MCP real do market-data-service. Pula se RECOMMEND_MCP_URL não estiver set."""

    def test_live_list_tickers(self):
        url = os.environ["RECOMMEND_MCP_URL"]
        outcome = run(mcp_url=url)
        assert outcome.stage_name == "universe"
        # apenas exigimos que retorne sem erro; o conteúdo varia dia a dia
        for c in outcome.candidates:
            assert c.verdict == "PASS"
            assert c.ticker == c.ticker.upper()
