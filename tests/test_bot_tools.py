"""Unit tests for tradingagents.bot — MCP tools, cache, and scheduler.

All tests use mocks for MDS, LLM, and filesystem — no live network calls.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tradingagents.bot.results_cache import ResultsCache
from tradingagents.bot.analysis_tools import (
    check_status,
    get_last_result,
    get_rejected,
    run_screener,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_noisy_df(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Build a DataFrame with noisy close prices (mix of up and down days)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    close = 100.0 + (rng.standard_normal(n) * 0.5).cumsum()
    close = np.maximum(close, 1.0)
    return pd.DataFrame(
        {"Open": close, "High": close * 1.005, "Low": close * 0.995, "Close": close, "Volume": 1_000_000.0},
        index=idx,
    )


# ── ResultsCache ─────────────────────────────────────────────────────────────


class TestResultsCache:
    def test_save_and_load_round_trip(self, tmp_path):
        cache = ResultsCache(cache_dir=tmp_path)
        data = {"date": "2026-04-03", "passed_count": 5}
        cache.save("screen", data)

        loaded = cache.load("screen")
        assert loaded is not None
        assert loaded["date"] == "2026-04-03"
        assert loaded["passed_count"] == 5
        assert "saved_at" in loaded

    def test_load_missing_returns_none(self, tmp_path):
        cache = ResultsCache(cache_dir=tmp_path)
        assert cache.load("analyze") is None

    def test_save_and_load_schedules(self, tmp_path):
        cache = ResultsCache(cache_dir=tmp_path)
        schedules = [
            {"id": "sched_001", "recurrence_hours": 24},
            {"id": "sched_002", "recurrence_hours": 168},
        ]
        cache.save_schedules(schedules)

        loaded = cache.load_schedules()
        assert len(loaded) == 2
        assert loaded[0]["id"] == "sched_001"

    def test_load_schedules_empty_returns_list(self, tmp_path):
        cache = ResultsCache(cache_dir=tmp_path)
        assert cache.load_schedules() == []

    def test_save_and_load_chat_id(self, tmp_path):
        cache = ResultsCache(cache_dir=tmp_path)
        cache.save_chat_id(123456789)
        assert cache.load_chat_id() == 123456789

    def test_load_chat_id_missing_returns_none(self, tmp_path):
        cache = ResultsCache(cache_dir=tmp_path)
        assert cache.load_chat_id() is None

    def test_save_overwrites_previous(self, tmp_path):
        cache = ResultsCache(cache_dir=tmp_path)
        cache.save("screen", {"date": "2026-04-01", "count": 1})
        cache.save("screen", {"date": "2026-04-02", "count": 2})

        loaded = cache.load("screen")
        assert loaded["date"] == "2026-04-02"
        assert loaded["count"] == 2

    def test_corrupt_json_returns_none(self, tmp_path):
        cache = ResultsCache(cache_dir=tmp_path)
        (tmp_path / "last_screen.json").write_text("{broken json")
        assert cache.load("screen") is None


# ── run_screener ──────────────────────────────────────────────────────────────


class TestRunScreener:
    @patch("tradingagents.bot.analysis_tools.ScreenerConfigLoader")
    @patch("tradingagents.bot.analysis_tools._cache")
    @patch("tradingagents.bot.analysis_tools.MDSClient")
    def test_screener_returns_passed_and_failed(self, mock_mds_cls, mock_cache, mock_loader_cls):
        df = _make_noisy_df(n=200)
        mock_client = MagicMock()
        mock_client.get_tickers.return_value = ["AAPL", "NVDA"]
        mock_client.get_ohlcv.return_value = df
        mock_mds_cls.return_value = mock_client

        # Mock the config loader so screener.yaml defaults don't override the
        # wide thresholds — each ticker gets a permissive ScreenerConfig
        from tradingagents.screener.technical_screener import ScreenerConfig
        wide_cfg = ScreenerConfig(rsi_min=0.0, rsi_max=100.0, vol_osc_min=-100.0, dist_ma_min=-100.0, dist_ma_max=100.0)
        mock_loader = MagicMock()
        mock_loader.get_config.return_value = wide_cfg
        mock_loader_cls.from_yaml.return_value = mock_loader

        # Wide thresholds so everything passes
        result = run_screener(
            date="2025-09-15",
            rsi_min=0.0,
            rsi_max=100.0,
            vol_osc_min=-100.0,
            dist_ma_min=-100.0,
            dist_ma_max=100.0,
        )

        assert result["total"] == 2
        assert result["passed_count"] == 2
        assert result["date"] == "2025-09-15"

    @patch("tradingagents.bot.analysis_tools._cache")
    @patch("tradingagents.bot.analysis_tools.MDSClient")
    def test_screener_with_explicit_tickers(self, mock_mds_cls, mock_cache):
        df = _make_noisy_df(n=120)
        mock_client = MagicMock()
        mock_client.get_ohlcv.return_value = df
        mock_mds_cls.return_value = mock_client

        result = run_screener(
            date="2025-06-15",
            tickers=["AAPL"],
            rsi_min=0.0,
            rsi_max=100.0,
            vol_osc_min=-100.0,
            dist_ma_min=-100.0,
            dist_ma_max=100.0,
        )

        assert result["total"] == 1
        # get_tickers should NOT have been called since we specified tickers
        mock_client.get_tickers.assert_not_called()

    @patch("tradingagents.bot.analysis_tools._cache")
    @patch("tradingagents.bot.analysis_tools.MDSClient")
    def test_screener_mds_down_returns_error(self, mock_mds_cls, mock_cache):
        from tradingagents.dataflows.mds_client import MDSUnavailableError

        mock_client = MagicMock()
        mock_client.get_tickers.side_effect = MDSUnavailableError("timeout")
        mock_mds_cls.return_value = mock_client

        result = run_screener(date="2025-06-15")
        assert "error" in result

    @patch("tradingagents.bot.analysis_tools._cache")
    @patch("tradingagents.bot.analysis_tools.MDSClient")
    def test_screener_saves_to_cache(self, mock_mds_cls, mock_cache):
        df = _make_noisy_df(n=120)
        mock_client = MagicMock()
        mock_client.get_tickers.return_value = ["AAPL"]
        mock_client.get_ohlcv.return_value = df
        mock_mds_cls.return_value = mock_client

        run_screener(
            date="2025-06-15",
            rsi_min=0.0, rsi_max=100.0,
            vol_osc_min=-100.0, dist_ma_min=-100.0, dist_ma_max=100.0,
        )

        mock_cache.save.assert_called_once()
        call_args = mock_cache.save.call_args
        assert call_args[0][0] == "screen"


# ── get_rejected ──────────────────────────────────────────────────────────────


class TestGetRejected:
    @patch("tradingagents.bot.analysis_tools._cache")
    def test_uses_cached_screen(self, mock_cache):
        mock_cache.load.return_value = {
            "date": "2025-06-15",
            "failed": [
                {"ticker": "NVDA", "rsi": 38.0, "reason": "RSI 38.0 outside [50.0, 70.0]"},
            ],
        }

        result = get_rejected(date="2025-06-15")
        assert result["source"] == "cached screen"
        assert len(result["rejected"]) == 1
        assert result["rejected"][0]["ticker"] == "NVDA"

    @patch("tradingagents.bot.analysis_tools.run_screener")
    @patch("tradingagents.bot.analysis_tools._cache")
    def test_runs_fresh_screen_when_no_cache(self, mock_cache, mock_run_screener):
        mock_cache.load.return_value = None
        mock_run_screener.return_value = {
            "failed_count": 1,
            "failed": [{"ticker": "NVDA", "reason": "RSI too low"}],
        }

        result = get_rejected(date="2025-06-15")
        assert result["source"] == "fresh screen"
        mock_run_screener.assert_called_once()


# ── check_status ──────────────────────────────────────────────────────────────


class TestCheckStatus:
    @patch("requests.get")
    def test_returns_status_dict(self, mock_get):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"tickers": [{"symbol": "AAPL"}]}
        mock_get.return_value = mock_response

        result = check_status()
        assert "mds" in result
        assert result["mds"]["status"] == "healthy"

    @patch("requests.get")
    def test_mds_unreachable(self, mock_get):
        import requests as real_requests
        mock_get.side_effect = real_requests.exceptions.ConnectionError("refused")

        result = check_status()
        assert result["mds"]["status"] == "unreachable"


# ── get_last_result ───────────────────────────────────────────────────────────


class TestGetLastResult:
    @patch("tradingagents.bot.analysis_tools._cache")
    def test_returns_cached_analyze(self, mock_cache):
        mock_cache.load.return_value = {"date": "2026-04-03", "trade_decisions": []}
        result = get_last_result(command="analyze")
        assert result["date"] == "2026-04-03"

    @patch("tradingagents.bot.analysis_tools._cache")
    def test_returns_error_when_no_cache(self, mock_cache):
        mock_cache.load.return_value = None
        result = get_last_result(command="analyze")
        assert "error" in result

    @patch("tradingagents.bot.analysis_tools._cache")
    def test_returns_summary_when_no_command(self, mock_cache):
        def load_side_effect(cmd):
            if cmd == "analyze":
                return {"saved_at": "2026-04-03T10:00:00Z", "date": "2026-04-03"}
            return None

        mock_cache.load.side_effect = load_side_effect

        result = get_last_result(command=None)
        assert "cached_results" in result
        assert result["cached_results"]["analyze"] is not None
        assert result["cached_results"]["screen"] is None


# ── Scheduler ─────────────────────────────────────────────────────────────────


class TestTradingScheduler:
    def test_create_and_list_schedule(self):
        from tradingagents.bot.scheduler import TradingScheduler

        sched = TradingScheduler()
        # Patch _persist_schedules to avoid filesystem
        sched._persist_schedules = MagicMock()
        sched.start()

        try:
            sid = sched.create_schedule(
                recurrence_hours=24,
                start_datetime="2026-04-10T09:00",
                params={"rsi_min": 45.0},
            )

            assert sid.startswith("sched_")
            schedules = sched.list_schedules()
            assert len(schedules) == 1
            assert schedules[0]["id"] == sid
            assert schedules[0]["recurrence_hours"] == 24
        finally:
            sched.shutdown()

    def test_delete_schedule(self):
        from tradingagents.bot.scheduler import TradingScheduler

        sched = TradingScheduler()
        sched._persist_schedules = MagicMock()
        sched.start()

        try:
            sid = sched.create_schedule(recurrence_hours=24)

            assert sched.delete_schedule(sid) is True
            assert sched.delete_schedule(sid) is False  # already deleted
            assert sched.list_schedules() == []
        finally:
            sched.shutdown()

    def test_delete_nonexistent_returns_false(self):
        from tradingagents.bot.scheduler import TradingScheduler

        sched = TradingScheduler()
        sched._persist_schedules = MagicMock()
        sched.start()

        try:
            assert sched.delete_schedule("sched_999") is False
        finally:
            sched.shutdown()
