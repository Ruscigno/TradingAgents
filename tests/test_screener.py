"""Unit tests for TechnicalScreener and its indicator helpers.

All tests use mock MDS responses — no live network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tradingagents.dataflows.mds_client import MDSUnavailableError
from tradingagents.screener.technical_screener import (
    ScreenerConfig,
    ScreenerResult,
    TechnicalScreener,
    _compute_dist_sma,
    _compute_rsi,
    _compute_vol_osc,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_df(n: int = 100, close_val: float = 100.0, volume_val: float = 1_000_000.0) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame with constant close and volume."""
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": close_val,
            "High": close_val * 1.01,
            "Low": close_val * 0.99,
            "Close": close_val,
            "Volume": volume_val,
        },
        index=idx,
    )


def _make_trending_df(n: int = 100, start: float = 90.0, end: float = 110.0) -> pd.DataFrame:
    """Build a DataFrame with a linearly trending close price."""
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    close = np.linspace(start, end, n)
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1_000_000.0},
        index=idx,
    )


def _make_noisy_df(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Build a DataFrame with noisy close prices (mix of up and down days)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    close = 100.0 + (rng.standard_normal(n) * 0.5).cumsum()
    close = np.maximum(close, 1.0)  # keep positive
    return pd.DataFrame(
        {"Open": close, "High": close * 1.005, "Low": close * 0.995, "Close": close, "Volume": 1_000_000.0},
        index=idx,
    )


def _mock_client(df: pd.DataFrame | None = None, raises: Exception | None = None) -> MagicMock:
    client = MagicMock()
    if raises is not None:
        client.get_ohlcv.side_effect = raises
    else:
        client.get_ohlcv.return_value = df if df is not None else _make_df()
    return client


# ── RSI indicator ─────────────────────────────────────────────────────────────


class TestComputeRSI:
    def test_insufficient_data_returns_none(self):
        s = pd.Series([100.0] * 10)
        assert _compute_rsi(s, period=14) is None

    def test_flat_series_returns_100(self):
        # No losses → RS = ∞ → RSI = 100
        s = pd.Series([100.0] * 30)
        result = _compute_rsi(s, period=14)
        assert result == pytest.approx(100.0)

    def test_declining_series_returns_low_rsi(self):
        # Monotonically declining → no gains → RSI near 0
        s = pd.Series(range(100, 60, -1), dtype=float)
        result = _compute_rsi(s, period=14)
        assert result is not None
        assert result < 10

    def test_mixed_series_returns_mid_range(self):
        rng = np.random.default_rng(42)
        s = pd.Series(100 + rng.standard_normal(60).cumsum())
        result = _compute_rsi(s, period=14)
        assert result is not None
        assert 0 <= result <= 100

    def test_returns_float(self):
        s = pd.Series(range(50, 80, 1), dtype=float)
        result = _compute_rsi(s, period=14)
        assert isinstance(result, float)


# ── Volume Oscillator ─────────────────────────────────────────────────────────


class TestComputeVolOsc:
    def test_insufficient_data_returns_none(self):
        s = pd.Series([1_000_000.0] * 10)
        assert _compute_vol_osc(s, fast=5, slow=20) is None

    def test_constant_volume_near_zero(self):
        s = pd.Series([1_000_000.0] * 50)
        result = _compute_vol_osc(s, fast=5, slow=20)
        assert result is not None
        assert abs(result) < 0.01  # EMA fast ≈ EMA slow

    def test_rising_volume_positive(self):
        # Volume doubles over 30 bars → fast EMA > slow EMA
        s = pd.Series(np.linspace(1_000_000, 2_000_000, 50))
        result = _compute_vol_osc(s, fast=5, slow=20)
        assert result is not None
        assert result > 0

    def test_falling_volume_negative(self):
        s = pd.Series(np.linspace(2_000_000, 1_000_000, 50))
        result = _compute_vol_osc(s, fast=5, slow=20)
        assert result is not None
        assert result < 0

    def test_zero_slow_ema_returns_none(self):
        s = pd.Series([0.0] * 50)
        result = _compute_vol_osc(s, fast=5, slow=20)
        assert result is None


# ── Distance from SMA50 ───────────────────────────────────────────────────────


class TestComputeDistSMA:
    def test_insufficient_data_returns_none(self):
        s = pd.Series([100.0] * 30)
        assert _compute_dist_sma(s, 50) is None

    def test_flat_at_sma_returns_zero(self):
        s = pd.Series([100.0] * 60)
        result = _compute_dist_sma(s, 50)
        assert result is not None
        assert result == pytest.approx(0.0)

    def test_price_above_sma(self):
        # First 50 bars flat at 100, then spike to 110
        base = [100.0] * 50
        spike = [110.0] * 10
        s = pd.Series(base + spike)
        result = _compute_dist_sma(s, 50)
        assert result is not None
        assert result > 0

    def test_price_below_sma(self):
        base = [100.0] * 50
        drop = [90.0] * 10
        s = pd.Series(base + drop)
        result = _compute_dist_sma(s, 50)
        assert result is not None
        assert result < 0


# ── TechnicalScreener ─────────────────────────────────────────────────────────


class TestTechnicalScreenerPass:
    """Happy-path: a ticker passes all three filters."""

    def test_all_pass_returns_pass_result(self):
        # Use noisy data so RSI is in mid-range; widen thresholds to be robust
        df = _make_noisy_df(n=120)
        client = _mock_client(df=df)
        cfg = ScreenerConfig(rsi_min=0.0, rsi_max=100.0, vol_osc_min=-100.0, dist_ma_min=-100.0, dist_ma_max=100.0)
        screener = TechnicalScreener(mds_client=client, config=cfg)
        results = screener.screen(["AAPL"], "2025-06-15")
        assert len(results) == 1
        r = results[0]
        assert r.ticker == "AAPL"
        assert r.passed is True
        assert r.reason == "PASS"

    def test_passing_returns_only_tickers_that_pass(self):
        df = _make_noisy_df(n=120)
        client = _mock_client(df=df)
        cfg = ScreenerConfig(rsi_min=0.0, rsi_max=100.0, vol_osc_min=-100.0, dist_ma_min=-100.0, dist_ma_max=100.0)
        screener = TechnicalScreener(mds_client=client, config=cfg)
        passing = screener.passing(["AAPL"], "2025-06-15")
        assert "AAPL" in passing


class TestTechnicalScreenerFail:
    def test_empty_dataframe_fails(self):
        client = _mock_client(df=pd.DataFrame())
        screener = TechnicalScreener(mds_client=client)
        results = screener.screen(["TSLA"], "2025-06-15")
        assert not results[0].passed
        assert "No data" in results[0].reason

    def test_mds_unavailable_fails_gracefully(self):
        client = _mock_client(raises=MDSUnavailableError("timeout"))
        screener = TechnicalScreener(mds_client=client)
        results = screener.screen(["NVDA"], "2025-06-15")
        assert not results[0].passed
        assert "MDS unavailable" in results[0].reason

    def test_insufficient_bars_fails(self):
        df = _make_df(n=10)  # Way too few bars for SMA50
        client = _mock_client(df=df)
        screener = TechnicalScreener(mds_client=client)
        results = screener.screen(["MSFT"], "2025-06-15")
        assert not results[0].passed
        assert "Insufficient data" in results[0].reason

    def test_rsi_too_high_fails(self):
        # Monotonically upward trend → RSI = 100 (no down days)
        df = _make_trending_df(n=120, start=50.0, end=200.0)
        client = _mock_client(df=df)
        cfg = ScreenerConfig(rsi_min=50.0, rsi_max=90.0, vol_osc_min=-100.0, dist_ma_min=-100.0, dist_ma_max=100.0)
        screener = TechnicalScreener(mds_client=client, config=cfg)
        results = screener.screen(["GOOG"], "2025-06-15")
        assert not results[0].passed
        assert "RSI" in results[0].reason

    def test_vol_osc_too_low_fails(self):
        # Flat price (neutral RSI), falling volume
        df = _make_df(n=120)
        # Replace volume with declining trend
        df["Volume"] = np.linspace(2_000_000, 500_000, 120)
        client = _mock_client(df=df)
        cfg = ScreenerConfig(rsi_min=0.0, rsi_max=100.0, vol_osc_min=0.0, dist_ma_min=-100.0, dist_ma_max=100.0)
        screener = TechnicalScreener(mds_client=client, config=cfg)
        results = screener.screen(["META"], "2025-06-15")
        assert not results[0].passed
        assert "VolOsc" in results[0].reason

    def test_dist_ma_too_large_fails(self):
        # Price is flat for 60 days at 100, then permanently jumps to 130.
        # SMA50 at the end will still contain many 100-bars → close (130) is
        # well above SMA50, so DistMA > 5%.
        # 100 bars at 100, then 20 bars at 130 → SMA50 still contains ~30 bars
        # at 100 → SMA50 ≈ 112 → DistMA ≈ +16% → fails the ±5% threshold
        base = [100.0] * 100
        spike = [130.0] * 20
        idx = pd.date_range("2025-01-01", periods=120, freq="B")
        close = np.array(base + spike)
        df = pd.DataFrame(
            {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1_000_000.0},
            index=idx,
        )
        client = _mock_client(df=df)
        # Widen RSI/VolOsc so the only failing filter is DistMA
        cfg = ScreenerConfig(rsi_min=0.0, rsi_max=100.0, vol_osc_min=-100.0, dist_ma_min=-5.0, dist_ma_max=5.0)
        screener = TechnicalScreener(mds_client=client, config=cfg)
        results = screener.screen(["AMZN"], "2025-06-15")
        assert not results[0].passed
        assert "DistMA" in results[0].reason


class TestTechnicalScreenerMultipleTickers:
    def test_screens_all_tickers(self):
        df_pass = _make_noisy_df(n=120)
        df_fail = pd.DataFrame()  # will fail with "No data"
        call_count = [0]

        def get_ohlcv_side_effect(ticker, *args, **kwargs):
            call_count[0] += 1
            if ticker == "AAPL":
                return df_pass
            return df_fail

        client = MagicMock()
        client.get_ohlcv.side_effect = get_ohlcv_side_effect
        cfg = ScreenerConfig(rsi_min=0.0, rsi_max=100.0, vol_osc_min=-100.0, dist_ma_min=-100.0, dist_ma_max=100.0)
        screener = TechnicalScreener(mds_client=client, config=cfg)
        results = screener.screen(["AAPL", "TSLA", "GOOG"], "2025-06-15")
        assert len(results) == 3
        assert call_count[0] == 3

    def test_passing_filters_correctly(self):
        df_pass = _make_noisy_df(n=120)
        df_fail = pd.DataFrame()

        def get_ohlcv_side_effect(ticker, *args, **kwargs):
            return df_pass if ticker == "AAPL" else df_fail

        client = MagicMock()
        client.get_ohlcv.side_effect = get_ohlcv_side_effect
        cfg = ScreenerConfig(rsi_min=0.0, rsi_max=100.0, vol_osc_min=-100.0, dist_ma_min=-100.0, dist_ma_max=100.0)
        screener = TechnicalScreener(mds_client=client, config=cfg)
        passing = screener.passing(["AAPL", "TSLA"], "2025-06-15")
        assert passing == ["AAPL"]


class TestScreenerResultFields:
    def test_result_has_all_fields(self):
        r = ScreenerResult(ticker="AAPL", passed=True, rsi=55.0, vol_osc=1.5, dist_ma_pct=-1.2, reason="PASS")
        assert r.ticker == "AAPL"
        assert r.passed is True
        assert r.rsi == pytest.approx(55.0)
        assert r.vol_osc == pytest.approx(1.5)
        assert r.dist_ma_pct == pytest.approx(-1.2)
        assert r.reason == "PASS"

    def test_failed_result_has_reason(self):
        r = ScreenerResult(ticker="TSLA", passed=False, rsi=75.0, vol_osc=None, dist_ma_pct=None, reason="RSI 75.0 outside [50.0, 70.0]")
        assert not r.passed
        assert "RSI" in r.reason
