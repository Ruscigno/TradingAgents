"""Pre-flight technical screener for TradingAgents.

Filters all tickers tracked by MDS to a small candidate list using three
momentum/breakout indicators computed purely in pandas — no LLM calls.

Default thresholds target stocks in a bullish momentum zone:
  - RSI(14):             50 ≤ x ≤ 70
  - Volume Oscillator:   > 0   (fast EMA above slow EMA → rising volume)
  - Distance from SMA50: −5% to +5% (price near the 50d moving average)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from ..dataflows.mds_client import MDSClient, MDSUnavailableError


@dataclass
class ScreenerConfig:
    rsi_period: int = 14
    rsi_min: float = 50.0
    rsi_max: float = 70.0
    vol_osc_fast: int = 5        # fast EMA period for Volume Oscillator
    vol_osc_slow: int = 20       # slow EMA period for Volume Oscillator
    vol_osc_min: float = 0.0     # VO must be > vol_osc_min
    ma_period: int = 50
    dist_ma_min: float = -5.0    # % distance from MA (negative = below)
    dist_ma_max: float = 5.0
    lookback_days: int = 90      # calendar days of history to fetch


@dataclass
class ScreenerResult:
    ticker: str
    passed: bool
    rsi: float | None
    vol_osc: float | None
    dist_ma_pct: float | None
    reason: str  # "PASS" or which filter failed


class TechnicalScreener:
    """Filters tickers using RSI, Volume Oscillator, and Distance from SMA50.

    Args:
        mds_client: MDSClient instance to fetch OHLCV data from.
        config:     Threshold configuration. Defaults to momentum/breakout zone.
    """

    def __init__(
        self,
        mds_client: MDSClient,
        config: ScreenerConfig = ScreenerConfig(),
    ) -> None:
        self._client = mds_client
        self._config = config

    def screen(self, tickers: list[str], as_of_date: str) -> list[ScreenerResult]:
        """Screen a list of tickers and return all results (pass and fail).

        Args:
            tickers:    List of ticker symbols to screen.
            as_of_date: Date to screen as of, "YYYY-MM-DD". Data up to and
                        including this date is used.

        Returns:
            List of ScreenerResult for every ticker, in the same order as input.
        """
        as_of_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
        # Need enough history for SMA50 + RSI(14) to warm up
        history_needed = max(self._config.ma_period, self._config.rsi_period) + self._config.lookback_days
        fetch_start = as_of_dt - pd.Timedelta(days=history_needed)
        # end is exclusive in MDS API — add one day
        fetch_end = as_of_dt + pd.Timedelta(days=1)

        start_str = fetch_start.strftime("%Y-%m-%d")
        end_str = fetch_end.strftime("%Y-%m-%d")

        results: list[ScreenerResult] = []
        for ticker in tickers:
            result = self._screen_one(ticker, start_str, end_str, as_of_date)
            results.append(result)
        return results

    def passing(self, tickers: list[str], as_of_date: str) -> list[str]:
        """Convenience method: return only ticker symbols that passed all filters."""
        return [r.ticker for r in self.screen(tickers, as_of_date) if r.passed]

    # ── Private ───────────────────────────────────────────────────────────────

    def _screen_one(
        self,
        ticker: str,
        fetch_start: str,
        fetch_end: str,
        as_of_date: str,
    ) -> ScreenerResult:
        try:
            df = self._client.get_ohlcv(ticker, fetch_start, fetch_end, resolution="1d")
        except MDSUnavailableError as exc:
            return ScreenerResult(
                ticker=ticker,
                passed=False,
                rsi=None,
                vol_osc=None,
                dist_ma_pct=None,
                reason=f"MDS unavailable: {exc}",
            )

        if df.empty:
            return ScreenerResult(
                ticker=ticker,
                passed=False,
                rsi=None,
                vol_osc=None,
                dist_ma_pct=None,
                reason="No data",
            )

        # Trim to as_of_date (inclusive) — drop tz for comparison simplicity
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df[df.index <= as_of_date]

        if len(df) < self._config.ma_period:
            return ScreenerResult(
                ticker=ticker,
                passed=False,
                rsi=None,
                vol_osc=None,
                dist_ma_pct=None,
                reason=f"Insufficient data ({len(df)} bars, need {self._config.ma_period})",
            )

        rsi_val = _compute_rsi(df["Close"], self._config.rsi_period)
        vol_osc_val = _compute_vol_osc(df["Volume"], self._config.vol_osc_fast, self._config.vol_osc_slow)
        dist_ma_val = _compute_dist_sma(df["Close"], self._config.ma_period)

        cfg = self._config

        if rsi_val is None:
            return ScreenerResult(ticker=ticker, passed=False, rsi=None, vol_osc=vol_osc_val, dist_ma_pct=dist_ma_val, reason="RSI could not be computed")
        if not (cfg.rsi_min <= rsi_val <= cfg.rsi_max):
            return ScreenerResult(ticker=ticker, passed=False, rsi=rsi_val, vol_osc=vol_osc_val, dist_ma_pct=dist_ma_val, reason=f"RSI {rsi_val:.1f} outside [{cfg.rsi_min}, {cfg.rsi_max}]")

        if vol_osc_val is None:
            return ScreenerResult(ticker=ticker, passed=False, rsi=rsi_val, vol_osc=None, dist_ma_pct=dist_ma_val, reason="VolOsc could not be computed")
        if vol_osc_val <= cfg.vol_osc_min:
            return ScreenerResult(ticker=ticker, passed=False, rsi=rsi_val, vol_osc=vol_osc_val, dist_ma_pct=dist_ma_val, reason=f"VolOsc {vol_osc_val:.2f}% ≤ {cfg.vol_osc_min}")

        if dist_ma_val is None:
            return ScreenerResult(ticker=ticker, passed=False, rsi=rsi_val, vol_osc=vol_osc_val, dist_ma_pct=None, reason="DistMA could not be computed")
        if not (cfg.dist_ma_min <= dist_ma_val <= cfg.dist_ma_max):
            return ScreenerResult(ticker=ticker, passed=False, rsi=rsi_val, vol_osc=vol_osc_val, dist_ma_pct=dist_ma_val, reason=f"DistMA {dist_ma_val:.1f}% outside [{cfg.dist_ma_min}%, {cfg.dist_ma_max}%]")

        return ScreenerResult(ticker=ticker, passed=True, rsi=rsi_val, vol_osc=vol_osc_val, dist_ma_pct=dist_ma_val, reason="PASS")


# ── Indicator helpers (pure pandas, no extra deps) ────────────────────────────


def _compute_rsi(close: pd.Series, period: int) -> float | None:
    """Wilder's RSI for the most recent bar."""
    if len(close) < period + 1:
        return None
    delta = close.diff().dropna()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    # Wilder smoothing: SMA for first window, then EWMA with alpha=1/period
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    last_gain = avg_gain.iloc[-1]
    last_loss = avg_loss.iloc[-1]
    if last_loss == 0:
        return 100.0
    rs = last_gain / last_loss
    return float(100 - (100 / (1 + rs)))


def _compute_vol_osc(volume: pd.Series, fast: int, slow: int) -> float | None:
    """Volume Oscillator = (EMA_fast - EMA_slow) / EMA_slow * 100."""
    if len(volume) < slow:
        return None
    ema_fast = volume.ewm(span=fast, adjust=False).mean()
    ema_slow = volume.ewm(span=slow, adjust=False).mean()
    last_slow = ema_slow.iloc[-1]
    if last_slow == 0:
        return None
    return float((ema_fast.iloc[-1] - last_slow) / last_slow * 100)


def _compute_dist_sma(close: pd.Series, period: int) -> float | None:
    """Distance from SMA(period) = (Close - SMA) / SMA * 100."""
    if len(close) < period:
        return None
    sma = close.rolling(period).mean()
    last_sma = sma.iloc[-1]
    if pd.isna(last_sma) or last_sma == 0:
        return None
    return float((close.iloc[-1] - last_sma) / last_sma * 100)
