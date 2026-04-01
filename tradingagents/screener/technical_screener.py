"""Pre-flight technical screener for TradingAgents.

Filters all tickers tracked by MDS to a small candidate list using three
momentum/breakout indicators computed purely in pandas — no LLM calls.

Default thresholds target stocks in a bullish momentum zone:
  - RSI(14):             50 ≤ x ≤ 70
  - Volume Oscillator:   > 0   (fast EMA above slow EMA → rising volume)
  - Distance from SMA50: −5% to +5% (price near the 50d moving average)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from ..dataflows.mds_client import MDSClient, MDSUnavailableError

if TYPE_CHECKING:
    from .config_loader import ScreenerConfigLoader

logger = logging.getLogger(__name__)


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
        mds_client:    MDSClient instance to fetch OHLCV data from.
        config:        Default threshold configuration (momentum/breakout zone).
        config_loader: Optional per-ticker config loader. When set,
                       ``screen()`` resolves a ScreenerConfig for each ticker
                       via the loader (merging YAML overrides onto ``config``).
    """

    def __init__(
        self,
        mds_client: MDSClient,
        config: ScreenerConfig = ScreenerConfig(),
        config_loader: ScreenerConfigLoader | None = None,
    ) -> None:
        self._client = mds_client
        self._config = config
        self._config_loader = config_loader

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

        logger.debug(json.dumps({
            "stage": "screening_start",
            "tickers": len(tickers),
            "as_of_date": as_of_date,
            "fetch_start": start_str,
            "fetch_end": end_str,
        }))

        results: list[ScreenerResult] = []
        for ticker in tickers:
            cfg = self._config_loader.get_config(ticker) if self._config_loader else self._config
            result = self._screen_one(ticker, start_str, end_str, as_of_date, cfg)
            results.append(result)

        passed = sum(1 for r in results if r.passed)
        logger.debug(json.dumps({
            "stage": "screening_done",
            "total": len(tickers),
            "passed": passed,
            "failed": len(tickers) - passed,
        }))

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
        cfg: ScreenerConfig,
    ) -> ScreenerResult:
        logger.debug(json.dumps({
            "ticker": ticker,
            "stage": "fetch",
            "start": fetch_start,
            "end": fetch_end,
            "rsi_threshold": [cfg.rsi_min, cfg.rsi_max],
            "vol_osc_min": cfg.vol_osc_min,
            "dist_ma_range": [cfg.dist_ma_min, cfg.dist_ma_max],
        }))

        try:
            df = self._client.get_ohlcv(ticker, fetch_start, fetch_end, resolution="1d")
        except MDSUnavailableError as exc:
            result = ScreenerResult(
                ticker=ticker, passed=False, rsi=None, vol_osc=None,
                dist_ma_pct=None, reason=f"MDS unavailable: {exc}",
            )
            _log_result(result)
            return result

        if df.empty:
            result = ScreenerResult(
                ticker=ticker, passed=False, rsi=None, vol_osc=None,
                dist_ma_pct=None, reason="No data",
            )
            _log_result(result)
            return result

        # Trim to as_of_date (inclusive) — drop tz for comparison simplicity
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df[df.index <= as_of_date]

        logger.debug(json.dumps({"ticker": ticker, "stage": "data_trimmed", "bars": len(df)}))

        if len(df) < cfg.ma_period:
            result = ScreenerResult(
                ticker=ticker, passed=False, rsi=None, vol_osc=None,
                dist_ma_pct=None,
                reason=f"Insufficient data ({len(df)} bars, need {cfg.ma_period})",
            )
            _log_result(result)
            return result

        rsi_val = _compute_rsi(df["Close"], cfg.rsi_period)
        vol_osc_val = _compute_vol_osc(df["Volume"], cfg.vol_osc_fast, cfg.vol_osc_slow)
        dist_ma_val = _compute_dist_sma(df["Close"], cfg.ma_period)

        logger.debug(json.dumps({
            "ticker": ticker, "stage": "rsi",
            "value": round(rsi_val, 2) if rsi_val is not None else None,
            "min": cfg.rsi_min, "max": cfg.rsi_max,
            "passed": rsi_val is not None and cfg.rsi_min <= rsi_val <= cfg.rsi_max,
        }))
        logger.debug(json.dumps({
            "ticker": ticker, "stage": "vol_osc",
            "value": round(vol_osc_val, 2) if vol_osc_val is not None else None,
            "min": cfg.vol_osc_min,
            "passed": vol_osc_val is not None and vol_osc_val > cfg.vol_osc_min,
        }))
        logger.debug(json.dumps({
            "ticker": ticker, "stage": "dist_ma",
            "value": round(dist_ma_val, 2) if dist_ma_val is not None else None,
            "min": cfg.dist_ma_min, "max": cfg.dist_ma_max,
            "passed": dist_ma_val is not None and cfg.dist_ma_min <= dist_ma_val <= cfg.dist_ma_max,
        }))

        if rsi_val is None:
            result = ScreenerResult(ticker=ticker, passed=False, rsi=None, vol_osc=vol_osc_val, dist_ma_pct=dist_ma_val, reason="RSI could not be computed")
        elif not (cfg.rsi_min <= rsi_val <= cfg.rsi_max):
            result = ScreenerResult(ticker=ticker, passed=False, rsi=rsi_val, vol_osc=vol_osc_val, dist_ma_pct=dist_ma_val, reason=f"RSI {rsi_val:.1f} outside [{cfg.rsi_min}, {cfg.rsi_max}]")
        elif vol_osc_val is None:
            result = ScreenerResult(ticker=ticker, passed=False, rsi=rsi_val, vol_osc=None, dist_ma_pct=dist_ma_val, reason="VolOsc could not be computed")
        elif vol_osc_val <= cfg.vol_osc_min:
            result = ScreenerResult(ticker=ticker, passed=False, rsi=rsi_val, vol_osc=vol_osc_val, dist_ma_pct=dist_ma_val, reason=f"VolOsc {vol_osc_val:.2f}% ≤ {cfg.vol_osc_min}")
        elif dist_ma_val is None:
            result = ScreenerResult(ticker=ticker, passed=False, rsi=rsi_val, vol_osc=vol_osc_val, dist_ma_pct=None, reason="DistMA could not be computed")
        elif not (cfg.dist_ma_min <= dist_ma_val <= cfg.dist_ma_max):
            result = ScreenerResult(ticker=ticker, passed=False, rsi=rsi_val, vol_osc=vol_osc_val, dist_ma_pct=dist_ma_val, reason=f"DistMA {dist_ma_val:.1f}% outside [{cfg.dist_ma_min}%, {cfg.dist_ma_max}%]")
        else:
            result = ScreenerResult(ticker=ticker, passed=True, rsi=rsi_val, vol_osc=vol_osc_val, dist_ma_pct=dist_ma_val, reason="PASS")

        _log_result(result)
        return result


def _log_result(result: ScreenerResult) -> None:
    logger.info(json.dumps({
        "ticker": result.ticker,
        "passed": result.passed,
        "rsi": round(result.rsi, 2) if result.rsi is not None else None,
        "vol_osc": round(result.vol_osc, 2) if result.vol_osc is not None else None,
        "dist_ma_pct": round(result.dist_ma_pct, 2) if result.dist_ma_pct is not None else None,
        "reason": result.reason,
    }))


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
