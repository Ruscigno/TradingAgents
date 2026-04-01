"""Market Data Service vendor implementation for TradingAgents.

Implements the same function contract as y_finance.py so that interface.py
can route ``get_stock_data`` and ``get_indicators`` calls to MDS instead of
fetching directly from Yahoo Finance.

MDS provides:
  - OHLCV data for 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo timeframes
  - All tickers in the configured stocks.txt list

MDS does NOT provide fundamentals, news, or insider transactions — those
categories remain routed to yfinance in interface.py.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Annotated, Any

import pandas as pd
from dateutil.relativedelta import relativedelta
from stockstats import wrap

from .mds_client import MDSUnavailableError, get_mds_client
from .stockstats_utils import _clean_dataframe


# ── OHLCV data ────────────────────────────────────────────────────────────────


def get_MDS_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Fetch OHLCV data from the Market Data Service and return as a CSV string.

    Mirrors the return format of ``get_YFin_data_online`` so the LLM agents
    receive identically structured text regardless of the underlying vendor.

    Raises:
        MDSUnavailableError: Propagated to trigger yfinance fallback in
                             ``route_to_vendor``.
    """
    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    client = get_mds_client()
    df = client.get_ohlcv(symbol, start_date, end_date, resolution="1d")

    if df.empty:
        return f"No data found for symbol '{symbol}' between {start_date} and {end_date}"

    # Drop tz info for cleaner display (matches yfinance output)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Round numeric columns to 2 dp (matches yfinance output)
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = df[col].round(2)

    csv_string = df.to_csv()

    header = f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(df)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    header += "# Source: Market Data Service (InfluxDB cache)\n\n"

    return header + csv_string


# ── Technical indicators ───────────────────────────────────────────────────────


def get_MDS_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator name (e.g. 'rsi', 'macd')"],
    curr_date: Annotated[str, "The current trading date, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    """Compute a technical indicator from MDS OHLCV data.

    Fetches raw OHLCV from MDS, then computes the indicator using stockstats —
    identical computation to the yfinance path, different data source.

    Mirrors the return format of ``get_stock_stats_indicators_window``.

    Raises:
        MDSUnavailableError: Propagated to trigger yfinance fallback.
        ValueError: If the indicator name is not supported.
    """
    # Reuse the exact indicator metadata from y_finance.py
    best_ind_params = _INDICATOR_DESCRIPTIONS

    if indicator not in best_ind_params:
        raise ValueError(
            f"Indicator '{indicator}' is not supported. "
            f"Choose from: {list(best_ind_params.keys())}"
        )

    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - relativedelta(days=look_back_days)

    # Fetch enough history for the indicator to warm up (add extra buffer)
    fetch_start = curr_date_dt - relativedelta(days=look_back_days + 250)
    fetch_end = curr_date_dt + relativedelta(days=1)  # exclusive

    client = get_mds_client()
    df_raw = client.get_ohlcv(
        symbol,
        fetch_start.strftime("%Y-%m-%d"),
        fetch_end.strftime("%Y-%m-%d"),
        resolution="1d",
    )

    if df_raw.empty:
        raise MDSUnavailableError(
            f"No data from MDS for '{symbol}' in range {fetch_start.date()} – {fetch_end.date()}"
        )

    # Normalise index for stockstats
    df_raw = df_raw.copy()
    if df_raw.index.tz is not None:
        df_raw.index = df_raw.index.tz_localize(None)
    df_raw = df_raw.reset_index().rename(columns={"time": "Date", "index": "Date"})

    df_clean = _clean_dataframe(df_raw)
    df_ss = wrap(df_clean)
    df_ss["Date"] = df_ss["Date"].dt.strftime("%Y-%m-%d")

    # Trigger stockstats to compute the indicator
    df_ss[indicator]  # noqa: B018

    # Build date → value dict
    indicator_data: dict[str, Any] = {}
    for _, row in df_ss.iterrows():
        val = row[indicator]
        indicator_data[row["Date"]] = "N/A" if pd.isna(val) else str(val)

    # Build result string (same format as get_stock_stats_indicators_window)
    ind_string = ""
    current = curr_date_dt
    while current >= before:
        date_str = current.strftime("%Y-%m-%d")
        val = indicator_data.get(date_str, "N/A: Not a trading day (weekend or holiday)")
        ind_string += f"{date_str}: {val}\n"
        current -= relativedelta(days=1)

    return (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {curr_date}:\n\n"
        + ind_string
        + "\n\n"
        + best_ind_params.get(indicator, "No description available.")
        + "\n# Source: Market Data Service (InfluxDB cache)"
    )


# ── Indicator descriptions (copied from y_finance.py to keep parity) ─────────

_INDICATOR_DESCRIPTIONS = {
    "close_50_sma": (
        "50 SMA: A medium-term trend indicator. "
        "Usage: Identify trend direction and serve as dynamic support/resistance. "
        "Tips: It lags price; combine with faster indicators for timely signals."
    ),
    "close_200_sma": (
        "200 SMA: A long-term trend benchmark. "
        "Usage: Confirm overall market trend and identify golden/death cross setups. "
        "Tips: It reacts slowly; best for strategic trend confirmation."
    ),
    "close_10_ema": (
        "10 EMA: A responsive short-term average. "
        "Usage: Capture quick shifts in momentum and potential entry points. "
        "Tips: Prone to noise in choppy markets; use alongside longer averages."
    ),
    "macd": (
        "MACD: Computes momentum via differences of EMAs. "
        "Usage: Look for crossovers and divergence as signals of trend changes. "
        "Tips: Confirm with other indicators in low-volatility or sideways markets."
    ),
    "macds": (
        "MACD Signal: An EMA smoothing of the MACD line. "
        "Usage: Use crossovers with the MACD line to trigger trades. "
        "Tips: Should be part of a broader strategy to avoid false positives."
    ),
    "macdh": (
        "MACD Histogram: Shows the gap between the MACD line and its signal. "
        "Usage: Visualize momentum strength and spot divergence early. "
        "Tips: Can be volatile; complement with additional filters."
    ),
    "rsi": (
        "RSI: Measures momentum to flag overbought/oversold conditions. "
        "Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. "
        "Tips: In strong trends, RSI may remain extreme; always cross-check with trend."
    ),
    "boll": (
        "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. "
        "Usage: Acts as a dynamic benchmark for price movement. "
        "Tips: Combine with upper/lower bands to spot breakouts or reversals."
    ),
    "boll_ub": (
        "Bollinger Upper Band: Typically 2 standard deviations above the middle. "
        "Usage: Signals potential overbought conditions and breakout zones. "
        "Tips: Prices may ride the band in strong trends."
    ),
    "boll_lb": (
        "Bollinger Lower Band: Typically 2 standard deviations below the middle. "
        "Usage: Indicates potential oversold conditions. "
        "Tips: Use additional analysis to avoid false reversal signals."
    ),
    "atr": (
        "ATR: Averages true range to measure volatility. "
        "Usage: Set stop-loss levels and adjust position sizes. "
        "Tips: It's reactive; use as part of a broader risk management strategy."
    ),
    "vwma": (
        "VWMA: A moving average weighted by volume. "
        "Usage: Confirm trends by integrating price action with volume data. "
        "Tips: Watch for skewed results from volume spikes."
    ),
    "mfi": (
        "MFI: Money Flow Index — momentum using both price and volume. "
        "Usage: Identify overbought (>80) or oversold (<20) conditions. "
        "Tips: Use alongside RSI or MACD to confirm signals."
    ),
}
