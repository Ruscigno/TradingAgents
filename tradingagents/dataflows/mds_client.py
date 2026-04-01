"""HTTP client for the Market Data Service REST API.

The Market Data Service (MDS) is a local InfluxDB-backed cache of OHLCV data
for a configured list of tickers. It exposes a REST API at (default)
http://localhost:8080.

This module provides:
  - MDSClient: low-level HTTP wrapper around the MDS REST API
  - MDSUnavailableError: raised when MDS is unreachable or returns unexpected errors
  - get_mds_client(): singleton factory — reuse across vendor calls in one process
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pandas as pd
import requests

if TYPE_CHECKING:
    pass


class MDSUnavailableError(Exception):
    """Raised when the Market Data Service is unreachable or returns an error.

    ``route_to_vendor`` in interface.py catches this and falls back to yfinance,
    so a downed MDS is gracefully handled without crashing the pipeline.
    """


class MDSClient:
    """Thin HTTP client for the Market Data Service REST API.

    Args:
        base_url: Base URL of the MDS instance, e.g. "http://localhost:8080".
                  Defaults to env var ``MDS_BASE_URL`` or "http://localhost:8080".
        timeout:  Per-request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 15.0,
    ) -> None:
        self._base_url = (
            base_url
            or os.environ.get("MDS_BASE_URL", "http://localhost:8080")
        ).rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_ohlcv(
        self,
        ticker: str,
        start: str,
        end: str,
        resolution: str = "1d",
        limit: int = 5000,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars from MDS.

        Args:
            ticker:     Stock ticker symbol (e.g. "AAPL").
            start:      ISO-8601 datetime string, UTC (e.g. "2024-01-01T00:00:00Z").
                        Also accepts "YYYY-MM-DD" — will be interpreted as UTC midnight.
            end:        ISO-8601 end datetime string (exclusive). Same format as ``start``.
            resolution: Timeframe: "1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo".
            limit:      Maximum number of bars to return (capped at 5000 by MDS).

        Returns:
            DataFrame with DatetimeIndex (UTC-aware), columns:
            Open, High, Low, Close, Volume. Empty if no data.

        Raises:
            MDSUnavailableError: On connection error, 404, or any unexpected HTTP error.
        """
        start_iso = _to_iso(start)
        end_iso = _to_iso(end)

        try:
            resp = self._session.get(
                f"{self._base_url}/api/v1/ohlc/{ticker.upper()}",
                params={
                    "resolution": resolution,
                    "start": start_iso,
                    "end": end_iso,
                    "limit": limit,
                },
                timeout=self._timeout,
            )
        except requests.exceptions.ConnectionError as exc:
            raise MDSUnavailableError(
                f"Cannot connect to Market Data Service at {self._base_url}: {exc}"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise MDSUnavailableError(
                f"Market Data Service timed out after {self._timeout}s: {exc}"
            ) from exc

        if resp.status_code == 404:
            # Ticker not tracked — return empty (not an error, yfinance will fallback)
            raise MDSUnavailableError(
                f"Ticker '{ticker}' not found in Market Data Service"
            )
        if not resp.ok:
            raise MDSUnavailableError(
                f"Market Data Service returned HTTP {resp.status_code} for {ticker}"
            )

        data = resp.json()
        bars = data.get("bars", [])
        if not bars:
            return pd.DataFrame()

        df = pd.DataFrame(bars)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time").sort_index()
        # Capitalise column names to match yfinance convention: Open/High/Low/Close/Volume
        df.columns = [c.capitalize() for c in df.columns]
        return df

    def get_tickers(self) -> list[str]:
        """Return all ticker symbols currently tracked by MDS.

        Raises:
            MDSUnavailableError: On connection error or unexpected HTTP error.
        """
        try:
            resp = self._session.get(
                f"{self._base_url}/api/v1/tickers",
                timeout=self._timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise MDSUnavailableError(
                f"Cannot connect to Market Data Service at {self._base_url}: {exc}"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise MDSUnavailableError(
                f"Market Data Service timed out after {self._timeout}s: {exc}"
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise MDSUnavailableError(f"Market Data Service error: {exc}") from exc

        return [t["symbol"] for t in resp.json().get("tickers", [])]

    def is_healthy(self) -> bool:
        """Return True if MDS is reachable and ready."""
        try:
            resp = self._session.get(
                f"{self._base_url}/ready",
                timeout=5.0,
            )
            return resp.ok
        except Exception:
            return False


# ── Singleton ─────────────────────────────────────────────────────────────────

_mds_client: MDSClient | None = None


def get_mds_client(base_url: str | None = None) -> MDSClient:
    """Return the process-wide MDSClient singleton.

    The singleton is created on first call. ``base_url`` is only used on
    the first call; subsequent calls ignore it and return the cached instance.
    To reconfigure (e.g. in tests), call ``reset_mds_client()`` first.
    """
    global _mds_client
    if _mds_client is None:
        _mds_client = MDSClient(base_url=base_url)
    return _mds_client


def reset_mds_client() -> None:
    """Reset the singleton. For use in tests only."""
    global _mds_client
    _mds_client = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_iso(dt_str: str) -> str:
    """Normalise a date/datetime string to ISO-8601 with UTC suffix.

    Accepts:
      - "YYYY-MM-DD"              → "YYYY-MM-DDT00:00:00Z"
      - "YYYY-MM-DDTHH:MM:SSZ"   → unchanged
      - "YYYY-MM-DDTHH:MM:SS+00:00" → unchanged
    """
    if len(dt_str) == 10:
        # Plain date — treat as UTC midnight
        return f"{dt_str}T00:00:00Z"
    if not dt_str.endswith("Z") and "+" not in dt_str[-6:]:
        return dt_str + "Z"
    return dt_str
