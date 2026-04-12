"""Pure Python functions called by the MCP server tools.

These functions wrap existing TradingAgents modules and are the sole
interface between the MCP layer and the pipeline. They are designed to be
called synchronously and to return plain dicts suitable for JSON
serialisation.

Each function catches exceptions per-ticker and returns error strings
rather than raising (per CLAUDE.md rule 6).
"""

from __future__ import annotations

import json
import logging
import os
import requests
from datetime import datetime
from pathlib import Path
from typing import Any

from tradingagents.dataflows.mds_client import MDSClient, MDSUnavailableError
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.screener.config_loader import ScreenerConfigLoader
from tradingagents.screener.technical_screener import ScreenerConfig, TechnicalScreener

from .results_cache import ResultsCache

logger = logging.getLogger(__name__)

_SCREENER_YAML = Path(__file__).resolve().parent.parent.parent / "screener.yaml"
if not _SCREENER_YAML.exists():
    logger.debug("screener.yaml not found at %s — using code defaults for all tickers", _SCREENER_YAML)
_cache = ResultsCache()


# ── Screener helpers ──────────────────────────────────────────────────────────


def _build_screener_config(
    *,
    rsi_min: float = 50.0,
    rsi_max: float = 70.0,
    vol_osc_min: float = 0.0,
    dist_ma_min: float = -5.0,
    dist_ma_max: float = 5.0,
) -> ScreenerConfig:
    """Build a ScreenerConfig from caller-supplied thresholds."""
    return ScreenerConfig(
        rsi_min=rsi_min,
        rsi_max=rsi_max,
        vol_osc_min=vol_osc_min,
        dist_ma_min=dist_ma_min,
        dist_ma_max=dist_ma_max,
    )


def _resolve_tickers(mds_client: MDSClient, tickers: list[str] | None) -> list[str]:
    """Return the ticker list — from the caller if provided, otherwise from MDS."""
    if tickers:
        return [t.upper() for t in tickers]
    return mds_client.get_tickers()


def _serialize_screen_results(results, date: str) -> dict[str, Any]:
    """Convert a list of ScreenerResult objects to a JSON-friendly dict."""
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]
    return {
        "date": date,
        "total": len(results),
        "passed_count": len(passed),
        "failed_count": len(failed),
        "passed": [
            {
                "ticker": r.ticker,
                "rsi": round(r.rsi, 2) if r.rsi is not None else None,
                "vol_osc": round(r.vol_osc, 2) if r.vol_osc is not None else None,
                "dist_ma_pct": round(r.dist_ma_pct, 2) if r.dist_ma_pct is not None else None,
            }
            for r in passed
        ],
        "failed": [
            {
                "ticker": r.ticker,
                "rsi": round(r.rsi, 2) if r.rsi is not None else None,
                "vol_osc": round(r.vol_osc, 2) if r.vol_osc is not None else None,
                "dist_ma_pct": round(r.dist_ma_pct, 2) if r.dist_ma_pct is not None else None,
                "reason": r.reason,
            }
            for r in failed
        ],
    }


# ── Public tool functions ─────────────────────────────────────────────────────


def run_screener(
    *,
    date: str | None = None,
    tickers: list[str] | None = None,
    rsi_min: float = 50.0,
    rsi_max: float = 70.0,
    vol_osc_min: float = 0.0,
    dist_ma_min: float = -5.0,
    dist_ma_max: float = 5.0,
) -> dict[str, Any]:
    """Run the technical screener only (no LLM calls). Fast and free.

    Returns a dict with ``passed`` and ``failed`` lists, each containing
    ticker symbols and their indicator values.
    """
    date = date or datetime.today().strftime("%Y-%m-%d")
    mds_url = os.environ.get("MDS_BASE_URL", "http://localhost:8080")
    client = MDSClient(base_url=mds_url)

    screener_cfg = _build_screener_config(
        rsi_min=rsi_min,
        rsi_max=rsi_max,
        vol_osc_min=vol_osc_min,
        dist_ma_min=dist_ma_min,
        dist_ma_max=dist_ma_max,
    )
    config_loader = ScreenerConfigLoader.from_yaml(str(_SCREENER_YAML), base_defaults=screener_cfg)
    screener = TechnicalScreener(mds_client=client, config=screener_cfg, config_loader=config_loader)

    try:
        ticker_list = _resolve_tickers(client, tickers)
    except MDSUnavailableError as exc:
        return {"error": f"Cannot fetch tickers from MDS: {exc}"}

    results = screener.screen(ticker_list, date)
    serialized = _serialize_screen_results(results, date)
    _cache.save("screen", serialized)
    return serialized


def run_analysis(
    *,
    date: str | None = None,
    tickers: list[str] | None = None,
    rsi_min: float = 50.0,
    rsi_max: float = 70.0,
    vol_osc_min: float = 0.0,
    dist_ma_min: float = -5.0,
    dist_ma_max: float = 5.0,
    max_candidates: int = 20,
    analysts: str = "market,news,fundamentals",
) -> dict[str, Any]:
    """Run the full screener → LLM pipeline.

    1. Runs TechnicalScreener to filter the universe.
    2. For each passing ticker (up to *max_candidates*), runs
       ``TradingAgentsGraph.propagate()`` to get a trade decision.
    3. Caches and returns the combined result.
    """
    date = date or datetime.today().strftime("%Y-%m-%d")
    mds_url = os.environ.get("MDS_BASE_URL", "http://localhost:8080")
    client = MDSClient(base_url=mds_url)

    # ── 1. Screener ──────────────────────────────────────────────────────
    screener_cfg = _build_screener_config(
        rsi_min=rsi_min,
        rsi_max=rsi_max,
        vol_osc_min=vol_osc_min,
        dist_ma_min=dist_ma_min,
        dist_ma_max=dist_ma_max,
    )
    config_loader = ScreenerConfigLoader.from_yaml(str(_SCREENER_YAML), base_defaults=screener_cfg)
    screener = TechnicalScreener(mds_client=client, config=screener_cfg, config_loader=config_loader)

    try:
        ticker_list = _resolve_tickers(client, tickers)
    except MDSUnavailableError as exc:
        return {"error": f"Cannot fetch tickers from MDS: {exc}"}

    screen_results = screener.screen(ticker_list, date)
    candidates = [r.ticker for r in screen_results if r.passed]

    if not candidates:
        result = {
            "date": date,
            "screening": _serialize_screen_results(screen_results, date),
            "trade_decisions": [],
            "message": "No candidates passed screening.",
        }
        _cache.save("analyze", result)
        return result

    if len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]

    # ── 2. LLM pipeline ─────────────────────────────────────────────────
    # Lazy import — only needed when actually running the LLM pipeline
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    analyst_list = [a.strip() for a in analysts.split(",")]
    ta_config = {
        **DEFAULT_CONFIG,
        "mds_base_url": mds_url,
        "data_vendors": {
            "core_stock_apis": "mds",
            "technical_indicators": "mds",
            "fundamental_data": "yfinance",
            "news_data": "yfinance",
        },
    }

    ta = TradingAgentsGraph(selected_analysts=analyst_list, config=ta_config)
    trade_decisions: list[dict[str, Any]] = []

    for ticker in candidates:
        try:
            final_state, signal = ta.propagate(ticker, date)
            decision = final_state.get("final_trade_decision", "UNKNOWN")
            trade_decisions.append({
                "ticker": ticker,
                "signal": signal,
                "decision": decision,
            })
        except Exception as exc:
            logger.error(f"LLM pipeline error for {ticker}: {exc}")
            trade_decisions.append({
                "ticker": ticker,
                "signal": None,
                "decision": "ERROR",
                "error": str(exc),
            })

    result = {
        "date": date,
        "screening": _serialize_screen_results(screen_results, date),
        "candidates_analyzed": len(candidates),
        "trade_decisions": trade_decisions,
    }
    _cache.save("analyze", result)
    return result


def get_rejected(*, date: str | None = None) -> dict[str, Any]:
    """Return tickers that failed screening, with indicator values and reasons.

    Uses the cached result from the last screen/analyze for *date*. If no
    cache is available (or the date doesn't match), runs a fresh screen.
    """
    date = date or datetime.today().strftime("%Y-%m-%d")

    # Try cache first
    for command in ("screen", "analyze"):
        cached = _cache.load(command)
        if cached and cached.get("date") == date:
            failed = cached.get("failed") if command == "screen" else (
                cached.get("screening", {}).get("failed", [])
            )
            if failed is not None:
                return {
                    "date": date,
                    "source": f"cached {command}",
                    "rejected_count": len(failed),
                    "rejected": failed,
                }

    # No cache hit — run a fresh screen
    screen_result = run_screener(date=date)
    return {
        "date": date,
        "source": "fresh screen",
        "rejected_count": screen_result.get("failed_count", 0),
        "rejected": screen_result.get("failed", []),
    }


def check_status() -> dict[str, Any]:
    """Ping all services and return a status dict.

    Checks:
      - MDS /ready endpoint → healthy/unhealthy + ticker count
      - Redis (via MDS /api/v1/metrics) → healthy + queue depths
      - InfluxDB (via MDS /ready) → healthy
      - LLM API → test call to confirm key is valid
    """
    mds_url = os.environ.get("MDS_BASE_URL", "http://localhost:8080")
    statuses: dict[str, Any] = {}

    # ── MDS ───────────────────────────────────────────────────────────────
    try:
        resp = requests.get(f"{mds_url}/ready", timeout=5)
        if resp.ok:
            statuses["mds"] = {"status": "healthy"}
            # Try to get ticker count
            try:
                ticker_resp = requests.get(f"{mds_url}/api/v1/tickers", timeout=5)
                if ticker_resp.ok:
                    tickers = ticker_resp.json().get("tickers", [])
                    statuses["mds"]["ticker_count"] = len(tickers)
            except Exception as exc:
                logger.debug("Failed to get ticker count from MDS: %s", exc)
        else:
            statuses["mds"] = {"status": "unhealthy", "http_code": resp.status_code}
    except requests.exceptions.ConnectionError:
        statuses["mds"] = {"status": "unreachable"}
    except Exception as exc:
        statuses["mds"] = {"status": "error", "detail": str(exc)}

    # ── Metrics (Redis / InfluxDB via MDS) ────────────────────────────────
    try:
        resp = requests.get(f"{mds_url}/api/v1/metrics", timeout=5)
        if resp.ok:
            metrics = resp.json()
            # Redis info — look for queue depths
            redis_info = metrics.get("redis", metrics.get("queues", {}))
            statuses["redis"] = {
                "status": "healthy",
                "queue_depth": redis_info.get("queue_depth", redis_info.get("high_priority", "unknown")),
                "dlq_depth": redis_info.get("dlq_depth", redis_info.get("dead_letter", "unknown")),
            }
            statuses["influxdb"] = {"status": "healthy"}
        else:
            statuses["redis"] = {"status": "unknown"}
            statuses["influxdb"] = {"status": "unknown"}
    except Exception as exc:
        logger.debug("Failed to fetch MDS metrics: %s", exc)
        statuses["redis"] = {"status": "unknown", "detail": "Could not reach MDS metrics"}
        statuses["influxdb"] = {"status": "unknown", "detail": "Could not reach MDS metrics"}

    # ── LLM API ───────────────────────────────────────────────────────────
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if anthropic_key:
        try:
            resp = requests.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": anthropic_key,
                    "anthropic-version": "2023-06-01",
                },
                timeout=10,
            )
            statuses["llm"] = {
                "provider": "anthropic",
                "status": "reachable" if resp.ok else f"http_{resp.status_code}",
            }
        except Exception as exc:
            statuses["llm"] = {"provider": "anthropic", "status": "unreachable", "detail": str(exc)}
    elif openai_key:
        try:
            resp = requests.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {openai_key}"},
                timeout=10,
            )
            statuses["llm"] = {
                "provider": "openai",
                "status": "reachable" if resp.ok else f"http_{resp.status_code}",
            }
        except Exception as exc:
            statuses["llm"] = {"provider": "openai", "status": "unreachable", "detail": str(exc)}
    else:
        statuses["llm"] = {"provider": "none", "status": "no API key found"}

    return statuses


def get_last_result(*, command: str | None = None) -> dict[str, Any]:
    """Return the last cached result for the given command.

    If *command* is ``None``, returns timestamps for both ``analyze`` and
    ``screen`` cached results.
    """
    if command is None or command == "":
        summary: dict[str, Any] = {}
        for cmd in ("analyze", "screen"):
            cached = _cache.load(cmd)
            if cached:
                summary[cmd] = {
                    "saved_at": cached.get("saved_at"),
                    "date": cached.get("date"),
                }
            else:
                summary[cmd] = None
        return {"cached_results": summary}

    cached = _cache.load(command)
    if cached is None:
        return {"error": f"No cached result for '{command}'"}
    return cached
