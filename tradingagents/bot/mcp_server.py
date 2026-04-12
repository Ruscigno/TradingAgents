"""MCP server entry point — exposes TradingAgents tools to Claude via MCP.

Run standalone for testing::

    python tradingagents/bot/mcp_server.py

Or configure as an MCP server in OpenClaw's ``config.json5``::

    mcp: {
      servers: {
        trading: {
          command: "python",
          args: ["/path/to/tradingagents/bot/mcp_server.py"],
          env: { PYTHONPATH: "/path/to/TradingAgents" }
        }
      }
    }

Registers 6 tools:
  - run_analysis
  - run_screener
  - get_rejected
  - check_status
  - get_last_result
  - manage_schedule
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from tradingagents.bot import analysis_tools
from tradingagents.bot.scheduler import TradingScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Server + scheduler singletons ─────────────────────────────────────────────

mcp = FastMCP(
    "TradingAgents",
    description="Stock screening and multi-agent LLM analysis tools",
)

_scheduler = TradingScheduler()


# ── Tool definitions ──────────────────────────────────────────────────────────


@mcp.tool()
def run_analysis(
    date: str | None = None,
    tickers: list[str] | None = None,
    rsi_min: float = 50.0,
    rsi_max: float = 70.0,
    vol_osc_min: float = 0.0,
    dist_ma_min: float = -5.0,
    dist_ma_max: float = 5.0,
    max_candidates: int = 20,
    analysts: str = "market,news,fundamentals",
) -> str:
    """Run full stock analysis: technical screening followed by LLM-based multi-agent debate.

    Screens all tickers tracked by the Market Data Service (or a specified subset)
    using RSI, Volume Oscillator, and Distance from SMA50 filters. Tickers that
    pass screening are then analyzed by the TradingAgents LLM pipeline, which
    produces a BUY/OVERWEIGHT/HOLD/UNDERWEIGHT/SELL signal for each.

    Args:
        date: Analysis date in YYYY-MM-DD format. Defaults to today.
        tickers: Optional list of specific ticker symbols to analyze.
                 If omitted, all tickers tracked by MDS are used.
        rsi_min: RSI lower threshold (default 50).
        rsi_max: RSI upper threshold (default 70).
        vol_osc_min: Minimum Volume Oscillator % (default 0).
        dist_ma_min: Minimum distance from SMA50 in % (default -5).
        dist_ma_max: Maximum distance from SMA50 in % (default 5).
        max_candidates: Maximum number of candidates to run through LLM pipeline (default 20).
        analysts: Comma-separated analyst types to use (default: market,news,fundamentals).

    Returns:
        JSON string with screening results and trade decisions for each candidate.
    """
    result = analysis_tools.run_analysis(
        date=date,
        tickers=tickers,
        rsi_min=rsi_min,
        rsi_max=rsi_max,
        vol_osc_min=vol_osc_min,
        dist_ma_min=dist_ma_min,
        dist_ma_max=dist_ma_max,
        max_candidates=max_candidates,
        analysts=analysts,
    )
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def run_screener(
    date: str | None = None,
    tickers: list[str] | None = None,
    rsi_min: float = 50.0,
    rsi_max: float = 70.0,
    vol_osc_min: float = 0.0,
    dist_ma_min: float = -5.0,
    dist_ma_max: float = 5.0,
) -> str:
    """Run the technical screener only (no LLM calls). Fast and free.

    Screens tickers using RSI(14), Volume Oscillator, and Distance from SMA50.
    Returns a pass/fail table with indicator values for every ticker.

    Args:
        date: Screening date in YYYY-MM-DD format. Defaults to today.
        tickers: Optional list of specific ticker symbols. If omitted, all MDS tickers are used.
        rsi_min: RSI lower threshold (default 50).
        rsi_max: RSI upper threshold (default 70).
        vol_osc_min: Minimum Volume Oscillator % (default 0).
        dist_ma_min: Minimum distance from SMA50 in % (default -5).
        dist_ma_max: Maximum distance from SMA50 in % (default 5).

    Returns:
        JSON string with passed and failed tickers, each with indicator values.
    """
    result = analysis_tools.run_screener(
        date=date,
        tickers=tickers,
        rsi_min=rsi_min,
        rsi_max=rsi_max,
        vol_osc_min=vol_osc_min,
        dist_ma_min=dist_ma_min,
        dist_ma_max=dist_ma_max,
    )
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def get_rejected(date: str | None = None) -> str:
    """Show tickers that failed screening, with the indicator values and the exact filter that rejected them.

    Uses the cached result from the last /screen or /analyze for the given date.
    If no cache is available, runs a fresh screen.

    Args:
        date: Date to check rejections for, YYYY-MM-DD format. Defaults to today.

    Returns:
        JSON string with all rejected tickers, their indicator values, and failure reasons.
    """
    result = analysis_tools.get_rejected(date=date)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def check_status() -> str:
    """Check the health of all trading infrastructure services.

    Pings:
      - Market Data Service (MDS) — /ready endpoint
      - InfluxDB — via MDS health
      - Redis — via MDS metrics (queue depth + DLQ)
      - LLM API (Anthropic or OpenAI) — test request to confirm key is valid

    Returns:
        JSON string with status of each service (healthy/unhealthy/unreachable).
    """
    result = analysis_tools.check_status()
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def get_last_result(command: str | None = None) -> str:
    """Retrieve the last cached result without re-running anything.

    If no command is specified, returns timestamps of both cached results.

    Args:
        command: Either "analyze" or "screen". If omitted, shows summary of both.

    Returns:
        JSON string with the cached result data.
    """
    result = analysis_tools.get_last_result(command=command)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def manage_schedule(
    action: str,
    schedule_id: str | None = None,
    recurrence_hours: int | None = None,
    start_datetime: str | None = None,
    tickers: list[str] | None = None,
    rsi_min: float = 50.0,
    rsi_max: float = 70.0,
    vol_osc_min: float = 0.0,
    dist_ma_min: float = -5.0,
    dist_ma_max: float = 5.0,
    max_candidates: int = 20,
    analysts: str = "market,news,fundamentals",
) -> str:
    """Manage scheduled recurring analysis runs.

    Scheduled jobs run automatically and push results to Telegram.

    Args:
        action: One of "create", "list", or "delete".
        schedule_id: Required for "delete". The ID of the schedule to remove.
        recurrence_hours: Required for "create". How often to run (e.g. 24 for daily).
        start_datetime: For "create". First run time as YYYY-MM-DDTHH:MM. Defaults to now + recurrence.
        tickers: For "create". Optional ticker list for the scheduled analysis.
        rsi_min: For "create". RSI lower threshold (default 50).
        rsi_max: For "create". RSI upper threshold (default 70).
        vol_osc_min: For "create". Minimum Volume Oscillator % (default 0).
        dist_ma_min: For "create". Min distance from SMA50 % (default -5).
        dist_ma_max: For "create". Max distance from SMA50 % (default 5).
        max_candidates: For "create". Max tickers to run LLM pipeline on (default 20).
        analysts: For "create". Comma-separated analyst types (default: market,news,fundamentals).

    Returns:
        JSON string with the result of the action (schedule ID, list, or confirmation).
    """
    action = action.lower().strip()

    if action == "create":
        if recurrence_hours is None:
            return json.dumps({"error": "recurrence_hours is required for 'create' action"})

        params: dict[str, Any] = {
            "rsi_min": rsi_min,
            "rsi_max": rsi_max,
            "vol_osc_min": vol_osc_min,
            "dist_ma_min": dist_ma_min,
            "dist_ma_max": dist_ma_max,
            "max_candidates": max_candidates,
            "analysts": analysts,
        }
        if tickers:
            params["tickers"] = tickers

        sid = _scheduler.create_schedule(
            recurrence_hours=recurrence_hours,
            start_datetime=start_datetime,
            params=params,
        )
        return json.dumps({
            "action": "created",
            "schedule_id": sid,
            "recurrence_hours": recurrence_hours,
            "start_datetime": start_datetime,
        }, indent=2)

    elif action == "list":
        schedules = _scheduler.list_schedules()
        return json.dumps({"schedules": schedules}, indent=2, default=str)

    elif action == "delete":
        if not schedule_id:
            return json.dumps({"error": "schedule_id is required for 'delete' action"})
        deleted = _scheduler.delete_schedule(schedule_id)
        return json.dumps({
            "action": "deleted" if deleted else "not_found",
            "schedule_id": schedule_id,
        })

    else:
        return json.dumps({"error": f"Unknown action '{action}'. Use 'create', 'list', or 'delete'."})


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Start the MCP server (stdio transport) and the background scheduler."""
    logger.info("Starting TradingAgents MCP server...")
    _scheduler.start()
    try:
        mcp.run(transport="stdio")
    finally:
        _scheduler.shutdown()


if __name__ == "__main__":
    main()
