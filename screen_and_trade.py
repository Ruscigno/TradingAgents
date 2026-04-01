"""Pre-flight screener + TradingAgents pipeline.

Usage:
    python screen_and_trade.py --dry-run --date 2026-04-01
    python screen_and_trade.py --dry-run --date 2026-04-01 --verbose
    python screen_and_trade.py --date 2026-04-01 --max-candidates 5

The script:
  1. Fetches the list of tracked tickers from MDS (or uses --tickers override).
  2. Runs TechnicalScreener to filter by RSI, Volume Oscillator, and Distance
     from SMA50 — no LLM calls at this stage.
  3. Prints a screening summary table.
  4. For each passing ticker (unless --dry-run), runs TradingAgentsGraph.propagate()
     and prints the final trade decision.

Debug logging:
  Pass --verbose to emit structured JSON log lines on stderr for every
  indicator computation and filtering decision. Useful for understanding
  exactly why a specific ticker was dropped.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from tradingagents.dataflows.mds_client import MDSClient, MDSUnavailableError
from tradingagents.screener.config_loader import ScreenerConfigLoader
from tradingagents.screener.technical_screener import ScreenerConfig, TechnicalScreener

logger = logging.getLogger(__name__)

# Default config file location: screener.yaml next to this script
_DEFAULT_CONFIG_PATH = Path(__file__).parent / "screener.yaml"


def setup_logging(verbose: bool) -> None:
    """Configure root logger.

    With --verbose: DEBUG level, all structured JSON lines emitted to stderr.
    Without --verbose: WARNING level (silent unless something goes wrong).
    Format is '%(message)s' so the JSON payloads are printed as-is.
    """
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=sys.stderr,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Screen tickers via MDS then run TradingAgents on candidates"
    )
    p.add_argument(
        "--date",
        default=datetime.today().strftime("%Y-%m-%d"),
        help="Trade date in YYYY-MM-DD format (default: today)",
    )
    p.add_argument(
        "--mds-url",
        default="http://localhost:8080",
        help="Market Data Service base URL (default: http://localhost:8080)",
    )
    p.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        help="Override the MDS ticker list with a specific set (space-separated)",
    )
    p.add_argument(
        "--screener-config",
        default=str(_DEFAULT_CONFIG_PATH),
        metavar="PATH",
        help=(
            f"YAML file with per-ticker threshold overrides "
            f"(default: screener.yaml next to this script; silently ignored if absent)"
        ),
    )
    # Screener thresholds (global defaults; per-ticker YAML overrides these)
    p.add_argument("--rsi-min", type=float, default=50.0, help="Default RSI minimum (default: 50)")
    p.add_argument("--rsi-max", type=float, default=70.0, help="Default RSI maximum (default: 70)")
    p.add_argument("--vol-osc-min", type=float, default=0.0, help="Default Volume Oscillator minimum %% (default: 0)")
    p.add_argument("--dist-ma-min", type=float, default=-5.0, help="Default Distance from SMA50 minimum %% (default: -5)")
    p.add_argument("--dist-ma-max", type=float, default=5.0, help="Default Distance from SMA50 maximum %% (default: 5)")
    # Pipeline controls
    p.add_argument(
        "--analysts",
        default="market,news,fundamentals",
        help="Comma-separated list of analysts to run (default: market,news,fundamentals)",
    )
    p.add_argument(
        "--max-candidates",
        type=int,
        default=20,
        help="Safety cap: run LLM pipeline for at most N candidates (default: 20)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run screening only — skip LLM pipeline",
    )
    p.add_argument(
        "--output",
        metavar="FILE",
        help="Save JSON report to FILE",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Emit structured JSON debug logs on stderr for every indicator and filter decision",
    )
    return p.parse_args()


def fmt_float(v: float | None, decimals: int = 1) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


def print_screening_table(results) -> None:
    header = f"{'Ticker':<8} {'RSI':>6} {'VolOsc%':>8} {'DistMA%':>8}  {'Result'}"
    print(header)
    print("-" * len(header))
    for r in results:
        status = "PASS" if r.passed else f"FAIL ({r.reason})"
        print(
            f"{r.ticker:<8} {fmt_float(r.rsi):>6} {fmt_float(r.vol_osc, 2):>8} "
            f"{fmt_float(r.dist_ma_pct, 2):>8}  {status}"
        )


def run_llm_pipeline(candidates: list[str], trade_date: str, analysts: list[str], config: dict) -> list[dict]:
    # Lazy import — only needed when not --dry-run
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    ta = TradingAgentsGraph(selected_analysts=analysts, config=config)
    results = []
    for ticker in candidates:
        print(f"\n{'='*60}")
        print(f"Running TradingAgents for {ticker} on {trade_date}")
        print('='*60)
        logger.info(json.dumps({"stage": "llm_pipeline_start", "ticker": ticker, "date": trade_date}))
        try:
            final_state, signal = ta.propagate(ticker, trade_date)
            decision = final_state.get("final_trade_decision", "UNKNOWN")
            print(f"  Decision: {decision}")
            print(f"  Signal:   {signal}")
            logger.info(json.dumps({"stage": "llm_pipeline_done", "ticker": ticker, "decision": decision, "signal": signal}))
            results.append({"ticker": ticker, "decision": decision, "signal": signal})
        except Exception as exc:
            print(f"  ERROR: {exc}")
            logger.info(json.dumps({"stage": "llm_pipeline_error", "ticker": ticker, "error": str(exc)}))
            results.append({"ticker": ticker, "decision": "ERROR", "signal": None, "error": str(exc)})
    return results


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    # ── 1. Connect to MDS ────────────────────────────────────────────────────
    client = MDSClient(base_url=args.mds_url)
    logger.info(json.dumps({"stage": "mds_connect", "url": args.mds_url, "healthy": client.is_healthy()}))
    if not client.is_healthy():
        print(f"WARNING: MDS at {args.mds_url} is not reachable. Screener will fail for all tickers.", file=sys.stderr)

    # ── 2. Resolve ticker list ───────────────────────────────────────────────
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
        print(f"Using {len(tickers)} tickers from --tickers flag")
        logger.info(json.dumps({"stage": "tickers_resolved", "source": "cli_flag", "count": len(tickers)}))
    else:
        try:
            tickers = client.get_tickers()
            print(f"Fetched {len(tickers)} tickers from MDS")
            logger.info(json.dumps({"stage": "tickers_resolved", "source": "mds", "count": len(tickers)}))
        except MDSUnavailableError as exc:
            print(f"ERROR: Cannot fetch tickers from MDS: {exc}", file=sys.stderr)
            sys.exit(1)

    # ── 3. Build screener with per-ticker YAML config ────────────────────────
    screener_config = ScreenerConfig(
        rsi_min=args.rsi_min,
        rsi_max=args.rsi_max,
        vol_osc_min=args.vol_osc_min,
        dist_ma_min=args.dist_ma_min,
        dist_ma_max=args.dist_ma_max,
    )
    config_loader = ScreenerConfigLoader.from_yaml(args.screener_config, base_defaults=screener_config)

    if config_loader.tickers_with_overrides():
        logger.info(json.dumps({
            "stage": "per_ticker_config",
            "overrides": config_loader.tickers_with_overrides(),
        }))

    screener = TechnicalScreener(mds_client=client, config=screener_config, config_loader=config_loader)

    print(f"\nScreening {len(tickers)} tickers as of {args.date} ...")
    print(f"  Default thresholds — RSI: [{args.rsi_min}, {args.rsi_max}]  VolOsc > {args.vol_osc_min}%  DistMA: [{args.dist_ma_min}%, {args.dist_ma_max}%]")
    if config_loader.tickers_with_overrides():
        print(f"  Per-ticker overrides loaded for: {', '.join(config_loader.tickers_with_overrides())}")
    print()

    screen_results = screener.screen(tickers, args.date)
    print_screening_table(screen_results)

    candidates = [r.ticker for r in screen_results if r.passed]
    print(f"\nCandidates: {len(candidates)} / {len(tickers)} tickers passed")

    if not candidates:
        print("No candidates passed screening. Exiting.")
        sys.exit(0)

    if len(candidates) > args.max_candidates:
        print(f"Capping candidates to {args.max_candidates} (use --max-candidates to change)")
        logger.info(json.dumps({"stage": "candidate_capped", "from": len(candidates), "to": args.max_candidates}))
        candidates = candidates[: args.max_candidates]

    # ── 4. LLM pipeline ─────────────────────────────────────────────────────
    trade_results: list[dict] = []
    if args.dry_run:
        print("\n--dry-run: skipping LLM pipeline")
        for ticker in candidates:
            trade_results.append({"ticker": ticker, "decision": "DRY_RUN"})
    else:
        analysts = [a.strip() for a in args.analysts.split(",")]
        ta_config = {
            "mds_base_url": args.mds_url,
            "data_vendors": {
                "core_stock_apis": "mds",
                "technical_indicators": "mds",
                "fundamental_data": "yfinance",
                "news_data": "yfinance",
            },
        }
        from tradingagents.default_config import DEFAULT_CONFIG
        merged = {**DEFAULT_CONFIG, **ta_config}
        trade_results = run_llm_pipeline(candidates, args.date, analysts, merged)

    # ── 5. Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Final Results")
    print('='*60)
    print(f"{'Ticker':<8}  {'Decision'}")
    print("-" * 30)
    for r in trade_results:
        print(f"{r['ticker']:<8}  {r['decision']}")

    # ── 6. Optional JSON report ──────────────────────────────────────────────
    if args.output:
        report = {
            "date": args.date,
            "screener_config": vars(screener_config),
            "screener_config_file": args.screener_config,
            "per_ticker_overrides": config_loader.tickers_with_overrides(),
            "screening": [
                {
                    "ticker": r.ticker,
                    "passed": r.passed,
                    "rsi": r.rsi,
                    "vol_osc": r.vol_osc,
                    "dist_ma_pct": r.dist_ma_pct,
                    "reason": r.reason,
                }
                for r in screen_results
            ],
            "trade_decisions": trade_results,
        }
        Path(args.output).write_text(json.dumps(report, indent=2, default=str))
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
