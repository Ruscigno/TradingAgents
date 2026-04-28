"""Etapa 0 — seleção do universo via MCP do market-data-service.

Chama a tool ``list_tickers``, descarta tickers cujo ``last_fetch_1d`` é
mais antigo que ``max_staleness_days`` (default 5) ou ausente, e devolve
uma lista de :class:`Candidate` para a próxima etapa.

Esta etapa **não emite verdict** por candidata — quem não passa nem entra
na lista. Ver decisão D1 em ``studies/01-...md``.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Callable

from tradingagents.recommend.mcp_client import call_tool_sync
from tradingagents.recommend.types import Candidate, StageOutcome


log = logging.getLogger(__name__)


def run(
    mcp_url: str,
    max_staleness_days: int = 5,
    *,
    now: datetime | None = None,
    mcp_caller: Callable[..., dict] | None = None,
) -> StageOutcome:
    """Etapa 0: busca universo via MCP, filtra dados estagnados.

    Args:
        mcp_url: URL do servidor MCP (ex: ``http://host.docker.internal:8080/mcp``).
        max_staleness_days: descarta tickers com ``last_fetch_1d`` anterior a
            ``now - max_staleness_days``. Default 5.
        now: instante de referência. Default ``datetime.now(UTC)``.
        mcp_caller: injeção para testes — assinatura compatível com
            :func:`tradingagents.recommend.mcp_client.call_tool_sync`.

    Returns:
        :class:`StageOutcome` com ``stage_name="universe"`` e a lista de
        :class:`Candidate` que sobreviveram à filtragem inicial.
    """
    t0 = time.monotonic()
    now = now or datetime.now(tz=timezone.utc)
    cutoff = now - timedelta(days=max_staleness_days)
    caller = mcp_caller or call_tool_sync

    payload = caller(mcp_url, "list_tickers")
    raw_tickers = payload.get("tickers", []) or []

    candidates: list[Candidate] = []
    dropped_stale = 0
    dropped_missing = 0
    for entry in raw_tickers:
        ticker = entry.get("symbol")
        if not ticker:
            continue

        last_1d = entry.get("last_fetch_1d")
        if not last_1d:
            log.info("stage_universe: dropping %s (last_fetch_1d=None)", ticker)
            dropped_missing += 1
            continue

        last_dt = _parse_iso(last_1d)
        if last_dt is None or last_dt < cutoff:
            log.info(
                "stage_universe: dropping %s (stale, last_fetch_1d=%s)",
                ticker,
                last_1d,
            )
            dropped_stale += 1
            continue

        candidates.append(Candidate(ticker=ticker.upper()))

    duration = time.monotonic() - t0
    log.info(
        "stage_universe: %d in, %d kept (%d stale, %d missing), %.2fs",
        len(raw_tickers),
        len(candidates),
        dropped_stale,
        dropped_missing,
        duration,
    )

    return StageOutcome(
        stage_name="universe",
        candidates=candidates,
        duration_s=duration,
        cost_usd=0.0,
    )


def _parse_iso(value: str) -> datetime | None:
    """Parse ISO-8601 timestamp, accepting ``Z`` suffix.

    Returns None if unparseable so callers can drop the candidate
    rather than crash the whole run on one malformed row.
    """
    try:
        s = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
    except (ValueError, AttributeError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
