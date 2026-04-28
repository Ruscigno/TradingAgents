"""Orquestrador da cascata — fase 4 (dry-run, etapas 0 e 1 apenas).

Etapas 2–6 (LLM) ainda não estão ligadas. Esta função roda o gate de
calendário, busca o universo via MCP e aplica o screener técnico.
Sem custo $ (zero LLM).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from tradingagents.dataflows.mds_client import MDSClient
from tradingagents.recommend.calendar import is_trading_day
from tradingagents.recommend.stage_screener import run as run_stage_screener
from tradingagents.recommend.stage_universe import run as run_stage_universe
from tradingagents.recommend.types import Candidate, StageOutcome
from tradingagents.screener.config_loader import ScreenerConfigLoader
from tradingagents.screener.technical_screener import ScreenerConfig


log = logging.getLogger(__name__)


@dataclass
class DryRunResult:
    """Resultado do dry-run.

    Se ``skipped_reason`` está set, o gate de calendário disparou e nenhuma
    etapa rodou. Caso contrário, ``stage_outcomes`` tem entradas para
    universe + screener, e ``final`` tem as candidatas que passaram pelo
    screener (ordenadas por confidence desc).
    """

    skipped_reason: str | None = None
    stage_outcomes: list[StageOutcome] = field(default_factory=list)
    final: list[Candidate] = field(default_factory=list)
    total_duration_s: float = 0.0


def run_dry_run(
    *,
    mds_client: MDSClient,
    as_of_date: str,
    mcp_url: str = "http://localhost:8080/mcp",
    config: ScreenerConfig | None = None,
    config_loader: ScreenerConfigLoader | None = None,
    max_staleness_days: int = 5,
    skip_calendar_check: bool = False,
    now: datetime | None = None,
) -> DryRunResult:
    """Roda gate calendário → etapa 0 → etapa 1.

    Args:
        mds_client: cliente HTTP do market-data-service (REST API) para o
            screener.
        as_of_date: data de referência para os indicadores ``YYYY-MM-DD``.
        mcp_url: URL do servidor MCP (etapa 0). Default ``localhost:8080/mcp``.
        config: defaults dos thresholds do screener.
        config_loader: loader de overrides por ticker.
        max_staleness_days: descarta tickers com dados mais antigos que isso
            na etapa 0.
        skip_calendar_check: se True, ignora o gate ``is_trading_day``. Útil
            pra rodar smoke tests fora de pregão.
        now: instante de referência pra ``is_trading_day`` e staleness.
            Default ``datetime.now(UTC)``.

    Returns:
        :class:`DryRunResult` com outcomes por etapa e candidatas finais
        ordenadas por confidence desc.
    """
    t_total = time.monotonic()
    now = now or datetime.now(tz=timezone.utc)

    if not skip_calendar_check and not is_trading_day(now):
        log.info("recommend: not a trading day — skipping run")
        return DryRunResult(
            skipped_reason="not a trading day (XNYS calendar)",
            total_duration_s=time.monotonic() - t_total,
        )

    universe = run_stage_universe(
        mcp_url=mcp_url,
        max_staleness_days=max_staleness_days,
        now=now,
    )
    if not universe.candidates:
        return DryRunResult(
            stage_outcomes=[universe],
            total_duration_s=time.monotonic() - t_total,
        )

    screener = run_stage_screener(
        universe.candidates,
        as_of_date,
        mds_client=mds_client,
        config=config,
        config_loader=config_loader,
    )

    final = sorted(
        screener.passed,
        key=lambda c: c.confidence_history.get("screener", 0.0),
        reverse=True,
    )

    return DryRunResult(
        stage_outcomes=[universe, screener],
        final=final,
        total_duration_s=time.monotonic() - t_total,
    )
