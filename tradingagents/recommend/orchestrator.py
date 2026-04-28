"""Orquestrador da cascata — encadeia etapas 0..6, skip-on-zero, score acumulado.

Modos:
- :func:`run_dry_run`: etapas 0 + 1 (sem LLM, custo $0).
- :func:`run_full`: cascata completa 0..6 com Kimi K2.6 + persistência.

Pesos do score final (decisão D3 do estudo 01) crescem com a profundidade do
estágio: análises mais profundas pesam mais. Ver :data:`FINAL_SCORE_WEIGHTS`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from tradingagents.dataflows.mds_client import MDSClient
from tradingagents.recommend.calendar import is_trading_day
from tradingagents.recommend.llm import LLMConfig, build_llm
from tradingagents.recommend.stage_analysts import run as run_stage_analysts
from tradingagents.recommend.stage_debate import run as run_stage_debate
from tradingagents.recommend.stage_decision import PositionSizing
from tradingagents.recommend.stage_decision import run as run_stage_decision
from tradingagents.recommend.stage_market import run as run_stage_market
from tradingagents.recommend.stage_research import run as run_stage_research
from tradingagents.recommend.stage_screener import run as run_stage_screener
from tradingagents.recommend.stage_universe import run as run_stage_universe
from tradingagents.recommend.types import Candidate, StageOutcome
from tradingagents.screener.config_loader import ScreenerConfigLoader
from tradingagents.screener.technical_screener import ScreenerConfig


log = logging.getLogger(__name__)


# Pesos crescentes por profundidade — etapas mais tarde viram mais informação.
# Soma = 1.0. Veja "Mecanismo de scoring acumulado" no estudo 01.
FINAL_SCORE_WEIGHTS: dict[str, float] = {
    "screener": 0.05,
    "market":   0.10,
    "analysts": 0.15,
    "debate":   0.20,
    "research": 0.25,
    "decision": 0.25,
}


def _final_score(c: Candidate) -> float:
    """Média ponderada das confidences acumuladas, normalizada pelos pesos
    das etapas que efetivamente rodaram (renormaliza se faltam stages)."""
    relevant = {k: w for k, w in FINAL_SCORE_WEIGHTS.items() if k in c.confidence_history}
    total = sum(relevant.values())
    if total == 0:
        return 0.0
    return sum(c.confidence_history[k] * (w / total) for k, w in relevant.items())


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


@dataclass
class RecommendResult:
    """Resultado de uma run completa da cascata 0..6.

    ``final`` é a lista de candidatas que sobreviveram à última etapa que
    rodou, ordenadas por ``final_score`` decrescente.
    ``skipped_stages`` lista as etapas que **não** rodaram por causa de
    skip-on-zero (alguma etapa anterior zerou).
    """

    skipped_reason: str | None = None
    stage_outcomes: list[StageOutcome] = field(default_factory=list)
    final: list[Candidate] = field(default_factory=list)
    skipped_stages: list[str] = field(default_factory=list)
    total_duration_s: float = 0.0


_LLM_STAGE_NAMES = ("market", "analysts", "debate", "research", "decision")


def _llm_stage_runner(name: str):
    """Lookup the runner module-attribute by stage name at call time.

    Resolving via ``globals()`` (rather than capturing references in a
    module-level tuple) lets ``unittest.mock.patch`` intercept the runner
    in tests without us having to refactor away the abstraction.
    """
    return globals()[f"run_stage_{name}"]


def run_full(
    *,
    mds_client: MDSClient,
    as_of_date: str,
    mcp_url: str = "http://localhost:8080/mcp",
    llm: Any | None = None,
    llm_config: LLMConfig | None = None,
    screener_config: ScreenerConfig | None = None,
    screener_loader: ScreenerConfigLoader | None = None,
    position_sizing: PositionSizing | None = None,
    max_staleness_days: int = 5,
    max_workers: int = 4,
    skip_calendar_check: bool = False,
    now: datetime | None = None,
) -> RecommendResult:
    """Roda a cascata inteira: 0=universe → 1=screener → 2=market → 3=analysts
    → 4=debate → 5=research → 6=decision. Skip-on-zero entre etapas LLM.

    Args:
        mds_client: cliente HTTP do market-data-service.
        as_of_date: data ``YYYY-MM-DD`` de referência.
        mcp_url: endpoint MCP do market-data-service.
        llm: LangChain chat client (opcional). Se None, instancia via
            :func:`build_llm` com ``llm_config``. Compartilhado por todas as
            etapas LLM — economiza re-instanciação por estágio.
        llm_config: usado quando ``llm`` é None.
        position_sizing: passado para a etapa 6.
        max_workers: paralelismo entre tickers em cada etapa LLM.
        skip_calendar_check: bypass do gate XNYS.
        now: instante de referência (testing).

    Returns:
        :class:`RecommendResult` com o histórico de etapas, lista final
        ordenada por ``final_score`` desc, e ``skipped_stages``.
    """
    t_total = time.monotonic()
    now = now or datetime.now(tz=timezone.utc)

    # Calendar gate
    if not skip_calendar_check and not is_trading_day(now):
        return RecommendResult(
            skipped_reason="not a trading day (XNYS calendar)",
            total_duration_s=time.monotonic() - t_total,
        )

    # ── Etapas determinísticas (0 + 1) ────────────────────────────────────────
    universe = run_stage_universe(
        mcp_url=mcp_url, max_staleness_days=max_staleness_days, now=now,
    )
    if not universe.candidates:
        return RecommendResult(
            stage_outcomes=[universe],
            skipped_stages=["screener", "market", "analysts", "debate", "research", "decision"],
            total_duration_s=time.monotonic() - t_total,
        )

    screener = run_stage_screener(
        universe.candidates, as_of_date,
        mds_client=mds_client, config=screener_config, config_loader=screener_loader,
    )

    stage_outcomes: list[StageOutcome] = [universe, screener]
    survivors = screener.passed

    # ── Etapas LLM (2..6) com skip-on-zero ────────────────────────────────────
    skipped_stages: list[str] = []
    if not survivors:
        skipped_stages = list(_LLM_STAGE_NAMES)
    else:
        # Reuse the same LLM instance across all LLM stages.
        if llm is None:
            llm = build_llm(llm_config)

        for name in _LLM_STAGE_NAMES:
            if not survivors:
                skipped_stages.append(name)
                log.info("orchestrator: skipping stage %s (zero survivors)", name)
                continue

            kwargs: dict[str, Any] = {
                "as_of_date": as_of_date,
                "llm": llm,
                "max_candidate_workers": max_workers,
            }
            if name == "decision":
                kwargs["position"] = position_sizing or PositionSizing()

            runner = _llm_stage_runner(name)
            outcome = runner(survivors, **kwargs)
            stage_outcomes.append(outcome)
            survivors = outcome.passed

    # ── Score final + ordenação ──────────────────────────────────────────────
    for c in survivors:
        c.final_score = _final_score(c)
    final_sorted = sorted(survivors, key=lambda c: c.final_score or 0.0, reverse=True)

    return RecommendResult(
        stage_outcomes=stage_outcomes,
        final=final_sorted,
        skipped_stages=skipped_stages,
        total_duration_s=time.monotonic() - t_total,
    )
