"""Etapa 1 — filtro técnico determinístico (RSI / VolOsc / Dist SMA50).

Reaproveita :class:`tradingagents.screener.technical_screener.TechnicalScreener`
existente. Esta camada apenas:

  - despacha as candidatas para o screener,
  - converte o resultado em :class:`Candidate` com ``verdict`` e ``confidence``,
  - calcula a confidence em [0, 1] como média de 3 indicadores normalizados.

Critérios de PASS (vêm da config; defaults em ``ScreenerConfig``):

  - ``rsi_min ≤ RSI ≤ rsi_max``
  - ``vol_osc > vol_osc_min``
  - ``dist_ma_min ≤ dist_ma_pct ≤ dist_ma_max``

Falhar em qualquer um → ``verdict = NO_PASS`` com a razão registrada.
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

from tradingagents.dataflows.mds_client import MDSClient
from tradingagents.recommend.types import Candidate, Elimination, StageOutcome
from tradingagents.screener.config_loader import ScreenerConfigLoader
from tradingagents.screener.technical_screener import (
    ScreenerConfig,
    ScreenerResult,
    TechnicalScreener,
)


log = logging.getLogger(__name__)
STAGE = "screener"

# vol_osc satura nesta porcentagem — escolha pragmática para mapear strength → confidence
_VOL_OSC_REFERENCE_STRENGTH = 30.0


class _ScreenerLike(Protocol):
    """Subset of :class:`TechnicalScreener` que esta etapa precisa."""

    def screen(self, tickers: list[str], as_of_date: str) -> list[ScreenerResult]: ...


def run(
    candidates: list[Candidate],
    as_of_date: str,
    *,
    mds_client: MDSClient | None = None,
    config: ScreenerConfig | None = None,
    config_loader: ScreenerConfigLoader | None = None,
    screener: _ScreenerLike | None = None,
) -> StageOutcome:
    """Aplica o filtro técnico em todas as candidatas.

    Args:
        candidates: lista vinda da etapa 0 (universe).
        as_of_date: data de referência ``YYYY-MM-DD``. Os indicadores são
            calculados com OHLC histórico até (incluindo) este dia.
        mds_client: cliente HTTP do market-data-service. Obrigatório se
            ``screener`` não for fornecido.
        config: thresholds default. Default: :class:`ScreenerConfig` zero-arg.
        config_loader: para overrides por ticker via YAML. Default: None.
        screener: injeção para testes — qualquer objeto com ``screen()``
            compatível.

    Returns:
        :class:`StageOutcome` com **todas** as candidatas (PASS e NO_PASS),
        cada uma com ``confidence_history['screener']`` (se PASS) ou
        ``elimination`` (se NO_PASS).
    """
    t0 = time.monotonic()

    if not candidates:
        return StageOutcome(stage_name=STAGE, candidates=[], duration_s=0.0)

    if screener is None:
        if mds_client is None:
            raise ValueError("Either `screener` or `mds_client` must be provided")
        screener = TechnicalScreener(
            mds_client=mds_client,
            config=config or ScreenerConfig(),
            config_loader=config_loader,
        )

    base_cfg = config or ScreenerConfig()
    tickers = [c.ticker for c in candidates]
    results = screener.screen(tickers, as_of_date)
    by_ticker = {r.ticker: r for r in results}

    out: list[Candidate] = []
    for cand in candidates:
        # Clone to avoid mutating the input — preserves earlier stage outcomes
        # for accurate summary stats and audit logs.
        new_cand = Candidate(
            ticker=cand.ticker,
            confidence_history=dict(cand.confidence_history),
            reports=dict(cand.reports),
            elimination=cand.elimination,
            final_score=cand.final_score,
        )

        result = by_ticker.get(new_cand.ticker)
        if result is None:
            new_cand.elimination = Elimination(STAGE, "no screener result returned", 0.0)
            out.append(new_cand)
            continue

        ticker_cfg = (
            config_loader.get_config(new_cand.ticker)
            if config_loader is not None
            else base_cfg
        )
        confidence = _compute_confidence(result, ticker_cfg)

        if result.passed:
            new_cand.confidence_history[STAGE] = confidence
            new_cand.reports[STAGE] = _format_pass_report(result)
        else:
            new_cand.elimination = Elimination(STAGE, result.reason, confidence)
        out.append(new_cand)

    duration = time.monotonic() - t0
    passed = sum(1 for c in out if c.elimination is None)
    log.info(
        "stage_screener: %d in, %d kept, %d eliminated, %.2fs",
        len(candidates),
        passed,
        len(candidates) - passed,
        duration,
    )

    return StageOutcome(
        stage_name=STAGE,
        candidates=out,
        cost_usd=0.0,
        duration_s=duration,
    )


# ── Confidence calculation ────────────────────────────────────────────────────


def _compute_confidence(result: ScreenerResult, cfg: ScreenerConfig) -> float:
    """Confidence em [0, 1] — média dos 3 indicadores normalizados.

    Mesmo NO_PASS recebe um valor (≥ 0) que reflete "quão forte" foi a falha;
    usamos isso só para registrar em :class:`Elimination`. Não vai pro
    ``confidence_history`` que entra no score acumulado final.
    """
    c_rsi = _normalize_two_sided(result.rsi, cfg.rsi_min, cfg.rsi_max)
    c_sma = _normalize_two_sided(result.dist_ma_pct, cfg.dist_ma_min, cfg.dist_ma_max)
    c_vol = _normalize_one_sided(
        result.vol_osc, cfg.vol_osc_min, _VOL_OSC_REFERENCE_STRENGTH
    )
    return (c_rsi + c_sma + c_vol) / 3


def _normalize_two_sided(value: float | None, lo: float, hi: float) -> float:
    """1.0 no centro de [lo, hi], 0.0 na borda ou fora, clipped.

    Permite intervalos assimétricos (ex: dist_ma_min=-8, dist_ma_max=5).
    """
    if value is None or hi <= lo:
        return 0.0
    center = (lo + hi) / 2
    half = (hi - lo) / 2
    return max(0.0, 1.0 - abs(value - center) / half)


def _normalize_one_sided(
    value: float | None, threshold: float, reference_strength: float
) -> float:
    """0.0 no threshold, 1.0 em ``threshold + reference_strength``, clamped."""
    if value is None or reference_strength <= 0:
        return 0.0
    above = max(0.0, value - threshold)
    return min(above / reference_strength, 1.0)


def _format_pass_report(result: ScreenerResult) -> str:
    """Resumo curto pra alimentar etapas seguintes (entra no prompt do LLM)."""
    return (
        f"Technical screener PASS: "
        f"RSI={result.rsi:.1f}, "
        f"VolOsc={result.vol_osc:+.2f}%, "
        f"DistSMA50={result.dist_ma_pct:+.2f}%"
    )
