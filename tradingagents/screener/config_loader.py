"""Per-ticker screener threshold configuration loaded from YAML.

Merging priority (highest wins):
  per-ticker overrides  >  YAML defaults section  >  ScreenerConfig code defaults

Only the five threshold fields are configurable per-ticker.
Structural fields (rsi_period, ma_period, lookback_days, vol_osc_fast/slow)
are shared across all tickers and set once in the defaults section or via CLI.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .technical_screener import ScreenerConfig

logger = logging.getLogger(__name__)

# The five fields that may be overridden per-ticker in YAML
_THRESHOLD_FIELDS = frozenset({"rsi_min", "rsi_max", "vol_osc_min", "dist_ma_min", "dist_ma_max"})


@dataclass
class TickerScreenerConfig:
    """Partial overrides for a single ticker — all fields are optional (None = inherit)."""

    rsi_min: float | None = None
    rsi_max: float | None = None
    vol_osc_min: float | None = None
    dist_ma_min: float | None = None
    dist_ma_max: float | None = None


class ScreenerConfigLoader:
    """Resolves a per-ticker ScreenerConfig by merging YAML overrides onto defaults.

    Usage::

        loader = ScreenerConfigLoader.from_yaml("screener.yaml")
        cfg = loader.get_config("AAPL")   # returns ScreenerConfig with AAPL overrides applied

    If the YAML file does not exist the loader returns the base defaults for every ticker.
    """

    def __init__(
        self,
        defaults: ScreenerConfig,
        ticker_overrides: dict[str, TickerScreenerConfig],
    ) -> None:
        self._defaults = defaults
        self._ticker_overrides = {t.upper(): o for t, o in ticker_overrides.items()}

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        base_defaults: ScreenerConfig | None = None,
    ) -> "ScreenerConfigLoader":
        """Load a YAML config file and return a ScreenerConfigLoader.

        If *path* does not exist the loader silently uses *base_defaults* for
        every ticker (no exception raised).

        Args:
            path:          Path to the YAML file.
            base_defaults: Starting ScreenerConfig before applying the YAML
                           ``defaults:`` section. If None, uses code defaults.
        """
        base = base_defaults or ScreenerConfig()
        yaml_path = Path(path)

        if not yaml_path.exists():
            logger.debug(
                '{"stage": "config_loader", "msg": "YAML not found, using code defaults", '
                f'"path": "{yaml_path}"}}',
            )
            return cls(defaults=base, ticker_overrides={})

        try:
            import yaml  # PyYAML — transitive dep via LangChain
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for screener YAML config. "
                "Install it with: pip install pyyaml"
            ) from exc

        with yaml_path.open() as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        # Apply YAML defaults: section on top of base_defaults
        merged_defaults = _apply_threshold_overrides(base, data.get("defaults") or {})

        # Parse per-ticker overrides
        ticker_overrides: dict[str, TickerScreenerConfig] = {}
        for ticker, raw in (data.get("tickers") or {}).items():
            if not isinstance(raw, dict):
                continue
            ticker_overrides[ticker.upper()] = _parse_ticker_overrides(raw)

        logger.debug(
            f'{{"stage": "config_loader", "loaded": "{yaml_path}", '
            f'"ticker_overrides": {len(ticker_overrides)}}}',
        )
        return cls(defaults=merged_defaults, ticker_overrides=ticker_overrides)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_config(self, ticker: str) -> ScreenerConfig:
        """Return the resolved ScreenerConfig for *ticker*.

        Starts from the merged defaults and applies any per-ticker overrides.
        """
        override = self._ticker_overrides.get(ticker.upper())
        if override is None:
            return self._defaults
        return _apply_threshold_overrides(self._defaults, dataclasses.asdict(override))

    @property
    def defaults(self) -> ScreenerConfig:
        """The effective defaults (code defaults + YAML defaults section)."""
        return self._defaults

    def tickers_with_overrides(self) -> list[str]:
        """Return ticker symbols that have explicit per-ticker config."""
        return list(self._ticker_overrides.keys())


# ── Helpers ───────────────────────────────────────────────────────────────────


def _apply_threshold_overrides(base: ScreenerConfig, overrides: dict[str, Any]) -> ScreenerConfig:
    """Return a copy of *base* with threshold fields replaced by non-None values in *overrides*."""
    updates = {
        k: float(v)
        for k, v in overrides.items()
        if k in _THRESHOLD_FIELDS and v is not None
    }
    if not updates:
        return base
    return dataclasses.replace(base, **updates)


def _parse_ticker_overrides(raw: dict[str, Any]) -> TickerScreenerConfig:
    """Parse a ticker's YAML mapping into a TickerScreenerConfig."""
    kwargs = {
        k: float(v)
        for k, v in raw.items()
        if k in _THRESHOLD_FIELDS and v is not None
    }
    return TickerScreenerConfig(**kwargs)
