"""Unit tests for ScreenerConfigLoader (YAML-based per-ticker config)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tradingagents.screener.config_loader import ScreenerConfigLoader, TickerScreenerConfig
from tradingagents.screener.technical_screener import ScreenerConfig


# ── Helpers ───────────────────────────────────────────────────────────────────


def _write_yaml(tmp_path: Path, content: str) -> Path:
    """Write YAML content to a temp file and return its path."""
    p = tmp_path / "screener.yaml"
    p.write_text(textwrap.dedent(content))
    return p


# ── from_yaml: missing file ───────────────────────────────────────────────────


class TestMissingFile:
    def test_missing_file_returns_base_defaults(self, tmp_path):
        loader = ScreenerConfigLoader.from_yaml(tmp_path / "nonexistent.yaml")
        cfg = loader.get_config("AAPL")
        defaults = ScreenerConfig()
        assert cfg.rsi_min == defaults.rsi_min
        assert cfg.rsi_max == defaults.rsi_max
        assert cfg.vol_osc_min == defaults.vol_osc_min
        assert cfg.dist_ma_min == defaults.dist_ma_min
        assert cfg.dist_ma_max == defaults.dist_ma_max

    def test_missing_file_no_exception_raised(self, tmp_path):
        # Must not raise even if the file is completely absent
        ScreenerConfigLoader.from_yaml(tmp_path / "nonexistent.yaml")

    def test_missing_file_with_custom_base_defaults(self, tmp_path):
        base = ScreenerConfig(rsi_min=30.0, rsi_max=80.0)
        loader = ScreenerConfigLoader.from_yaml(tmp_path / "nonexistent.yaml", base_defaults=base)
        cfg = loader.get_config("AAPL")
        assert cfg.rsi_min == 30.0
        assert cfg.rsi_max == 80.0

    def test_missing_file_tickers_with_overrides_empty(self, tmp_path):
        loader = ScreenerConfigLoader.from_yaml(tmp_path / "nonexistent.yaml")
        assert loader.tickers_with_overrides() == []


# ── from_yaml: defaults section only ─────────────────────────────────────────


class TestDefaultsSection:
    def test_defaults_section_overrides_code_defaults(self, tmp_path):
        yaml_file = _write_yaml(tmp_path, """
            defaults:
              rsi_min: 40.0
              rsi_max: 80.0
              vol_osc_min: -5.0
              dist_ma_min: -10.0
              dist_ma_max: 10.0
        """)
        loader = ScreenerConfigLoader.from_yaml(yaml_file)
        cfg = loader.get_config("MSFT")
        assert cfg.rsi_min == 40.0
        assert cfg.rsi_max == 80.0
        assert cfg.vol_osc_min == -5.0
        assert cfg.dist_ma_min == -10.0
        assert cfg.dist_ma_max == 10.0

    def test_partial_defaults_section_inherits_code_defaults(self, tmp_path):
        yaml_file = _write_yaml(tmp_path, """
            defaults:
              rsi_min: 40.0
        """)
        loader = ScreenerConfigLoader.from_yaml(yaml_file)
        cfg = loader.get_config("MSFT")
        assert cfg.rsi_min == 40.0
        # Other fields unchanged from code defaults
        assert cfg.rsi_max == ScreenerConfig().rsi_max

    def test_empty_defaults_section_uses_code_defaults(self, tmp_path):
        yaml_file = _write_yaml(tmp_path, """
            defaults:
        """)
        loader = ScreenerConfigLoader.from_yaml(yaml_file)
        code_defaults = ScreenerConfig()
        cfg = loader.get_config("GOOG")
        assert cfg.rsi_min == code_defaults.rsi_min
        assert cfg.rsi_max == code_defaults.rsi_max

    def test_cli_base_defaults_take_precedence_over_code_defaults(self, tmp_path):
        # CLI args are passed as base_defaults; YAML defaults section overrides them
        yaml_file = _write_yaml(tmp_path, """
            defaults:
              rsi_min: 45.0
        """)
        cli_base = ScreenerConfig(rsi_min=55.0, rsi_max=75.0)
        loader = ScreenerConfigLoader.from_yaml(yaml_file, base_defaults=cli_base)
        cfg = loader.get_config("AAPL")
        assert cfg.rsi_min == 45.0   # YAML defaults overrides cli_base
        assert cfg.rsi_max == 75.0   # not in YAML defaults → from cli_base


# ── from_yaml: per-ticker overrides ──────────────────────────────────────────


class TestTickerOverrides:
    def test_per_ticker_override_applies(self, tmp_path):
        yaml_file = _write_yaml(tmp_path, """
            defaults:
              rsi_min: 50.0
              rsi_max: 70.0
            tickers:
              AAPL:
                rsi_min: 45.0
        """)
        loader = ScreenerConfigLoader.from_yaml(yaml_file)
        cfg = loader.get_config("AAPL")
        assert cfg.rsi_min == 45.0

    def test_per_ticker_unspecified_fields_inherit_defaults(self, tmp_path):
        yaml_file = _write_yaml(tmp_path, """
            defaults:
              rsi_min: 50.0
              rsi_max: 70.0
            tickers:
              AAPL:
                rsi_min: 45.0   # only rsi_min overridden
        """)
        loader = ScreenerConfigLoader.from_yaml(yaml_file)
        cfg = loader.get_config("AAPL")
        assert cfg.rsi_min == 45.0
        assert cfg.rsi_max == 70.0   # inherited from YAML defaults
        assert cfg.vol_osc_min == ScreenerConfig().vol_osc_min  # from code defaults

    def test_ticker_not_in_overrides_gets_defaults(self, tmp_path):
        yaml_file = _write_yaml(tmp_path, """
            defaults:
              rsi_min: 50.0
            tickers:
              AAPL:
                rsi_min: 45.0
        """)
        loader = ScreenerConfigLoader.from_yaml(yaml_file)
        cfg = loader.get_config("NVDA")
        assert cfg.rsi_min == 50.0   # from YAML defaults, not AAPL override

    def test_ticker_lookup_is_case_insensitive(self, tmp_path):
        yaml_file = _write_yaml(tmp_path, """
            tickers:
              aapl:
                rsi_min: 20.0
        """)
        loader = ScreenerConfigLoader.from_yaml(yaml_file)
        assert loader.get_config("AAPL").rsi_min == 20.0
        assert loader.get_config("aapl").rsi_min == 20.0

    def test_multiple_tickers_independent(self, tmp_path):
        yaml_file = _write_yaml(tmp_path, """
            tickers:
              AAPL:
                rsi_min: 45.0
              GOOG:
                rsi_min: 30.0
                rsi_max: 65.0
        """)
        loader = ScreenerConfigLoader.from_yaml(yaml_file)
        aapl = loader.get_config("AAPL")
        goog = loader.get_config("GOOG")
        assert aapl.rsi_min == 45.0
        assert aapl.rsi_max == ScreenerConfig().rsi_max  # not overridden
        assert goog.rsi_min == 30.0
        assert goog.rsi_max == 65.0

    def test_tickers_with_overrides_returns_correct_list(self, tmp_path):
        yaml_file = _write_yaml(tmp_path, """
            tickers:
              AAPL:
                rsi_min: 45.0
              TSLA:
                dist_ma_max: 8.0
        """)
        loader = ScreenerConfigLoader.from_yaml(yaml_file)
        overrides = loader.tickers_with_overrides()
        assert "AAPL" in overrides
        assert "TSLA" in overrides
        assert len(overrides) == 2

    def test_all_five_fields_overridable(self, tmp_path):
        yaml_file = _write_yaml(tmp_path, """
            tickers:
              AAPL:
                rsi_min: 30.0
                rsi_max: 80.0
                vol_osc_min: -2.0
                dist_ma_min: -8.0
                dist_ma_max: 8.0
        """)
        loader = ScreenerConfigLoader.from_yaml(yaml_file)
        cfg = loader.get_config("AAPL")
        assert cfg.rsi_min == 30.0
        assert cfg.rsi_max == 80.0
        assert cfg.vol_osc_min == -2.0
        assert cfg.dist_ma_min == -8.0
        assert cfg.dist_ma_max == 8.0

    def test_structural_fields_not_overridable_per_ticker(self, tmp_path):
        """rsi_period, ma_period, lookback_days cannot be changed per-ticker via YAML."""
        yaml_file = _write_yaml(tmp_path, """
            tickers:
              AAPL:
                rsi_period: 7   # ignored — structural field
                rsi_min: 40.0
        """)
        loader = ScreenerConfigLoader.from_yaml(yaml_file)
        cfg = loader.get_config("AAPL")
        assert cfg.rsi_period == ScreenerConfig().rsi_period  # unchanged
        assert cfg.rsi_min == 40.0  # threshold field, should work


# ── TickerScreenerConfig ──────────────────────────────────────────────────────


class TestTickerScreenerConfig:
    def test_defaults_are_none(self):
        t = TickerScreenerConfig()
        assert t.rsi_min is None
        assert t.rsi_max is None
        assert t.vol_osc_min is None
        assert t.dist_ma_min is None
        assert t.dist_ma_max is None

    def test_explicit_values_stored(self):
        t = TickerScreenerConfig(rsi_min=45.0, rsi_max=75.0)
        assert t.rsi_min == 45.0
        assert t.rsi_max == 75.0
        assert t.vol_osc_min is None  # not set


# ── defaults property ─────────────────────────────────────────────────────────


class TestDefaultsProperty:
    def test_defaults_reflects_yaml_section(self, tmp_path):
        yaml_file = _write_yaml(tmp_path, """
            defaults:
              rsi_min: 42.0
        """)
        loader = ScreenerConfigLoader.from_yaml(yaml_file)
        assert loader.defaults.rsi_min == 42.0
