"""Tests for `tradingagents.recommend.calendar.is_trading_day`."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from tradingagents.recommend.calendar import is_trading_day


# Fixed reference dates known from the NYSE 2026 calendar.
# (Pinned to dates already in the past for reproducibility.)

# Regular trading day: Monday 2026-01-05.
TRADING_DAY = datetime(2026, 1, 5, 16, 0, tzinfo=timezone.utc)

# Saturday 2026-01-03 — weekend.
WEEKEND = datetime(2026, 1, 3, 16, 0, tzinfo=timezone.utc)

# Sunday 2026-01-04.
SUNDAY = datetime(2026, 1, 4, 16, 0, tzinfo=timezone.utc)

# Thursday 2026-01-01 — New Year's Day, NYSE closed.
NEW_YEARS_DAY = datetime(2026, 1, 1, 16, 0, tzinfo=timezone.utc)

# Friday 2026-12-25 — Christmas Day, NYSE closed.
CHRISTMAS_DAY = datetime(2026, 12, 25, 16, 0, tzinfo=timezone.utc)

# Friday 2026-11-27 — Day after Thanksgiving, NYSE early close (1pm ET)
# but still a trading session.
DAY_AFTER_THANKSGIVING = datetime(2026, 11, 27, 16, 0, tzinfo=timezone.utc)


class TestIsTradingDay:
    def test_regular_weekday_is_trading(self):
        assert is_trading_day(TRADING_DAY) is True

    def test_saturday_not_trading(self):
        assert is_trading_day(WEEKEND) is False

    def test_sunday_not_trading(self):
        assert is_trading_day(SUNDAY) is False

    def test_new_years_day_not_trading(self):
        assert is_trading_day(NEW_YEARS_DAY) is False

    def test_christmas_not_trading(self):
        assert is_trading_day(CHRISTMAS_DAY) is False

    def test_half_day_still_counts_as_trading_day(self):
        # Day after Thanksgiving closes at 1pm ET, but it IS a session.
        assert is_trading_day(DAY_AFTER_THANKSGIVING) is True

    def test_default_uses_now_utc(self):
        # We can't pin "now" but we can assert it returns a bool without raising.
        result = is_trading_day()
        assert isinstance(result, bool)

    def test_naive_datetime_treated_as_utc(self):
        # Naive datetime equivalent to TRADING_DAY in UTC.
        naive = TRADING_DAY.replace(tzinfo=None)
        assert is_trading_day(naive) is True

    def test_timezone_normalization_does_not_change_session(self):
        # Same instant expressed in two different timezones must yield
        # the same answer.
        as_ny = TRADING_DAY.astimezone(
            datetime.now(tz=timezone(timedelta(hours=-5))).tzinfo
        )
        assert is_trading_day(TRADING_DAY) == is_trading_day(as_ny)

    @pytest.mark.parametrize(
        "ts,expected",
        [
            (TRADING_DAY, True),
            (WEEKEND, False),
            (SUNDAY, False),
            (NEW_YEARS_DAY, False),
            (CHRISTMAS_DAY, False),
            (DAY_AFTER_THANKSGIVING, True),
        ],
    )
    def test_table(self, ts, expected):
        assert is_trading_day(ts) is expected
