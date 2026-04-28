"""US-market trading-day gate using the XNYS exchange calendar.

The cascade `recommend` command must skip weekends and US holidays
(decision D7 in `studies/01-cascata-eliminatoria-de-recomendacoes.md`).
This module wraps `exchange_calendars` so callers don't have to know
about timezones or calendar names.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache

import exchange_calendars as xcals


@lru_cache(maxsize=4)
def _get_calendar(name: str) -> xcals.ExchangeCalendar:
    return xcals.get_calendar(name)


def is_trading_day(
    ts: datetime | None = None,
    calendar_name: str = "XNYS",
) -> bool:
    """Return True if `ts` falls on a trading session of the given exchange.

    Args:
        ts: Datetime to check. Naive datetimes are interpreted as UTC.
            Defaults to the current UTC time.
        calendar_name: `exchange_calendars` calendar identifier.
            Defaults to ``"XNYS"`` (NYSE regular session).

    Half-days (early close) still count as trading days — they are sessions,
    just shorter ones.
    """
    cal = _get_calendar(calendar_name)
    if ts is None:
        ts = datetime.now(tz=timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    local_date = ts.astimezone(cal.tz).strftime("%Y-%m-%d")
    return bool(cal.is_session(local_date))
