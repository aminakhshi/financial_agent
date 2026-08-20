"""Thin exchange-calendar wrapper (XNYS by default).

Falls back to a weekday approximation when exchange_calendars is unavailable so
one-shot CLI runs still work in minimal environments.
"""

from datetime import date, datetime, time, timedelta, timezone
from typing import Optional

import pandas as pd

try:
    from loguru import logger
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)


class MarketCalendar:
    def __init__(self, name: str = "XNYS"):
        self.name = name
        self._calendar = None
        try:
            import exchange_calendars

            self._calendar = exchange_calendars.get_calendar(name)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"exchange_calendars unavailable ({exc}); falling back to weekday-only sessions."
            )

    def is_trading_day(self, day: Optional[date] = None) -> bool:
        day = day or datetime.now(timezone.utc).date()
        if self._calendar is not None:
            try:
                return bool(self._calendar.is_session(pd.Timestamp(day)))
            except Exception:  # out-of-range dates
                pass
        return day.weekday() < 5

    def sessions_between(self, start: date, end: date) -> int:
        """Number of trading sessions in [start, end]."""
        if self._calendar is not None:
            try:
                sessions = self._calendar.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
                return int(len(sessions))
            except Exception:
                pass
        count, day = 0, start
        while day <= end:
            if day.weekday() < 5:
                count += 1
            day += timedelta(days=1)
        return count

    def last_session_close(self, now: Optional[datetime] = None) -> datetime:
        """UTC close time of the most recent completed session."""
        now = now or datetime.now(timezone.utc)
        if self._calendar is not None:
            try:
                ts = pd.Timestamp(now)
                session = self._calendar.previous_session(ts.normalize() + pd.Timedelta(days=1))
                close = self._calendar.session_close(session)
                while close > ts:
                    session = self._calendar.previous_session(session)
                    close = self._calendar.session_close(session)
                return close.to_pydatetime()
            except Exception:
                pass
        # Fallback: assume 21:00 UTC close on the last weekday.
        day = now.date()
        while day.weekday() >= 5 or datetime.combine(day, time(21, 0), tzinfo=timezone.utc) > now:
            day -= timedelta(days=1)
        return datetime.combine(day, time(21, 0), tzinfo=timezone.utc)
