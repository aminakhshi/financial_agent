"""Incremental fetch planning from per-(symbol, timeframe) watermarks.

The plan re-fetches a small overlap window (default 2 bars) before each
watermark so partially formed final bars are repaired; upserts make the overlap
idempotent. Symbols with no stored bars get the full backfill window, clamped
to the provider's maximum lookback.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class FetchPlan:
    """One provider call: a set of symbols sharing the same fetch window."""

    symbols: List[str] = field(default_factory=list)
    start: Optional[str] = None  # ISO date; None = provider default/backfill_start
    end: Optional[str] = None


def parse_bar_delta(interval: str) -> timedelta:
    interval = interval.strip().lower()
    if interval.endswith("m") and not interval.endswith("mo"):
        return timedelta(minutes=int(interval[:-1] or "1"))
    if interval.endswith("h"):
        return timedelta(hours=int(interval[:-1] or "1"))
    if interval.endswith("d"):
        return timedelta(days=int(interval[:-1] or "1"))
    raise ValueError(f"Unsupported interval: {interval}")


def plan_incremental_fetch(
    watermarks: pd.DataFrame,
    symbols: List[str],
    interval: str,
    now: Optional[datetime] = None,
    max_lookback: Optional[timedelta] = None,
    backfill_start: Optional[str] = None,
    overlap_bars: int = 2,
    last_complete_bar: Optional[datetime] = None,
) -> List[FetchPlan]:
    """Group symbols into fetch windows.

    watermarks: frame with columns (symbol, max_timestamp) for this interval.
    last_complete_bar: symbols whose watermark already covers it are skipped.
    """
    now = pd.Timestamp(now or datetime.utcnow()).tz_localize(None)
    bar_delta = parse_bar_delta(interval)

    watermark_map: Dict[str, pd.Timestamp] = {}
    if watermarks is not None and not watermarks.empty:
        for _, row in watermarks.iterrows():
            watermark_map[str(row["symbol"]).upper()] = pd.to_datetime(row["max_timestamp"], utc=True)

    earliest_allowed: Optional[pd.Timestamp] = None
    if max_lookback is not None:
        earliest_allowed = pd.Timestamp(now).tz_localize("UTC") - max_lookback

    buckets: Dict[str, FetchPlan] = {}
    for symbol in symbols:
        symbol = str(symbol).strip().upper()
        watermark = watermark_map.get(symbol)

        if watermark is not None:
            if last_complete_bar is not None and watermark >= pd.to_datetime(last_complete_bar, utc=True):
                continue  # already up to date
            start_ts = watermark - overlap_bars * bar_delta
        else:
            if backfill_start is not None:
                start_ts = pd.to_datetime(backfill_start, utc=True)
            else:
                start_ts = None

        if start_ts is not None and earliest_allowed is not None and start_ts < earliest_allowed:
            start_ts = earliest_allowed
        if start_ts is None and earliest_allowed is not None:
            start_ts = earliest_allowed

        start_key = start_ts.strftime("%Y-%m-%d") if start_ts is not None else None
        bucket = buckets.setdefault(str(start_key), FetchPlan(start=start_key))
        bucket.symbols.append(symbol)

    # Oldest windows first so long backfills begin immediately.
    return sorted(buckets.values(), key=lambda plan: (plan.start is None, plan.start or ""))
