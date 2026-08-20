from datetime import datetime, timedelta, timezone

import pandas as pd

from ingestion.watermark import FetchPlan, parse_bar_delta, plan_incremental_fetch

NOW = datetime(2026, 8, 19, 22, 0, tzinfo=timezone.utc)


def test_parse_bar_delta():
    assert parse_bar_delta("1h") == timedelta(hours=1)
    assert parse_bar_delta("1d") == timedelta(days=1)
    assert parse_bar_delta("30m") == timedelta(minutes=30)


def test_empty_db_gets_full_backfill_window():
    plans = plan_incremental_fetch(
        pd.DataFrame(), ["AAPL", "MSFT"], "1d", now=NOW, backfill_start="1991-01-01"
    )
    assert len(plans) == 1
    assert plans[0].start == "1991-01-01"
    assert sorted(plans[0].symbols) == ["AAPL", "MSFT"]


def test_watermarked_symbol_starts_with_overlap():
    watermarks = pd.DataFrame(
        {"symbol": ["AAPL"], "max_timestamp": [pd.Timestamp("2026-08-15", tz="UTC")]}
    )
    plans = plan_incremental_fetch(
        watermarks, ["AAPL"], "1d", now=NOW, backfill_start="1991-01-01", overlap_bars=2
    )
    assert len(plans) == 1
    assert plans[0].start == "2026-08-13"  # watermark minus 2 daily bars


def test_hourly_clamped_to_provider_lookback():
    plans = plan_incremental_fetch(
        pd.DataFrame(),
        ["AAPL"],
        "1h",
        now=NOW,
        backfill_start="1991-01-01",
        max_lookback=timedelta(days=729),
    )
    earliest = (NOW - timedelta(days=729)).strftime("%Y-%m-%d")
    assert plans[0].start == earliest


def test_up_to_date_symbols_are_skipped():
    watermarks = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "max_timestamp": [
                pd.Timestamp("2026-08-19 20:00", tz="UTC"),
                pd.Timestamp("2026-08-10", tz="UTC"),
            ],
        }
    )
    plans = plan_incremental_fetch(
        watermarks,
        ["AAPL", "MSFT"],
        "1d",
        now=NOW,
        last_complete_bar=datetime(2026, 8, 19, 20, 0, tzinfo=timezone.utc),
    )
    symbols = [symbol for plan in plans for symbol in plan.symbols]
    assert symbols == ["MSFT"]


def test_bucketing_groups_identical_windows():
    watermarks = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "max_timestamp": [
                pd.Timestamp("2026-08-15", tz="UTC"),
                pd.Timestamp("2026-08-15", tz="UTC"),
                pd.Timestamp("2026-05-01", tz="UTC"),
            ],
        }
    )
    plans = plan_incremental_fetch(watermarks, ["A", "B", "C", "D"], "1d", now=NOW,
                                   backfill_start="1991-01-01")
    assert len(plans) == 3
    # Oldest window (full backfill for D) comes first.
    assert plans[0].symbols == ["D"] and plans[0].start == "1991-01-01"
    assert plans[1].symbols == ["C"]
    assert sorted(plans[2].symbols) == ["A", "B"]
    assert all(isinstance(plan, FetchPlan) for plan in plans)
