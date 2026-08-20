from datetime import date

import numpy as np
import pandas as pd
import pytest

from ingestion.aggregates import BASE_LEVEL, SectorAggregateService, sector_symbol


def _daily_bars(symbol: str, closes, start="2024-01-01") -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=len(closes), freq="D", tz="UTC")
    closes = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {
            "symbol": symbol,
            "timestamp": timestamps,
            "open": closes * 0.99,
            "high": closes * 1.01,
            "low": closes * 0.98,
            "close": closes,
            "volume": 100.0,
            "timeframe": "1d",
        }
    )


@pytest.fixture
def seeded(db):
    """Two sectors; ENE has a membership change (BBB joins on day 6)."""
    db.insert_market_data(
        pd.concat(
            [
                _daily_bars("AAA", [100, 102, 101, 103, 104, 105, 106, 108, 107, 110]),
                _daily_bars("BBB", [50, 51, 50, 52, 53, 54, 55, 56, 57, 58]),
                _daily_bars("CCC", [200, 202, 204, 202, 206, 208, 210, 209, 212, 214]),
            ],
            ignore_index=True,
        )
    )
    db.upsert_instruments(
        [
            {"symbol": "AAA", "kind": "equity", "sector": "Energy"},
            {"symbol": "BBB", "kind": "equity", "sector": "Energy"},
            {"symbol": "CCC", "kind": "equity", "sector": "Utilities"},
        ]
    )
    db.open_membership("^TEST", "AAA", date(2024, 1, 1), source="manual")
    db.open_membership("^TEST", "BBB", date(2024, 1, 6), source="manual")
    db.open_membership("^TEST", "CCC", date(2024, 1, 1), source="manual")
    return SectorAggregateService(db, index_symbol="^TEST", recompute_days=2)


def test_full_recompute_point_in_time_membership(db, seeded):
    report = seeded.recompute(timeframe="1d", full=True)
    assert report["status"] == "ok"
    energy = sector_symbol("Energy")
    utilities = sector_symbol("Utilities")
    assert report["rows_by_sector"][energy] == 10
    assert report["rows_by_sector"][utilities] == 10

    stored = db.get_latest_data(energy, timeframe="1d", ascending=True)
    levels = stored.set_index("timestamp")["close_price"]

    # Day 1 has no previous close: level stays at base.
    assert levels.iloc[0] == pytest.approx(BASE_LEVEL)
    # Day 2: only AAA is a member -> sector return = AAA return.
    assert levels.iloc[1] == pytest.approx(BASE_LEVEL * (102 / 100))
    # Day 6 (BBB joins): mean of AAA (105/104-1) and BBB (54/53-1) returns.
    expected_r6 = np.mean([105 / 104 - 1, 54 / 53 - 1])
    assert levels.iloc[5] / levels.iloc[4] - 1 == pytest.approx(expected_r6)
    # Day 5 (BBB not yet a member): only AAA contributes.
    assert levels.iloc[4] / levels.iloc[3] - 1 == pytest.approx(104 / 103 - 1)

    # Utilities is exactly CCC's chained return path.
    util_levels = db.get_latest_data(utilities, timeframe="1d", ascending=True)["close_price"]
    assert util_levels.iloc[-1] == pytest.approx(BASE_LEVEL * 214 / 200)

    # Synthetic instruments registered.
    instruments = db.get_instruments(kind="synthetic")
    assert set(instruments["symbol"]) >= {energy, utilities}


def test_incremental_recompute_matches_full(db, seeded):
    seeded.recompute(timeframe="1d", full=True)

    # New bars arrive (days 11-12).
    db.insert_market_data(
        pd.concat(
            [
                _daily_bars("AAA", [111, 112], start="2024-01-11"),
                _daily_bars("BBB", [59, 60], start="2024-01-11"),
                _daily_bars("CCC", [215, 216], start="2024-01-11"),
            ],
            ignore_index=True,
        )
    )
    incremental = seeded.recompute(timeframe="1d", full=False)
    assert incremental["status"] == "ok"
    assert incremental["full"] is False
    energy = sector_symbol("Energy")
    incremental_levels = db.get_latest_data(energy, timeframe="1d", ascending=True)[
        ["timestamp", "close_price"]
    ]

    full = seeded.recompute(timeframe="1d", full=True)
    assert full["status"] == "ok"
    full_levels = db.get_latest_data(energy, timeframe="1d", ascending=True)[
        ["timestamp", "close_price"]
    ]

    assert len(incremental_levels) == len(full_levels) == 12
    np.testing.assert_allclose(
        incremental_levels["close_price"].values, full_levels["close_price"].values, rtol=1e-9
    )


def test_recompute_without_sector_info_is_skipped(db):
    service = SectorAggregateService(db, index_symbol="^EMPTY")
    report = service.recompute(timeframe="1d")
    # The migrated DB seeds instruments without sectors, so aggregation skips.
    assert report["status"] in ("skipped", "no_data")
    assert report["rows_stored"] == 0
