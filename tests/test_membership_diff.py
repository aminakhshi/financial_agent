from datetime import date

import pandas as pd
import pytest

from ingestion.membership import MembershipService


@pytest.fixture
def membership(db, tmp_path):
    seed_file = tmp_path / "seed.txt"
    seed_file.write_text("AAA\nBBB\nCCC.D\n# comment\n\n")
    return MembershipService(db, index_symbol="^TEST", seed_path=seed_file)


def test_seed_opens_membership_rows(membership):
    report = membership.seed(effective_from=date(2024, 1, 1))
    assert report["status"] == "ok"
    assert report["opened"] == 3
    assert membership.current_members() == ["AAA", "BBB", "CCC-D"]  # dot normalized

    # Idempotent: seeding again opens nothing new.
    again = membership.seed(effective_from=date(2024, 1, 1))
    assert again["opened"] == 0


def _fake_constituents(symbols):
    return pd.DataFrame(
        {"symbol": symbols, "name": [f"{s} Corp" for s in symbols],
         "sector": ["Energy"] * len(symbols)}
    )


def test_refresh_closes_removed_and_opens_added(membership, db, monkeypatch):
    membership.seed(effective_from=date(2024, 1, 1))
    db.insert_market_data(
        pd.DataFrame(
            {
                "symbol": ["BBB"],
                "timestamp": [pd.Timestamp("2024-06-03", tz="UTC")],
                "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
                "volume": [10.0], "timeframe": ["1d"],
            }
        )
    )

    # Plausibility guardrail is tuned for the S&P 500; relax for the fixture.
    monkeypatch.setattr("ingestion.membership.MIN_PLAUSIBLE_SIZE", 1)
    monkeypatch.setattr(
        membership, "fetch_current_constituents", lambda: _fake_constituents(["AAA", "CCC-D", "DDD"])
    )

    report = membership.refresh(asof=date(2025, 1, 1))
    assert report["status"] == "ok"
    assert report["added"] == ["DDD"]
    assert report["removed"] == ["BBB"]
    assert membership.current_members() == ["AAA", "CCC-D", "DDD"]

    # Removed symbol: membership closed, instrument inactive, price rows untouched.
    intervals = db.get_membership_intervals("^TEST")
    closed = intervals[(intervals["symbol"] == "BBB")].iloc[0]
    assert closed["effective_to"] == pd.Timestamp("2025-01-01")
    instruments = db.get_instruments(symbols=["BBB"])
    assert not bool(instruments.iloc[0]["active"])
    assert len(db.get_latest_data("BBB", timeframe="1d")) == 1

    # Point-in-time: BBB was a member in 2024, not in 2025.
    assert "BBB" in db.get_members_asof("^TEST", date(2024, 6, 1))
    assert "BBB" not in db.get_members_asof("^TEST", date(2025, 6, 1))


def test_refresh_aborts_on_implausible_list_size(membership, monkeypatch):
    membership.seed(effective_from=date(2024, 1, 1))
    monkeypatch.setattr(
        membership, "fetch_current_constituents", lambda: _fake_constituents(["AAA"])
    )
    report = membership.refresh(asof=date(2025, 1, 1))
    assert report["status"] == "aborted"
    assert membership.current_members() == ["AAA", "BBB", "CCC-D"]  # unchanged


def test_refresh_survives_source_failure(membership, monkeypatch):
    membership.seed(effective_from=date(2024, 1, 1))

    def _boom():
        raise ConnectionError("network down")

    monkeypatch.setattr(membership, "fetch_current_constituents", _boom)
    report = membership.refresh()
    assert report["status"] == "source_unavailable"
    assert membership.current_members() == ["AAA", "BBB", "CCC-D"]
