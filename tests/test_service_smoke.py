import pandas as pd
from sqlalchemy import text

from ingestion.provider import FetchReport, FetchResult, MarketDataProvider
from ingestion.service import IngestionService


def _bars(symbol: str, n: int = 5, timeframe: str = "1d") -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "symbol": symbol,
            "timestamp": timestamps,
            "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5,
            "volume": 100.0,
            "timeframe": timeframe,
        }
    )


class FakeProvider(MarketDataProvider):
    name = "fake"

    def __init__(self, symbols_with_data, failing=()):
        self.symbols_with_data = set(symbols_with_data)
        self.failing = set(failing)
        self.batch_size = 25
        self.calls = []

    def fetch_bars(self, symbols, interval="1h", start=None, end=None, period=None):
        self.calls.append({"symbols": list(symbols), "interval": interval, "start": start})
        report = FetchReport()
        frames = []
        for symbol in symbols:
            if symbol in self.failing:
                report.failed[symbol] = "simulated provider failure"
            elif symbol in self.symbols_with_data:
                report.ok.append(symbol)
                frames.append(_bars(symbol, timeframe=interval))
            else:
                report.empty.append(symbol)
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return FetchResult(frame=frame, report=report)

    def fetch_instrument_info(self, symbols):
        return pd.DataFrame(
            [{"symbol": s, "name": f"{s} Corp", "exchange": "NMS", "sector": None, "currency": "USD"}
             for s in symbols]
        )


CONFIG = {
    "INGESTION_CONFIG": {
        "index_symbols": ["^TIDX"],
        "sector_etf_symbols": ["ETF1"],
        "constituents_index": "^TEST",
        "backfill_start": "2024-01-01",
    },
    "MARKET_CONFIG": {"default_symbols": ["AAA", "BBB"], "sp500_symbols": [], "nasdaq_symbols": []},
}


def test_collect_incremental_end_to_end(db):
    provider = FakeProvider(symbols_with_data=["AAA", "BBB"])
    service = IngestionService(CONFIG, db, provider=provider)

    report = service.collect(timeframe="1d")  # default universe -> AAA, BBB
    assert report["status"] == "ok"
    assert report["incremental"] is True
    assert report["rows_collected"] == 10
    assert report["rows_by_symbol"] == {"AAA": 5, "BBB": 5}
    assert report["failures"] == {}
    # First call used the configured backfill window.
    assert provider.calls[0]["start"] == "2024-01-01"

    # Second run: idempotent, still 10 rows in the DB.
    second = service.collect(timeframe="1d")
    assert second["status"] == "ok"
    with db.engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM market_data")).scalar()
    assert count == 10
    # The follow-up fetch starts near the watermark, not at the backfill start.
    assert provider.calls[-1]["start"] > "2024-01-01"

    # Instruments were touched with last_seen.
    instruments = db.get_instruments(symbols=["AAA", "BBB"])
    assert len(instruments) == 2
    assert instruments["last_seen"].notna().all()


def test_one_failing_symbol_does_not_fail_the_run(db):
    provider = FakeProvider(symbols_with_data=["AAA"], failing=["BBB"])
    service = IngestionService(CONFIG, db, provider=provider)

    report = service.collect(timeframe="1d")
    assert report["status"] == "ok"
    assert report["rows_by_symbol"] == {"AAA": 5}
    assert "BBB" in report["failures"]


def test_synthetic_symbols_never_fetched(db):
    provider = FakeProvider(symbols_with_data=["AAA"])
    service = IngestionService(CONFIG, db, provider=provider)
    report = service.collect(symbols=["AAA", "SECT_ENRG"], timeframe="1d")
    fetched = {symbol for call in provider.calls for symbol in call["symbols"]}
    assert "SECT_ENRG" not in fetched
    assert report["rows_by_symbol"] == {"AAA": 5}


def test_explicit_window_bypasses_incremental(db):
    provider = FakeProvider(symbols_with_data=["AAA"])
    service = IngestionService(CONFIG, db, provider=provider)
    report = service.collect(symbols=["AAA"], timeframe="1d", start="2023-06-01", end="2023-07-01")
    assert report["incremental"] is False
    assert provider.calls[0]["start"] == "2023-06-01"


def test_backfill_hourly_uses_provider_lookback(db):
    provider = FakeProvider(symbols_with_data=["AAA"])
    service = IngestionService(CONFIG, db, provider=provider)
    report = service.backfill_hourly(symbols=["AAA"])
    assert report["job"] == "backfill_hourly"
    assert report["timeframe"] == "1h"
    # FakeProvider has no lookback limit; the service default (~729d) applies.
    assert provider.calls[0]["start"] is not None
