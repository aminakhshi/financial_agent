import numpy as np
import pandas as pd
from sqlalchemy import text


def _bars(symbol: str, n: int, timeframe: str = "1d", close: float = 100.0) -> pd.DataFrame:
    timestamps = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "symbol": symbol,
            "timestamp": timestamps,
            "open": close - 1.0,
            "high": close + 1.0,
            "low": close - 2.0,
            "close": close,
            "volume": 1000.0,
            "timeframe": timeframe,
        }
    )


def test_chunked_insert_and_idempotent_reinsert(db):
    frame = pd.concat([_bars("AAPL", 600), _bars("MSFT", 600)], ignore_index=True)
    stored = db.insert_market_data(frame)  # 1200 rows -> 3 chunks of 500
    assert stored == 1200

    with db.engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM market_data")).scalar()
    assert count == 1200

    # Re-insert with changed closes: values update, row count unchanged.
    stored_again = db.insert_market_data(frame.assign(close=frame["close"] + 5.0))
    assert stored_again == 1200
    with db.engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM market_data")).scalar()
        updated = conn.execute(
            text("SELECT COUNT(*) FROM market_data WHERE close_price = 105.0")
        ).scalar()
    assert count == 1200
    assert updated == 1200


def test_mixed_timeframes_at_same_timestamp_coexist(db):
    daily = _bars("AAPL", 5, timeframe="1d", close=50.0)
    hourly = daily.assign(timeframe="1h", close=60.0)
    db.insert_market_data(pd.concat([daily, hourly], ignore_index=True))

    with db.engine.connect() as conn:
        rows = conn.execute(
            text("SELECT timeframe, COUNT(*), MAX(close_price) FROM market_data GROUP BY timeframe")
        ).fetchall()
    by_timeframe = {row[0]: (row[1], row[2]) for row in rows}
    assert by_timeframe["1d"] == (5, 50.0)
    assert by_timeframe["1h"] == (5, 60.0)


def test_invalid_rows_dropped_not_fatal(db):
    frame = _bars("AAPL", 3)
    frame.loc[1, "open"] = np.nan  # simulates a pre-listing/bad row
    stored = db.insert_market_data(frame)
    assert stored == 2


def test_exchange_is_optional(db):
    frame = _bars("^GSPC", 3).drop(columns=["timeframe"])
    stored = db.insert_market_data(frame)  # no exchange arg, no exchange column
    assert stored == 3
    with db.engine.connect() as conn:
        timeframes = conn.execute(text("SELECT DISTINCT timeframe FROM market_data")).fetchall()
    assert timeframes == [("1h",)]  # default timeframe applied


def test_get_watermarks(db):
    db.insert_market_data(pd.concat([_bars("AAPL", 10), _bars("MSFT", 4)], ignore_index=True))
    watermarks = db.get_watermarks(symbols=["AAPL", "MSFT", "GOOG"], timeframe="1d")
    marks = dict(zip(watermarks["symbol"], watermarks["max_timestamp"]))
    assert marks["AAPL"] == pd.Timestamp("2020-01-10", tz="UTC")
    assert marks["MSFT"] == pd.Timestamp("2020-01-04", tz="UTC")
    assert "GOOG" not in marks
