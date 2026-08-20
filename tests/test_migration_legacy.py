"""Upgrade path for databases created before Alembic (old raw-DDL schema)."""

import pandas as pd
from sqlalchemy import create_engine, text


def _build_legacy_db(url: str):
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text(
            """
            CREATE TABLE market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol VARCHAR(10) NOT NULL,
                exchange VARCHAR(10) NOT NULL,
                timeframe VARCHAR(10) DEFAULT '1h',
                timestamp DATETIME NOT NULL,
                open_price FLOAT NOT NULL, high_price FLOAT NOT NULL,
                low_price FLOAT NOT NULL, close_price FLOAT NOT NULL, volume FLOAT NOT NULL,
                sma_20 FLOAT, ema_12 FLOAT, rsi FLOAT, macd FLOAT, macd_signal FLOAT,
                bollinger_upper FLOAT, bollinger_lower FLOAT,
                created_at DATETIME,
                CONSTRAINT uq_marketdata_symbol_exch_ts UNIQUE (symbol, exchange, timestamp)
            )
            """
        ))
        conn.execute(text("CREATE INDEX idx_exchange_timestamp ON market_data(exchange, timestamp)"))
        conn.execute(text(
            """
            INSERT INTO market_data
              (symbol, exchange, timeframe, timestamp, open_price, high_price, low_price,
               close_price, volume, created_at)
            VALUES
              ('AAPL','SP500','1d','2020-01-02 00:00:00.000000', 1,1,1,10,100,'2020-01-03 00:00:00.000000'),
              ('AAPL','US','1d','2020-01-02 00:00:00.000000', 1,1,1,11,100,'2021-01-03 00:00:00.000000'),
              ('MSFT','SP500',NULL,'2020-01-02 00:00:00.000000', 1,1,1,20,100,'2020-01-03 00:00:00.000000')
            """
        ))
    engine.dispose()


def test_legacy_db_upgrade(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path}/legacy.db"
    _build_legacy_db(url)

    monkeypatch.setenv("DB_URL", url)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from data.database import DatabaseManager

    manager = DatabaseManager()
    manager.create_tables()  # auto-stamps baseline, then upgrades

    with manager.engine.connect() as conn:
        version = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
        rows = conn.execute(
            text("SELECT symbol, timeframe, close_price FROM market_data ORDER BY symbol")
        ).fetchall()
    assert version == "0003"
    # Exchange-churn duplicate deduped, keeping the most recent write.
    assert rows[0] == ("AAPL", "1d", 11.0)
    # NULL timeframe normalized to 1h.
    assert rows[1] == ("MSFT", "1h", 20.0)

    # The new unique key holds: same symbol/timestamp under two timeframes is fine...
    manager.insert_market_data(pd.DataFrame({
        "symbol": ["AAPL"],
        "timestamp": [pd.Timestamp("2020-01-02", tz="UTC")],
        "open": [1.0], "high": [1.0], "low": [1.0], "close": [12.0],
        "volume": [1.0], "timeframe": ["1h"],
    }))
    # ...and re-inserting an existing (symbol, timeframe, timestamp) updates in place.
    manager.insert_market_data(pd.DataFrame({
        "symbol": ["AAPL"],
        "timestamp": [pd.Timestamp("2020-01-02", tz="UTC")],
        "open": [1.0], "high": [1.0], "low": [1.0], "close": [13.0],
        "volume": [1.0], "timeframe": ["1d"],
    }))
    with manager.engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM market_data")).scalar()
        close = conn.execute(text(
            "SELECT close_price FROM market_data WHERE symbol='AAPL' AND timeframe='1d'"
        )).scalar()
    assert count == 3
    assert close == 13.0

    # Migration seeded instruments and membership from stored symbols + seed file.
    instruments = manager.get_instruments(symbols=["AAPL", "MSFT"])
    assert len(instruments) == 2
    members = manager.get_open_memberships("^GSPC")
    assert len(members) >= 500

    # Idempotent: running create_tables again is a no-op.
    manager.create_tables()
