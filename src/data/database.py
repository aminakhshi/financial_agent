import os
from datetime import datetime, timezone
from typing import Iterable, Optional

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Index, UniqueConstraint, text
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError
from urllib.parse import quote_plus
import pandas as pd

# load local .env 
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

Base = declarative_base()

class MarketData(Base):
    __tablename__ = 'market_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False)
    exchange = Column(String(10), nullable=False)  # 'SP500' or 'NASDAQ'
    timestamp = Column(DateTime, nullable=False)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

    # Technical indicators (optional)
    sma_20 = Column(Float)
    ema_12 = Column(Float)
    rsi = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    bollinger_upper = Column(Float)
    bollinger_lower = Column(Float)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        # Avoid duplicates when reloading the same bar
        UniqueConstraint('symbol', 'exchange', 'timestamp', name='uq_marketdata_symbol_exch_ts'),
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_exchange_timestamp', 'exchange', 'timestamp'),
        Index('idx_timestamp', 'timestamp'),
    )


class PredictionResults(Base):
    __tablename__ = 'prediction_results'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False)
    prediction_timestamp = Column(DateTime, nullable=False)
    predicted_price = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=False)
    model_version = Column(String(50), nullable=False)
    actual_price = Column(Float)  # Filled later for evaluation
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index('idx_prediction_symbol_timestamp', 'symbol', 'prediction_timestamp'),
        Index('idx_prediction_timestamp', 'prediction_timestamp'),
    )


def _create_db_url_from_config() -> str:
    """Create a psycopg2 URL from user-provided DB_* vars in the .env."""
    host = os.getenv("DB_HOST", "127.0.0.1") or "127.0.0.1"
    port = os.getenv("DB_PORT", "5432") or "5432"
    name = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    pwd = os.getenv("DB_PASSWORD")

    if not all([name, user, pwd]):
        raise RuntimeError(
            "Missing DB_* vars; set them in .env or provide DB_URL."
        )

    # special chars in the password (@, :, /, #, & ...)
    pwd_enc = quote_plus(pwd)
    return f"postgresql+psycopg2://{user}:{pwd_enc}@{host}:{port}/{name}"


def _get_db_url() -> str:
    # Prefer a full DB_URL or DATABASE_URL if provided; otherwise create it from DB_* vars.
    url = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
    if url:
        return url
    return _create_db_url_from_config()


class DatabaseManager:
    def __init__(self, config=None, use_sqlite_fallback=False):
        # Expose model classes for callers that reference them via manager instance.
        self.MarketData = MarketData
        self.PredictionResults = PredictionResults
        try:
            if config is None:
                # fall back to env 
                db_url = _get_db_url()
            else:
                from urllib.parse import quote_plus
                pwd_enc = quote_plus(config["password"])
                host = config.get("host", "127.0.0.1")
                port = config.get("port", 5432)
                db_url = (
                    f"postgresql+psycopg2://{config['user']}:{pwd_enc}@"
                    f"{host}:{port}/{config['database']}"
                )

            self.is_sqlite = False
            engine_kwargs = {
                "pool_pre_ping": True,
                "pool_recycle": 1800,
                "future": True,
            }
            if db_url.startswith("postgresql"):
                engine_kwargs["connect_args"] = {"connect_timeout": 5}
            self.engine = create_engine(db_url, **engine_kwargs)
            self.is_sqlite = db_url.startswith("sqlite")
        except Exception as e:
            if use_sqlite_fallback:
                print("\nWARNING: Unable to connect to PostgreSQL database. Falling back to SQLite.")
                print(f"{e}")
                sqlite_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                          "financial_data.db")
                db_url = f"sqlite:///{sqlite_path}"
                self.is_sqlite = True
                self.engine = create_engine(db_url)
            else:
                raise
        self.Session = sessionmaker(bind=self.engine, future=True)
    def _wait_for_db(self, tries=10, delay=1.5):
        """Optional: helpful in CI/startup races."""
        # Skip DB check for SQLite since it always works
        if hasattr(self, 'is_sqlite') and self.is_sqlite:
            return
            
        import time
        for i in range(tries):
            try:
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                return
            except OperationalError as e:
                if "database" in str(e) and "does not exist" in str(e):
                    db_user = os.getenv("DB_USER", "postgres")
                    print("\nERROR: Database does not exist. Please create it manually with:")
                    print("sudo -u postgres psql -c \"CREATE DATABASE financial_data;\"")
                    print(f"sudo -u postgres psql -c \"GRANT ALL PRIVILEGES ON DATABASE financial_data TO {db_user};\"\n")
                    print("Attempting to use SQLite as fallback...")
                    
                    # Create a SQLite connection instead
                    sqlite_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                             "financial_data.db")
                    db_url = f"sqlite:///{sqlite_path}"
                    self.engine = create_engine(db_url)
                    self.Session = sessionmaker(bind=self.engine, future=True)
                    self.is_sqlite = True
                    return
                    
                if i == tries - 1:
                    raise
                time.sleep(delay)

    def create_tables(self):
        self._wait_for_db()
        Base.metadata.create_all(self.engine)

    def insert_market_data(self, df: pd.DataFrame, exchange: Optional[str] = None):
        """Insert market data with handling for both PostgreSQL and SQLite."""
        if df.empty:
            return

        # Ensure required columns are present
        required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        if exchange is None and "exchange" not in df.columns:
            raise ValueError("Exchange must be provided either as an argument or a dataframe column.")
        
        if self.is_sqlite:
            self._insert_market_data_sqlite(df, exchange)
        else:
            self._insert_market_data_postgres(df, exchange)
            
    def _insert_market_data_postgres(self, df: pd.DataFrame, exchange: Optional[str]):
        """Fast bulk upsert on PostgreSQL, deduped by (symbol, exchange, timestamp)."""
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        # Normalize to Python types that SQLAlchemy can handle
        records = []
        for _, row in df.iterrows():
            records.append({
                "symbol": row["symbol"],
                "exchange": row["exchange"] if "exchange" in df.columns else exchange,
                "timestamp": pd.to_datetime(row["timestamp"], utc=True).to_pydatetime(),
                "open_price": float(row["open"]),
                "high_price": float(row["high"]),
                "low_price": float(row["low"]),
                "close_price": float(row["close"]),
                "volume": float(row["volume"]),
                # indicators if present
                "sma_20": float(row["sma_20"]) if "sma_20" in df.columns and pd.notna(row["sma_20"]) else None,
                "ema_12": float(row["ema_12"]) if "ema_12" in df.columns and pd.notna(row["ema_12"]) else None,
                "rsi": float(row["rsi"]) if "rsi" in df.columns and pd.notna(row["rsi"]) else None,
                "macd": float(row["macd"]) if "macd" in df.columns and pd.notna(row["macd"]) else None,
                "macd_signal": float(row["macd_signal"]) if "macd_signal" in df.columns and pd.notna(row["macd_signal"]) else None,
                "bollinger_upper": float(row["bollinger_upper"]) if "bollinger_upper" in df.columns and pd.notna(row["bollinger_upper"]) else None,
                "bollinger_lower": float(row["bollinger_lower"]) if "bollinger_lower" in df.columns and pd.notna(row["bollinger_lower"]) else None,
            })

        stmt = pg_insert(MarketData).values(records)
        update_cols = {
            c.name: stmt.excluded[c.name]
            for c in MarketData.__table__.columns
            if c.name not in ("id", "created_at")  # don't overwrite id/created_at
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "exchange", "timestamp"],
            set_=update_cols,
        )

        with self.engine.begin() as conn:
            conn.execute(stmt)

    def get_latest_data(
        self,
        symbol: str,
        limit_rows: int = 500,
        exchange: Optional[str] = None,
        ascending: bool = True,
        deduplicate: bool = True,
    ):
        """Return recent rows for a symbol with optional canonical timestamp deduplication."""
        from sqlalchemy import select, desc

        query_limit = limit_rows * 5 if deduplicate and exchange is None else limit_rows
        with self.engine.connect() as conn:
            stmt = (
                select(MarketData)
                .where(MarketData.symbol == symbol)
                .order_by(desc(MarketData.timestamp))
                .limit(query_limit)
            )
            if exchange:
                stmt = stmt.where(MarketData.exchange == exchange)
            df = pd.read_sql(stmt, conn)

        if df.empty:
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        if deduplicate:
            sort_columns = ["timestamp"]
            if "created_at" in df.columns:
                sort_columns.append("created_at")
            df = df.sort_values(sort_columns).drop_duplicates(subset=["timestamp"], keep="last")
        df = df.sort_values("timestamp", ascending=ascending)
        if limit_rows:
            df = df.tail(limit_rows) if ascending else df.head(limit_rows)
        return df.reset_index(drop=True)

    def get_market_history(
        self,
        symbols: Optional[Iterable[str]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        exchange: Optional[str] = None,
        limit_rows: Optional[int] = 1000,
        ascending: bool = False,
    ) -> pd.DataFrame:
        """Return market history for one or more symbols from SQL storage."""
        from sqlalchemy import asc, desc, select

        with self.engine.connect() as conn:
            stmt = select(MarketData)
            if symbols:
                normalized_symbols = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
                if normalized_symbols:
                    stmt = stmt.where(MarketData.symbol.in_(normalized_symbols))
            if exchange:
                stmt = stmt.where(MarketData.exchange == exchange)
            if start is not None:
                stmt = stmt.where(MarketData.timestamp >= pd.to_datetime(start, utc=True).to_pydatetime())
            if end is not None:
                stmt = stmt.where(MarketData.timestamp <= pd.to_datetime(end, utc=True).to_pydatetime())

            if limit_rows:
                stmt = stmt.order_by(desc(MarketData.timestamp)).limit(limit_rows)
            else:
                stmt = stmt.order_by(asc(MarketData.timestamp) if ascending else desc(MarketData.timestamp))

            df = pd.read_sql(stmt, conn)

        if df.empty:
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        return df.sort_values("timestamp", ascending=ascending).reset_index(drop=True)

    def upsert_prediction_results(self, records: Iterable[dict]) -> int:
        """Insert or update prediction rows keyed by symbol and prediction timestamp."""
        records = list(records)
        if not records:
            return 0

        with self.Session() as session:
            updated = 0
            for record in records:
                prediction_timestamp = pd.to_datetime(record["prediction_timestamp"], utc=True).to_pydatetime()
                existing = session.query(PredictionResults).filter(
                    PredictionResults.symbol == record["symbol"],
                    PredictionResults.prediction_timestamp == prediction_timestamp,
                ).first()

                values = {
                    "symbol": record["symbol"],
                    "prediction_timestamp": prediction_timestamp,
                    "predicted_price": float(record["predicted_price"]),
                    "confidence_score": float(record["confidence_score"]),
                    "model_version": record["model_version"],
                    "actual_price": float(record["actual_price"]) if record.get("actual_price") is not None else None,
                }

                if existing:
                    for key, value in values.items():
                        setattr(existing, key, value)
                else:
                    session.add(PredictionResults(**values))
                updated += 1
            session.commit()
        return updated

    def get_latest_predictions(
        self,
        symbol: str,
        limit_rows: int = 24,
        ascending: bool = False,
    ) -> pd.DataFrame:
        """Return recent predictions for a symbol."""
        from sqlalchemy import select, desc

        with self.engine.connect() as conn:
            stmt = (
                select(PredictionResults)
                .where(PredictionResults.symbol == symbol)
                .order_by(desc(PredictionResults.prediction_timestamp))
                .limit(limit_rows)
            )
            df = pd.read_sql(stmt, conn)

        if df.empty:
            return df

        df["prediction_timestamp"] = pd.to_datetime(df["prediction_timestamp"], utc=True)
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        return df.sort_values("prediction_timestamp", ascending=ascending).reset_index(drop=True)

    def get_prediction_history(
        self,
        symbols: Optional[Iterable[str]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit_rows: Optional[int] = 1000,
        ascending: bool = False,
        only_evaluated: bool = False,
    ) -> pd.DataFrame:
        """Return persisted prediction history across one or more symbols."""
        from sqlalchemy import asc, desc, select

        with self.engine.connect() as conn:
            stmt = select(PredictionResults)
            if symbols:
                normalized_symbols = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
                if normalized_symbols:
                    stmt = stmt.where(PredictionResults.symbol.in_(normalized_symbols))
            if start is not None:
                stmt = stmt.where(
                    PredictionResults.prediction_timestamp >= pd.to_datetime(start, utc=True).to_pydatetime()
                )
            if end is not None:
                stmt = stmt.where(
                    PredictionResults.prediction_timestamp <= pd.to_datetime(end, utc=True).to_pydatetime()
                )
            if only_evaluated:
                stmt = stmt.where(PredictionResults.actual_price.is_not(None))

            if limit_rows:
                stmt = stmt.order_by(desc(PredictionResults.prediction_timestamp)).limit(limit_rows)
            else:
                stmt = stmt.order_by(
                    asc(PredictionResults.prediction_timestamp)
                    if ascending
                    else desc(PredictionResults.prediction_timestamp)
                )

            df = pd.read_sql(stmt, conn)

        if df.empty:
            return df

        df["prediction_timestamp"] = pd.to_datetime(df["prediction_timestamp"], utc=True)
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        return df.sort_values("prediction_timestamp", ascending=ascending).reset_index(drop=True)

    def sync_prediction_actuals(self, symbols: Optional[Iterable[str]] = None) -> int:
        """Fill actual prices for predictions once matching market bars are available."""
        with self.Session() as session:
            pending_query = session.query(PredictionResults).filter(
                PredictionResults.actual_price.is_(None)
            )
            if symbols:
                pending_query = pending_query.filter(PredictionResults.symbol.in_(list(symbols)))

            pending_predictions = pending_query.all()
            updated = 0
            for prediction in pending_predictions:
                market_row = (
                    session.query(MarketData)
                    .filter(
                        MarketData.symbol == prediction.symbol,
                        MarketData.timestamp == prediction.prediction_timestamp,
                    )
                    .order_by(MarketData.created_at.desc())
                    .first()
                )
                if market_row is None:
                    continue
                prediction.actual_price = market_row.close_price
                updated += 1

            if updated:
                session.commit()
            else:
                session.rollback()
        return updated
        
    def _insert_market_data_sqlite(self, df: pd.DataFrame, exchange: Optional[str]):
        """SQLite implementation for inserting market data."""
        # Convert DataFrame to SQLAlchemy-compatible records
        records = []
        for _, row in df.iterrows():
            record = {
                "symbol": row["symbol"],
                "exchange": row["exchange"] if "exchange" in df.columns else exchange,
                "timestamp": pd.to_datetime(row["timestamp"], utc=True).to_pydatetime(),
                "open_price": float(row["open"]),
                "high_price": float(row["high"]),
                "low_price": float(row["low"]),
                "close_price": float(row["close"]),
                "volume": float(row["volume"]),
                # indicators if present
                "sma_20": float(row["sma_20"]) if "sma_20" in df.columns and pd.notna(row["sma_20"]) else None,
                "ema_12": float(row["ema_12"]) if "ema_12" in df.columns and pd.notna(row["ema_12"]) else None,
                "rsi": float(row["rsi"]) if "rsi" in df.columns and pd.notna(row["rsi"]) else None,
                "macd": float(row["macd"]) if "macd" in df.columns and pd.notna(row["macd"]) else None,
                "macd_signal": float(row["macd_signal"]) if "macd_signal" in df.columns and pd.notna(row["macd_signal"]) else None,
                "bollinger_upper": float(row["bollinger_upper"]) if "bollinger_upper" in df.columns and pd.notna(row["bollinger_upper"]) else None,
                "bollinger_lower": float(row["bollinger_lower"]) if "bollinger_lower" in df.columns and pd.notna(row["bollinger_lower"]) else None,
            }
            records.append(record)

        # For SQLite, we need to manually check and update existing records
        with self.Session() as session:
            for record in records:
                # Check if a record with the same symbol, exchange, and timestamp exists
                existing = session.query(MarketData).filter(
                    MarketData.symbol == record["symbol"],
                    MarketData.exchange == record["exchange"],
                    MarketData.timestamp == record["timestamp"]
                ).first()
                
                if existing:
                    # Update existing record
                    for key, value in record.items():
                        if key not in ("id", "created_at"):
                            setattr(existing, key, value)
                else:
                    # Insert new record
                    session.add(MarketData(**record))
            
            session.commit()
