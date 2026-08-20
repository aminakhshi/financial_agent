import os
from datetime import date, datetime, timezone
from typing import Iterable, List, Optional

from sqlalchemy import (
    create_engine, event, Boolean, Column, Date, DateTime, Float, Index, Integer, String,
    UniqueConstraint, text, inspect
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

# Rows per bulk upsert statement. Keeps bind-parameter counts well under the
# PostgreSQL 65535 limit (~16 columns/row) and bounds transaction size so a
# failure loses at most one chunk of a large backfill.
UPSERT_CHUNK_SIZE = 500


class MarketData(Base):
    __tablename__ = 'market_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    # Deprecated: legacy index label ('SP500'/'NASDAQ'/'US'). Real venue now
    # lives on instruments.exchange; kept nullable for API back-compat.
    exchange = Column(String(10), nullable=True)
    timeframe = Column(String(10), nullable=False, default="1h")
    timestamp = Column(DateTime(timezone=True), nullable=False)
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

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        # One bar per symbol/resolution/moment; a 1d and a 1h bar at the same
        # timestamp are distinct rows.
        UniqueConstraint('symbol', 'timeframe', 'timestamp', name='uq_marketdata_symbol_timeframe_ts'),
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_timestamp', 'timestamp'),
        Index('idx_symbol_timeframe_timestamp', 'symbol', 'timeframe', 'timestamp'),
    )


class Instrument(Base):
    __tablename__ = 'instruments'

    symbol = Column(String(20), primary_key=True)
    name = Column(String(200))
    exchange = Column(String(20))  # real listing venue (e.g. NMS, NYQ); NULL for synthetic
    sector = Column(String(50))    # GICS sector; for synthetic series, the aggregated sector
    currency = Column(String(10))
    kind = Column(String(20), nullable=False, default='equity')  # equity | index | etf | synthetic
    active = Column(Boolean, nullable=False, default=True)
    first_seen = Column(DateTime(timezone=True))
    last_seen = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class IndexMembership(Base):
    __tablename__ = 'index_membership'

    id = Column(Integer, primary_key=True, autoincrement=True)
    index_symbol = Column(String(20), nullable=False)
    symbol = Column(String(20), nullable=False)
    effective_from = Column(Date, nullable=False)
    effective_to = Column(Date, nullable=True)  # NULL = currently a member
    source = Column(String(50))  # 'seed_file' | 'wikipedia_refresh' | 'manual'
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint('index_symbol', 'symbol', 'effective_from', name='uq_membership_index_symbol_from'),
        Index('idx_membership_index_open', 'index_symbol', 'effective_to'),
        Index('idx_membership_symbol', 'symbol'),
    )


class PredictionResults(Base):
    __tablename__ = 'prediction_results'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False, default="1h")
    prediction_timestamp = Column(DateTime(timezone=True), nullable=False)
    predicted_price = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=False)
    model_version = Column(String(50), nullable=False)
    actual_price = Column(Float)  # Filled later for evaluation
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint('symbol', 'timeframe', 'prediction_timestamp', name='uq_prediction_symbol_timeframe_ts'),
        Index('idx_prediction_symbol_timestamp', 'symbol', 'prediction_timestamp'),
        Index('idx_prediction_timestamp', 'prediction_timestamp'),
        Index('idx_prediction_symbol_timeframe_timestamp', 'symbol', 'timeframe', 'prediction_timestamp'),
    )


class ModelMonitorEvent(Base):
    __tablename__ = 'model_monitor_events'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    prediction_timestamp = Column(DateTime(timezone=True))
    observed_accuracy_pct = Column(Float)
    observed_mape = Column(Float)
    degradation_streak = Column(Integer, nullable=False, default=0)
    action = Column(String(50), nullable=False)
    model_version = Column(String(50))
    note = Column(String(500))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index('idx_monitor_symbol_timeframe_created', 'symbol', 'timeframe', 'created_at'),
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


def _configure_sqlite_engine(engine) -> None:
    """Enable WAL and a busy timeout so scheduler/API/dashboard can share one file."""

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
        finally:
            cursor.close()


class DatabaseManager:
    def __init__(self, config=None, use_sqlite_fallback=False):
        # Expose model classes for callers that reference them via manager instance.
        self.MarketData = MarketData
        self.PredictionResults = PredictionResults
        self.ModelMonitorEvent = ModelMonitorEvent
        self.Instrument = Instrument
        self.IndexMembership = IndexMembership
        self.use_sqlite_fallback = use_sqlite_fallback
        try:
            if config is None:
                # fall back to env
                db_url = _get_db_url()
            else:
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
            if self.is_sqlite:
                _configure_sqlite_engine(self.engine)
        except Exception as e:
            if self.use_sqlite_fallback:
                self._activate_sqlite_fallback(e)
            else:
                raise
        self.Session = sessionmaker(bind=self.engine, future=True)

    def _sqlite_db_url(self) -> str:
        configured_path = os.getenv("SQLITE_DB_PATH", "").strip()
        if configured_path:
            sqlite_path = configured_path
        else:
            sqlite_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "financial_data.db",
            )
        return f"sqlite:///{sqlite_path}"

    def _activate_sqlite_fallback(self, reason) -> None:
        import sys

        # stderr keeps CLI stdout clean for JSON run reports
        print("\nWARNING: Unable to use PostgreSQL. Falling back to SQLite.", file=sys.stderr)
        print(f"{reason}", file=sys.stderr)
        db_url = self._sqlite_db_url()
        self.is_sqlite = True
        self.engine = create_engine(db_url, future=True)
        _configure_sqlite_engine(self.engine)
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
                error_text = str(e)
                if "database" in error_text and "does not exist" in error_text:
                    db_user = os.getenv("DB_USER", "postgres")
                    print("\nERROR: Database does not exist. Please create it manually with:")
                    print("sudo -u postgres psql -c \"CREATE DATABASE financial_data;\"")
                    print(f"sudo -u postgres psql -c \"GRANT ALL PRIVILEGES ON DATABASE financial_data TO {db_user};\"\n")

                    if self.use_sqlite_fallback:
                        self._activate_sqlite_fallback(error_text)
                        return

                common_connection_failure = any(
                    marker in error_text.lower()
                    for marker in (
                        "password authentication failed",
                        "connection refused",
                        "could not connect to server",
                        "timeout expired",
                        "name or service not known",
                        "temporary failure in name resolution",
                    )
                )
                if self.use_sqlite_fallback and common_connection_failure:
                    self._activate_sqlite_fallback(error_text)
                    return

                if i == tries - 1:
                    if self.use_sqlite_fallback:
                        self._activate_sqlite_fallback(error_text)
                        return
                    raise
                time.sleep(delay)

    def create_tables(self):
        """Bring the schema to the latest Alembic revision.

        Databases populated before Alembic was introduced (old raw-DDL schema)
        are stamped at the baseline revision first, then upgraded, so existing
        daily history is migrated in place rather than recreated.
        """
        self._wait_for_db()
        self.run_migrations()

    def run_migrations(self, revision: str = "head"):
        from alembic import command
        from alembic.config import Config as AlembicConfig

        migrations_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "migrations")
        cfg = AlembicConfig()
        cfg.set_main_option("script_location", migrations_dir)
        cfg.attributes["connectable"] = self.engine

        inspector = inspect(self.engine)
        tables = set(inspector.get_table_names())
        if "market_data" in tables and "alembic_version" not in tables:
            # Pre-Alembic database: mark the legacy schema as the baseline.
            command.stamp(cfg, "0001")
        command.upgrade(cfg, revision)

    # ------------------------------------------------------------------
    # Market data writes
    # ------------------------------------------------------------------

    def insert_market_data(self, df: pd.DataFrame, exchange: Optional[str] = None) -> int:
        """Chunked idempotent upsert keyed by (symbol, timeframe, timestamp).

        `exchange` is deprecated and optional; if provided (argument or column)
        it is stored on the deprecated column for back-compat only.
        """
        if df.empty:
            return 0

        required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        df = df.copy()
        numeric_columns = ["open", "high", "low", "close", "volume"]
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=["symbol", "timestamp", "open", "high", "low", "close"])
        df["volume"] = df["volume"].fillna(0.0)
        if df.empty:
            return 0

        df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        if "timeframe" not in df.columns:
            df["timeframe"] = "1h"
        if "exchange" not in df.columns:
            df["exchange"] = exchange

        indicator_columns = [
            "sma_20", "ema_12", "rsi", "macd", "macd_signal", "bollinger_upper", "bollinger_lower",
        ]
        rename_map = {
            "open": "open_price", "high": "high_price", "low": "low_price", "close": "close_price",
        }
        insert_columns = (
            ["symbol", "exchange", "timeframe", "timestamp",
             "open_price", "high_price", "low_price", "close_price", "volume"]
            + [column for column in indicator_columns if column in df.columns]
        )
        frame = df.rename(columns=rename_map)[insert_columns]
        # Bulk upserts bypass ORM defaults, so created_at must be provided explicitly.
        frame["created_at"] = datetime.now(timezone.utc)
        # Last write wins on duplicate keys inside one payload.
        frame = frame.drop_duplicates(subset=["symbol", "timeframe", "timestamp"], keep="last")

        records = [
            {key: (None if pd.isna(value) else value) for key, value in record.items()}
            for record in frame.to_dict(orient="records")
        ]

        if self.is_sqlite:
            from sqlalchemy.dialects.sqlite import insert as dialect_insert
        else:
            from sqlalchemy.dialects.postgresql import insert as dialect_insert

        total = 0
        for start in range(0, len(records), UPSERT_CHUNK_SIZE):
            chunk = records[start:start + UPSERT_CHUNK_SIZE]
            stmt = dialect_insert(MarketData).values(chunk)
            update_cols = {
                c.name: stmt.excluded[c.name]
                for c in MarketData.__table__.columns
                if c.name not in ("id", "created_at", "symbol", "timeframe", "timestamp")
            }
            stmt = stmt.on_conflict_do_update(
                index_elements=["symbol", "timeframe", "timestamp"],
                set_=update_cols,
            )
            with self.engine.begin() as conn:
                conn.execute(stmt)
            total += len(chunk)
        return total

    def get_watermarks(
        self,
        symbols: Optional[Iterable[str]] = None,
        timeframe: str = "1h",
    ) -> pd.DataFrame:
        """Latest stored bar per symbol for a timeframe: columns (symbol, max_timestamp)."""
        from sqlalchemy import func, select

        with self.engine.connect() as conn:
            stmt = (
                select(
                    MarketData.symbol.label("symbol"),
                    func.max(MarketData.timestamp).label("max_timestamp"),
                )
                .where(MarketData.timeframe == timeframe)
                .group_by(MarketData.symbol)
            )
            if symbols:
                normalized = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
                if normalized:
                    stmt = stmt.where(MarketData.symbol.in_(normalized))
            df = pd.read_sql(stmt, conn)

        if not df.empty:
            df["max_timestamp"] = pd.to_datetime(df["max_timestamp"], utc=True)
        return df

    # ------------------------------------------------------------------
    # Instruments and index membership
    # ------------------------------------------------------------------

    def upsert_instruments(self, records: Iterable[dict]) -> int:
        """Insert or update instrument metadata keyed by symbol."""
        records = list(records)
        if not records:
            return 0

        updatable = ("name", "exchange", "sector", "currency", "kind", "active", "first_seen", "last_seen")
        with self.Session() as session:
            for record in records:
                symbol = str(record["symbol"]).strip().upper()
                existing = session.get(Instrument, symbol)
                if existing is None:
                    existing = Instrument(symbol=symbol)
                    session.add(existing)
                for key in updatable:
                    if key in record and record[key] is not None:
                        setattr(existing, key, record[key])
            session.commit()
        return len(records)

    def get_instruments(
        self,
        symbols: Optional[Iterable[str]] = None,
        kind: Optional[str] = None,
        active: Optional[bool] = None,
    ) -> pd.DataFrame:
        from sqlalchemy import select

        with self.engine.connect() as conn:
            stmt = select(Instrument)
            if symbols:
                normalized = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
                if normalized:
                    stmt = stmt.where(Instrument.symbol.in_(normalized))
            if kind:
                stmt = stmt.where(Instrument.kind == kind)
            if active is not None:
                stmt = stmt.where(Instrument.active.is_(active))
            df = pd.read_sql(stmt.order_by(Instrument.symbol.asc()), conn)
        return df

    def get_open_memberships(self, index_symbol: str) -> pd.DataFrame:
        from sqlalchemy import select

        with self.engine.connect() as conn:
            stmt = (
                select(IndexMembership)
                .where(
                    IndexMembership.index_symbol == index_symbol,
                    IndexMembership.effective_to.is_(None),
                )
                .order_by(IndexMembership.symbol.asc())
            )
            return pd.read_sql(stmt, conn)

    def get_membership_intervals(self, index_symbol: str) -> pd.DataFrame:
        """All membership rows (open and closed) for point-in-time reconstruction."""
        from sqlalchemy import select

        with self.engine.connect() as conn:
            stmt = (
                select(IndexMembership)
                .where(IndexMembership.index_symbol == index_symbol)
                .order_by(IndexMembership.symbol.asc(), IndexMembership.effective_from.asc())
            )
            df = pd.read_sql(stmt, conn)

        if not df.empty:
            df["effective_from"] = pd.to_datetime(df["effective_from"])
            df["effective_to"] = pd.to_datetime(df["effective_to"])
        return df

    def get_members_asof(self, index_symbol: str, asof: date) -> List[str]:
        from sqlalchemy import or_, select

        asof = pd.to_datetime(asof).date()
        with self.engine.connect() as conn:
            stmt = (
                select(IndexMembership.symbol)
                .where(
                    IndexMembership.index_symbol == index_symbol,
                    IndexMembership.effective_from <= asof,
                    or_(
                        IndexMembership.effective_to.is_(None),
                        IndexMembership.effective_to > asof,
                    ),
                )
                .distinct()
                .order_by(IndexMembership.symbol.asc())
            )
            rows = conn.execute(stmt).fetchall()
        return [str(row[0]).upper() for row in rows]

    def open_membership(
        self,
        index_symbol: str,
        symbol: str,
        effective_from: date,
        source: str = "manual",
    ) -> None:
        symbol = str(symbol).strip().upper()
        effective_from = pd.to_datetime(effective_from).date()
        with self.Session() as session:
            existing = (
                session.query(IndexMembership)
                .filter(
                    IndexMembership.index_symbol == index_symbol,
                    IndexMembership.symbol == symbol,
                    IndexMembership.effective_to.is_(None),
                )
                .first()
            )
            if existing is not None:
                return
            session.add(
                IndexMembership(
                    index_symbol=index_symbol,
                    symbol=symbol,
                    effective_from=effective_from,
                    effective_to=None,
                    source=source,
                )
            )
            session.commit()

    def close_membership(
        self,
        index_symbol: str,
        symbol: str,
        effective_to: date,
    ) -> None:
        symbol = str(symbol).strip().upper()
        effective_to = pd.to_datetime(effective_to).date()
        with self.Session() as session:
            open_rows = (
                session.query(IndexMembership)
                .filter(
                    IndexMembership.index_symbol == index_symbol,
                    IndexMembership.symbol == symbol,
                    IndexMembership.effective_to.is_(None),
                )
                .all()
            )
            for row in open_rows:
                row.effective_to = effective_to
            if open_rows:
                session.commit()

    # ------------------------------------------------------------------
    # Market data reads
    # ------------------------------------------------------------------

    def get_latest_data(
        self,
        symbol: str,
        limit_rows: int = 500,
        exchange: Optional[str] = None,
        timeframe: Optional[str] = None,
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
            if timeframe:
                stmt = stmt.where(MarketData.timeframe == timeframe)
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
            dedupe_columns = ["timestamp"]
            if timeframe is None and "timeframe" in df.columns:
                dedupe_columns.append("timeframe")
            df = df.sort_values(sort_columns).drop_duplicates(subset=dedupe_columns, keep="last")
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
        timeframe: Optional[str] = None,
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
            if timeframe:
                stmt = stmt.where(MarketData.timeframe == timeframe)
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

    def get_available_symbols(
        self,
        exchange: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> list[str]:
        from sqlalchemy import select

        with self.engine.connect() as conn:
            stmt = select(MarketData.symbol).distinct()
            if exchange:
                stmt = stmt.where(MarketData.exchange == exchange)
            if timeframe:
                stmt = stmt.where(MarketData.timeframe == timeframe)
            rows = conn.execute(stmt.order_by(MarketData.symbol.asc())).fetchall()
        return [str(row[0]).upper() for row in rows]

    def get_market_coverage(
        self,
        symbols: Optional[Iterable[str]] = None,
        exchange: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> pd.DataFrame:
        from sqlalchemy import func, select

        with self.engine.connect() as conn:
            stmt = select(
                MarketData.symbol.label("symbol"),
                MarketData.exchange.label("exchange"),
                MarketData.timeframe.label("timeframe"),
                func.count().label("row_count"),
                func.min(MarketData.timestamp).label("first_timestamp"),
                func.max(MarketData.timestamp).label("last_timestamp"),
            )
            if symbols:
                normalized_symbols = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
                if normalized_symbols:
                    stmt = stmt.where(MarketData.symbol.in_(normalized_symbols))
            if exchange:
                stmt = stmt.where(MarketData.exchange == exchange)
            if timeframe:
                stmt = stmt.where(MarketData.timeframe == timeframe)
            stmt = stmt.group_by(MarketData.symbol, MarketData.exchange, MarketData.timeframe)
            df = pd.read_sql(stmt, conn)

        if df.empty:
            return df

        df["first_timestamp"] = pd.to_datetime(df["first_timestamp"], utc=True)
        df["last_timestamp"] = pd.to_datetime(df["last_timestamp"], utc=True)
        return df.sort_values(["timeframe", "symbol"]).reset_index(drop=True)

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
                    PredictionResults.timeframe == record.get("timeframe", "1h"),
                    PredictionResults.prediction_timestamp == prediction_timestamp,
                ).first()

                values = {
                    "symbol": record["symbol"],
                    "timeframe": record.get("timeframe", "1h"),
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
        timeframe: Optional[str] = None,
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
            if timeframe:
                stmt = stmt.where(PredictionResults.timeframe == timeframe)
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
        timeframe: Optional[str] = None,
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
            if timeframe:
                stmt = stmt.where(PredictionResults.timeframe == timeframe)
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

    def get_prediction_coverage(
        self,
        symbols: Optional[Iterable[str]] = None,
        timeframe: Optional[str] = None,
    ) -> pd.DataFrame:
        from sqlalchemy import case, func, select

        with self.engine.connect() as conn:
            stmt = select(
                PredictionResults.symbol.label("symbol"),
                PredictionResults.timeframe.label("timeframe"),
                func.count().label("prediction_count"),
                func.sum(case((PredictionResults.actual_price.is_not(None), 1), else_=0)).label("evaluated_count"),
                func.sum(case((PredictionResults.actual_price.is_(None), 1), else_=0)).label("pending_actual_count"),
                func.min(PredictionResults.prediction_timestamp).label("first_prediction_timestamp"),
                func.max(PredictionResults.prediction_timestamp).label("last_prediction_timestamp"),
            )
            if symbols:
                normalized_symbols = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
                if normalized_symbols:
                    stmt = stmt.where(PredictionResults.symbol.in_(normalized_symbols))
            if timeframe:
                stmt = stmt.where(PredictionResults.timeframe == timeframe)
            stmt = stmt.group_by(PredictionResults.symbol, PredictionResults.timeframe).order_by(
                PredictionResults.symbol.asc(),
                PredictionResults.timeframe.asc(),
            )
            df = pd.read_sql(stmt, conn)

        if df.empty:
            return df

        df["first_prediction_timestamp"] = pd.to_datetime(df["first_prediction_timestamp"], utc=True)
        df["last_prediction_timestamp"] = pd.to_datetime(df["last_prediction_timestamp"], utc=True)
        df["coverage_pct"] = (
            (df["evaluated_count"] / df["prediction_count"].replace(0, pd.NA)) * 100.0
        ).fillna(0.0)
        return df.reset_index(drop=True)

    def sync_prediction_actuals(
        self,
        symbols: Optional[Iterable[str]] = None,
        timeframe: Optional[str] = None,
    ) -> int:
        """Fill actual prices for predictions once matching market bars are available."""
        with self.Session() as session:
            pending_query = session.query(PredictionResults).filter(
                PredictionResults.actual_price.is_(None)
            )
            if symbols:
                pending_query = pending_query.filter(PredictionResults.symbol.in_(list(symbols)))
            if timeframe:
                pending_query = pending_query.filter(PredictionResults.timeframe == timeframe)

            pending_predictions = pending_query.all()
            updated = 0
            for prediction in pending_predictions:
                market_row = (
                    session.query(MarketData)
                    .filter(
                        MarketData.symbol == prediction.symbol,
                        MarketData.timeframe == prediction.timeframe,
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

    def insert_monitor_event(self, record: dict) -> int:
        with self.Session() as session:
            event = ModelMonitorEvent(
                symbol=record["symbol"],
                timeframe=record.get("timeframe", "1h"),
                prediction_timestamp=(
                    pd.to_datetime(record["prediction_timestamp"], utc=True).to_pydatetime()
                    if record.get("prediction_timestamp") is not None
                    else None
                ),
                observed_accuracy_pct=record.get("observed_accuracy_pct"),
                observed_mape=record.get("observed_mape"),
                degradation_streak=int(record.get("degradation_streak", 0)),
                action=record.get("action", "observed"),
                model_version=record.get("model_version"),
                note=record.get("note"),
            )
            session.add(event)
            session.commit()
            return int(event.id)

    def get_monitor_history(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        limit_rows: int = 100,
    ) -> pd.DataFrame:
        from sqlalchemy import desc, select

        with self.engine.connect() as conn:
            stmt = select(ModelMonitorEvent).order_by(desc(ModelMonitorEvent.created_at)).limit(limit_rows)
            if symbol:
                stmt = stmt.where(ModelMonitorEvent.symbol == symbol.upper())
            if timeframe:
                stmt = stmt.where(ModelMonitorEvent.timeframe == timeframe)
            df = pd.read_sql(stmt, conn)

        if df.empty:
            return df

        if "prediction_timestamp" in df.columns:
            df["prediction_timestamp"] = pd.to_datetime(df["prediction_timestamp"], utc=True)
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        return df.sort_values("created_at", ascending=False).reset_index(drop=True)
