"""Re-key market_data by (symbol, timeframe, timestamp).

- Normalizes NULL timeframes to '1h'.
- Deduplicates rows that only differed by the legacy exchange label, keeping the
  most recently written row (matches the previous upsert's last-write-wins).
- Drops the (symbol, exchange, timestamp) unique key and exchange indexes; adds
  the (symbol, timeframe, timestamp) unique key.
- Widens symbol columns to VARCHAR(20) for synthetic sector series.
- Makes exchange nullable (deprecated column).
- PostgreSQL only: converts timestamp columns to TIMESTAMPTZ (stored values are UTC).

This data migration is irreversible.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-20

"""
from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

MARKET_DATA_INDEXES = (
    ("idx_symbol_timestamp", ["symbol", "timestamp"]),
    ("idx_timestamp", ["timestamp"]),
    ("idx_symbol_timeframe_timestamp", ["symbol", "timeframe", "timestamp"]),
)


def _upgrade_postgres():
    op.execute(
        """
        DELETE FROM market_data a USING market_data b
        WHERE a.symbol = b.symbol
          AND a.timeframe = b.timeframe
          AND a.timestamp = b.timestamp
          AND (a.created_at < b.created_at
               OR (a.created_at = b.created_at AND a.id < b.id)
               OR (a.created_at IS NULL AND b.created_at IS NOT NULL))
        """
    )
    op.execute("ALTER TABLE market_data DROP CONSTRAINT IF EXISTS uq_marketdata_symbol_exch_ts")
    for index_name in (
        "idx_exchange_timestamp",
        "idx_exchange_timeframe_timestamp",
        "idx_marketdata_symbol_timeframe_timestamp",
        "idx_marketdata_exchange_timeframe_timestamp",
    ):
        op.execute(f"DROP INDEX IF EXISTS {index_name}")

    op.execute("ALTER TABLE market_data ALTER COLUMN symbol TYPE VARCHAR(20)")
    op.execute("ALTER TABLE market_data ALTER COLUMN exchange DROP NOT NULL")
    op.execute(
        "ALTER TABLE market_data ALTER COLUMN timestamp TYPE TIMESTAMPTZ USING timestamp AT TIME ZONE 'UTC'"
    )
    op.execute(
        "ALTER TABLE market_data ALTER COLUMN created_at TYPE TIMESTAMPTZ USING created_at AT TIME ZONE 'UTC'"
    )
    op.execute(
        "ALTER TABLE market_data ADD CONSTRAINT uq_marketdata_symbol_timeframe_ts "
        "UNIQUE (symbol, timeframe, timestamp)"
    )

    op.execute("ALTER TABLE prediction_results ALTER COLUMN symbol TYPE VARCHAR(20)")
    op.execute(
        "ALTER TABLE prediction_results ALTER COLUMN prediction_timestamp TYPE TIMESTAMPTZ "
        "USING prediction_timestamp AT TIME ZONE 'UTC'"
    )
    op.execute(
        "ALTER TABLE prediction_results ALTER COLUMN created_at TYPE TIMESTAMPTZ "
        "USING created_at AT TIME ZONE 'UTC'"
    )
    op.execute("ALTER TABLE model_monitor_events ALTER COLUMN symbol TYPE VARCHAR(20)")
    op.execute(
        "ALTER TABLE model_monitor_events ALTER COLUMN prediction_timestamp TYPE TIMESTAMPTZ "
        "USING prediction_timestamp AT TIME ZONE 'UTC'"
    )
    op.execute(
        "ALTER TABLE model_monitor_events ALTER COLUMN created_at TYPE TIMESTAMPTZ "
        "USING created_at AT TIME ZONE 'UTC'"
    )


def _upgrade_sqlite():
    # SQLite cannot alter constraints in place; rebuild the table deterministically.
    op.execute(
        """
        DELETE FROM market_data WHERE id NOT IN (
            SELECT MAX(id) FROM market_data GROUP BY symbol, timeframe, timestamp
        )
        """
    )
    op.execute("ALTER TABLE market_data RENAME TO market_data_old")
    op.create_table(
        "market_data",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("exchange", sa.String(10), nullable=True),
        sa.Column("timeframe", sa.String(10), nullable=False, server_default="1h"),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open_price", sa.Float(), nullable=False),
        sa.Column("high_price", sa.Float(), nullable=False),
        sa.Column("low_price", sa.Float(), nullable=False),
        sa.Column("close_price", sa.Float(), nullable=False),
        sa.Column("volume", sa.Float(), nullable=False),
        sa.Column("sma_20", sa.Float()),
        sa.Column("ema_12", sa.Float()),
        sa.Column("rsi", sa.Float()),
        sa.Column("macd", sa.Float()),
        sa.Column("macd_signal", sa.Float()),
        sa.Column("bollinger_upper", sa.Float()),
        sa.Column("bollinger_lower", sa.Float()),
        sa.Column("created_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_marketdata_symbol_timeframe_ts"),
    )
    columns = (
        "id, symbol, exchange, timeframe, timestamp, open_price, high_price, low_price, "
        "close_price, volume, sma_20, ema_12, rsi, macd, macd_signal, bollinger_upper, "
        "bollinger_lower, created_at"
    )
    op.execute(f"INSERT INTO market_data ({columns}) SELECT {columns} FROM market_data_old")
    op.execute("DROP TABLE market_data_old")
    for index_name, index_columns in MARKET_DATA_INDEXES:
        op.create_index(index_name, "market_data", index_columns)
    # SQLite ignores VARCHAR lengths and stores naive-UTC text timestamps; the
    # prediction/monitor tables need no rebuild.


def upgrade():
    bind = op.get_bind()
    op.execute("UPDATE market_data SET timeframe = '1h' WHERE timeframe IS NULL")
    if bind.dialect.name == "postgresql":
        _upgrade_postgres()
    else:
        _upgrade_sqlite()


def downgrade():
    raise RuntimeError("The market_data re-key migration is irreversible.")
