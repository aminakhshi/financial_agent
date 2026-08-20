"""Baseline: the legacy pre-Alembic schema.

Pre-existing databases are stamped at this revision (DatabaseManager.run_migrations
does that automatically) so their data is migrated in place by later revisions.
Fresh databases replay this and every later revision.

Revision ID: 0001
Revises:
Create Date: 2026-08-20

"""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())

    if "market_data" not in existing:
        op.create_table(
            "market_data",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("symbol", sa.String(10), nullable=False),
            sa.Column("exchange", sa.String(10), nullable=False),
            sa.Column("timeframe", sa.String(10), nullable=False, server_default="1h"),
            sa.Column("timestamp", sa.DateTime(), nullable=False),
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
            sa.Column("created_at", sa.DateTime()),
            sa.UniqueConstraint("symbol", "exchange", "timestamp", name="uq_marketdata_symbol_exch_ts"),
        )
        op.create_index("idx_symbol_timestamp", "market_data", ["symbol", "timestamp"])
        op.create_index("idx_exchange_timestamp", "market_data", ["exchange", "timestamp"])
        op.create_index("idx_timestamp", "market_data", ["timestamp"])
        op.create_index("idx_symbol_timeframe_timestamp", "market_data", ["symbol", "timeframe", "timestamp"])
        op.create_index("idx_exchange_timeframe_timestamp", "market_data", ["exchange", "timeframe", "timestamp"])

    if "prediction_results" not in existing:
        op.create_table(
            "prediction_results",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("symbol", sa.String(10), nullable=False),
            sa.Column("timeframe", sa.String(10), nullable=False, server_default="1h"),
            sa.Column("prediction_timestamp", sa.DateTime(), nullable=False),
            sa.Column("predicted_price", sa.Float(), nullable=False),
            sa.Column("confidence_score", sa.Float(), nullable=False),
            sa.Column("model_version", sa.String(50), nullable=False),
            sa.Column("actual_price", sa.Float()),
            sa.Column("created_at", sa.DateTime()),
            sa.UniqueConstraint(
                "symbol", "timeframe", "prediction_timestamp", name="uq_prediction_symbol_timeframe_ts"
            ),
        )
        op.create_index("idx_prediction_symbol_timestamp", "prediction_results", ["symbol", "prediction_timestamp"])
        op.create_index("idx_prediction_timestamp", "prediction_results", ["prediction_timestamp"])
        op.create_index(
            "idx_prediction_symbol_timeframe_timestamp",
            "prediction_results",
            ["symbol", "timeframe", "prediction_timestamp"],
        )

    if "model_monitor_events" not in existing:
        op.create_table(
            "model_monitor_events",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("symbol", sa.String(10), nullable=False),
            sa.Column("timeframe", sa.String(10), nullable=False),
            sa.Column("prediction_timestamp", sa.DateTime()),
            sa.Column("observed_accuracy_pct", sa.Float()),
            sa.Column("observed_mape", sa.Float()),
            sa.Column("degradation_streak", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("action", sa.String(50), nullable=False),
            sa.Column("model_version", sa.String(50)),
            sa.Column("note", sa.String(500)),
            sa.Column("created_at", sa.DateTime()),
        )
        op.create_index(
            "idx_monitor_symbol_timeframe_created",
            "model_monitor_events",
            ["symbol", "timeframe", "created_at"],
        )


def downgrade():
    raise RuntimeError("The baseline revision cannot be downgraded.")
