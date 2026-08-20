"""Add instrument dimension and point-in-time index membership tables, with seeds.

Seeds:
- instruments: one row per distinct symbol already present in market_data (kind
  inferred; first_seen/last_seen from stored bars) plus every symbol in the
  static S&P 500 seed file.
- index_membership: '^GSPC' membership seeded from src/data/static/sp500_symbols.txt.
  Symbols in the file get an OPEN membership row effective from
  SP500_SEED_EFFECTIVE_FROM (env var, default: 1991-01-01 to match the daily
  backfill horizon — i.e. current members are assumed to have been members
  through stored history, a documented survivorship-bias approximation until
  real historical changes are imported). Equity symbols that exist in
  market_data but not in the file get a CLOSED row ending today (their history
  stays and participates in point-in-time analysis; nothing is ever deleted).

The seed date is an approximation of true membership history; the schema supports
importing real historical changes later.

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-20

"""
import os
from datetime import date, datetime, timezone
from pathlib import Path

from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

KNOWN_ETFS = {
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
    "SPY", "QQQ", "DIA", "IWM", "VOO", "IVV", "VTI",
}


def _load_seed_symbols():
    path = Path(__file__).resolve().parents[2] / "static" / "sp500_symbols.txt"
    if not path.exists():
        return []
    seen, symbols = set(), []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        symbol = raw_line.strip().upper()
        if not symbol or symbol.startswith("#"):
            continue
        normalized = symbol.replace(".", "-")
        if normalized not in seen:
            seen.add(normalized)
            symbols.append(normalized)
    return symbols


def _infer_kind(symbol: str) -> str:
    if symbol.startswith("^"):
        return "index"
    if symbol.startswith("SECT_"):
        return "synthetic"
    if symbol in KNOWN_ETFS:
        return "etf"
    return "equity"


def upgrade():
    instruments = op.create_table(
        "instruments",
        sa.Column("symbol", sa.String(20), primary_key=True),
        sa.Column("name", sa.String(200)),
        sa.Column("exchange", sa.String(20)),
        sa.Column("sector", sa.String(50)),
        sa.Column("currency", sa.String(10)),
        sa.Column("kind", sa.String(20), nullable=False, server_default="equity"),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("first_seen", sa.DateTime(timezone=True)),
        sa.Column("last_seen", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True)),
        sa.Column("updated_at", sa.DateTime(timezone=True)),
    )
    membership = op.create_table(
        "index_membership",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("index_symbol", sa.String(20), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("effective_from", sa.Date(), nullable=False),
        sa.Column("effective_to", sa.Date(), nullable=True),
        sa.Column("source", sa.String(50)),
        sa.Column("created_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("index_symbol", "symbol", "effective_from", name="uq_membership_index_symbol_from"),
    )
    op.create_index("idx_membership_index_open", "index_membership", ["index_symbol", "effective_to"])
    op.create_index("idx_membership_symbol", "index_membership", ["symbol"])

    bind = op.get_bind()
    now = datetime.now(timezone.utc)

    seed_env = os.getenv("SP500_SEED_EFFECTIVE_FROM", "").strip()
    seed_date = date.fromisoformat(seed_env) if seed_env else date(1991, 1, 1)
    today = now.date()

    stored = bind.execute(
        sa.text(
            "SELECT symbol, MIN(timestamp) AS first_ts, MAX(timestamp) AS last_ts "
            "FROM market_data GROUP BY symbol"
        )
    ).fetchall()
    stored_info = {}
    for row in stored:
        symbol = str(row[0]).strip().upper()

        def _parse(value):
            if value is None or isinstance(value, datetime):
                return value
            try:
                return datetime.fromisoformat(str(value))
            except ValueError:
                return None

        stored_info[symbol] = (_parse(row[1]), _parse(row[2]))

    seed_symbols = _load_seed_symbols()
    all_symbols = sorted(set(stored_info) | set(seed_symbols))

    instrument_rows = []
    for symbol in all_symbols:
        first_seen, last_seen = stored_info.get(symbol, (None, None))
        instrument_rows.append(
            {
                "symbol": symbol,
                "name": None,
                "exchange": None,
                "sector": None,
                "currency": None,
                "kind": _infer_kind(symbol),
                "active": True,
                "first_seen": first_seen,
                "last_seen": last_seen,
                "created_at": now,
                "updated_at": now,
            }
        )
    if instrument_rows:
        op.bulk_insert(instruments, instrument_rows)

    membership_rows = []
    seed_set = set(seed_symbols)
    for symbol in seed_symbols:
        membership_rows.append(
            {
                "index_symbol": "^GSPC",
                "symbol": symbol,
                "effective_from": seed_date,
                "effective_to": None,
                "source": "seed_file",
                "created_at": now,
            }
        )
    if seed_symbols:
        # Equities already stored but absent from the current list: assume they
        # were members historically and left the index by the time of this seed.
        for symbol, (first_seen, _last_seen) in stored_info.items():
            if symbol in seed_set or _infer_kind(symbol) != "equity":
                continue
            effective_from = first_seen.date() if first_seen is not None else seed_date
            if effective_from >= today:
                effective_from = seed_date
            membership_rows.append(
                {
                    "index_symbol": "^GSPC",
                    "symbol": symbol,
                    "effective_from": effective_from,
                    "effective_to": today,
                    "source": "seed_file",
                    "created_at": now,
                }
            )
    if membership_rows:
        op.bulk_insert(membership, membership_rows)


def downgrade():
    op.drop_index("idx_membership_symbol", table_name="index_membership")
    op.drop_index("idx_membership_index_open", table_name="index_membership")
    op.drop_table("index_membership")
    op.drop_table("instruments")
