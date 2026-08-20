"""Point-in-time index membership maintenance.

Price history is never deleted: when a symbol leaves the index its membership
row is closed (effective_to set) and the instrument is marked inactive, so
historical analysis and point-in-time aggregates keep working.
"""

import io
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

try:
    from loguru import logger
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

from data.symbol_universe import load_symbol_file

WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# Abort a refresh whose scraped list size is implausible (protects against
# scrape breakage mass-closing memberships).
MIN_PLAUSIBLE_SIZE = 480
MAX_PLAUSIBLE_SIZE = 520


def _normalize_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "-")


class MembershipService:
    def __init__(self, db_manager, index_symbol: str = "^GSPC", seed_path: Optional[Path] = None):
        self.db = db_manager
        self.index_symbol = index_symbol
        self.seed_path = seed_path or (
            Path(__file__).resolve().parent.parent / "data" / "static" / "sp500_symbols.txt"
        )

    # ------------------------------------------------------------------

    def current_members(self) -> List[str]:
        rows = self.db.get_open_memberships(self.index_symbol)
        if rows.empty:
            return []
        return sorted({_normalize_symbol(symbol) for symbol in rows["symbol"]})

    def seed(self, effective_from: Optional[date] = None) -> dict:
        """Idempotently open membership rows for every symbol in the seed file.

        The default effective_from reaches back to the daily backfill horizon:
        current members are assumed to have been members through stored history
        (a documented survivorship-bias approximation until real historical
        membership changes are imported).
        """
        symbols = load_symbol_file(self.seed_path)
        if not symbols:
            return {
                "status": "no_data",
                "message": f"Seed file {self.seed_path} is missing or empty.",
                "opened": 0,
            }

        if effective_from is None:
            import os

            seed_env = os.getenv("SP500_SEED_EFFECTIVE_FROM", "").strip()
            effective_from = date.fromisoformat(seed_env) if seed_env else date(1991, 1, 1)
        effective_from = pd.to_datetime(effective_from).date()
        existing = set(self.current_members())
        opened = 0
        for symbol in symbols:
            if symbol in existing:
                continue
            self.db.open_membership(self.index_symbol, symbol, effective_from, source="seed_file")
            self.db.upsert_instruments([{"symbol": symbol, "kind": "equity", "active": True}])
            opened += 1
        return {
            "status": "ok",
            "index_symbol": self.index_symbol,
            "seed_file": str(self.seed_path),
            "seed_size": len(symbols),
            "opened": opened,
            "message": f"Seeded {opened} new membership rows ({len(symbols)} symbols in file).",
        }

    # ------------------------------------------------------------------

    def fetch_current_constituents(self) -> pd.DataFrame:
        """Current constituents (symbol, name, sector) from Wikipedia."""
        response = requests.get(
            WIKIPEDIA_SP500_URL,
            headers={"User-Agent": "financial-agent/1.0 (market data ingestion)"},
            timeout=30,
        )
        response.raise_for_status()
        tables = pd.read_html(io.StringIO(response.text))
        for table in tables:
            columns = {str(column).strip().lower() for column in table.columns}
            if "symbol" in columns and any("gics sector" in column for column in columns):
                table = table.rename(
                    columns={column: str(column).strip().lower() for column in table.columns}
                )
                frame = pd.DataFrame(
                    {
                        "symbol": table["symbol"].map(_normalize_symbol),
                        "name": table.get("security"),
                        "sector": table.get("gics sector"),
                    }
                )
                frame = frame.drop_duplicates(subset=["symbol"]).reset_index(drop=True)
                return frame
        raise ValueError("Could not locate the constituents table on the Wikipedia page.")

    def refresh(self, asof: Optional[date] = None) -> dict:
        """Diff the live constituent list against open memberships and apply it.

        Failure of the remote source leaves memberships untouched (safe direction).
        """
        asof = asof or datetime.now(timezone.utc).date()
        try:
            constituents = self.fetch_current_constituents()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Membership refresh source failed: {exc}. Memberships left unchanged.")
            return {
                "status": "source_unavailable",
                "message": f"Constituent source failed ({exc}); memberships left unchanged.",
                "added": [],
                "removed": [],
            }

        if not (MIN_PLAUSIBLE_SIZE <= len(constituents) <= MAX_PLAUSIBLE_SIZE):
            message = (
                f"Fetched constituent list has {len(constituents)} symbols "
                f"(expected {MIN_PLAUSIBLE_SIZE}-{MAX_PLAUSIBLE_SIZE}); aborting refresh."
            )
            logger.error(message)
            return {"status": "aborted", "message": message, "added": [], "removed": []}

        fetched = set(constituents["symbol"])
        open_members = set(self.current_members())

        added = sorted(fetched - open_members)
        removed = sorted(open_members - fetched)

        for symbol in removed:
            self.db.close_membership(self.index_symbol, symbol, effective_to=asof)
            self.db.upsert_instruments([{"symbol": symbol, "active": False}])
        for symbol in added:
            self.db.open_membership(self.index_symbol, symbol, effective_from=asof, source="wikipedia_refresh")

        # Refresh instrument metadata (name + GICS sector) for all constituents;
        # the Wikipedia table is more reliable than per-ticker provider info.
        metadata_records = [
            {
                "symbol": row["symbol"],
                "name": row["name"] if pd.notna(row["name"]) else None,
                "sector": row["sector"] if pd.notna(row["sector"]) else None,
                "kind": "equity",
                "active": True,
            }
            for _, row in constituents.iterrows()
        ]
        self.db.upsert_instruments(metadata_records)

        message = (
            f"Membership refresh complete: {len(added)} added, {len(removed)} removed, "
            f"{len(fetched)} current members."
        )
        logger.info(message)
        return {
            "status": "ok",
            "index_symbol": self.index_symbol,
            "asof": asof.isoformat(),
            "added": added,
            "removed": removed,
            "current_size": len(fetched),
            "message": message,
        }

    def sector_map(self) -> Dict[str, str]:
        """symbol -> GICS sector for known instruments (empty until refresh runs)."""
        instruments = self.db.get_instruments(kind="equity")
        if instruments.empty or "sector" not in instruments.columns:
            return {}
        with_sector = instruments.dropna(subset=["sector"])
        return dict(zip(with_sector["symbol"].str.upper(), with_sector["sector"]))
