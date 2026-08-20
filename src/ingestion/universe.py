"""Named-universe resolution for collection jobs.

After seeding, the database membership table is the source of truth for index
constituents; the static symbol file is only a seed/fallback.
"""

from typing import Iterable, List, Optional

from ingestion.aggregates import sector_symbols

SUPPORTED_UNIVERSES = (
    "default", "watchlist", "indices", "sector_etfs", "etfs", "constituents",
    "sp500", "s&p500", "sp500_full", "nasdaq", "all", "full",
    "configured", "configured_all", "aggregates",
)


def _dedupe(symbols: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for symbol in symbols:
        cleaned = str(symbol).strip().upper()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


class UniverseResolver:
    def __init__(self, config: dict, db_manager):
        self.config = config
        self.db = db_manager

    @property
    def _ingestion(self) -> dict:
        return self.config.get("INGESTION_CONFIG", {})

    @property
    def _market(self) -> dict:
        return self.config.get("MARKET_CONFIG", {})

    def constituents(self) -> List[str]:
        index_symbol = self._ingestion.get("constituents_index", "^GSPC")
        try:
            rows = self.db.get_open_memberships(index_symbol)
            if not rows.empty:
                return _dedupe(rows["symbol"])
        except Exception:
            pass
        # Membership not seeded yet: fall back to the configured static list.
        return _dedupe(self._market.get("sp500_symbols", []))

    def resolve(self, universe: str = "default", symbols: Optional[Iterable[str]] = None) -> List[str]:
        if symbols:
            resolved = _dedupe(symbols)
            if resolved:
                return resolved

        name = (universe or "default").strip().lower()
        indices = self._ingestion.get("index_symbols", [])
        etfs = self._ingestion.get("sector_etf_symbols", [])
        configured_sp500 = self._market.get("sp500_symbols", [])
        configured_nasdaq = self._market.get("nasdaq_symbols", [])
        default_symbols = self._market.get("default_symbols", [])

        universe_map = {
            "default": default_symbols,
            "watchlist": default_symbols,
            "indices": indices,
            "sector_etfs": etfs,
            "etfs": etfs,
            "constituents": self.constituents,
            "sp500": self.constituents,
            "s&p500": self.constituents,
            "sp500_full": self.constituents,
            "nasdaq": configured_nasdaq,
            "all": lambda: indices + etfs + self.constituents(),
            "full": lambda: indices + etfs + self.constituents(),
            # Legacy API names.
            "configured": configured_sp500 + configured_nasdaq,
            "configured_all": configured_sp500 + configured_nasdaq,
            "aggregates": sector_symbols,
        }

        if name not in universe_map:
            raise ValueError(
                f"Unsupported universe '{name}'. Use one of: {', '.join(SUPPORTED_UNIVERSES)}."
            )

        source = universe_map[name]
        resolved = _dedupe(source() if callable(source) else source)
        if not resolved:
            raise ValueError(f"No symbols are available for universe '{name}'.")
        return resolved
