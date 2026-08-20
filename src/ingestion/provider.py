"""Market data provider abstraction.

Every provider returns bar frames with the same contract so the rest of the
pipeline never needs provider-specific handling:

columns: symbol, timestamp (tz-aware UTC), open, high, low, close, volume, timeframe
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional

import pandas as pd

BAR_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume", "timeframe"]


class ProviderError(Exception):
    """Base error for market data providers."""


class RateLimitError(ProviderError):
    """The provider throttled the request; retry later."""


@dataclass
class FetchReport:
    """Per-symbol outcome of a fetch call. Nothing is silently swallowed."""

    ok: List[str] = field(default_factory=list)
    empty: List[str] = field(default_factory=list)
    failed: Dict[str, str] = field(default_factory=dict)

    def merge(self, other: "FetchReport") -> None:
        self.ok.extend(s for s in other.ok if s not in self.ok)
        self.empty.extend(s for s in other.empty if s not in self.empty)
        self.failed.update(other.failed)
        # A symbol that eventually succeeded is no longer empty/failed.
        self.empty = [s for s in self.empty if s not in self.ok]
        for symbol in list(self.failed):
            if symbol in self.ok:
                del self.failed[symbol]

    def to_dict(self) -> dict:
        return {"ok": sorted(self.ok), "empty": sorted(self.empty), "failed": dict(self.failed)}


@dataclass
class FetchResult:
    frame: pd.DataFrame
    report: FetchReport

    @classmethod
    def empty_result(cls) -> "FetchResult":
        return cls(frame=pd.DataFrame(columns=BAR_COLUMNS), report=FetchReport())


class MarketDataProvider(ABC):
    """Interface for bar-data providers (yfinance today; Polygon/AlphaVantage later)."""

    name: str = "abstract"

    @abstractmethod
    def fetch_bars(
        self,
        symbols,
        interval: str = "1h",
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: Optional[str] = None,
    ) -> FetchResult:
        """Fetch bars for symbols. `start`/`end` take precedence over `period`."""

    @abstractmethod
    def fetch_instrument_info(self, symbols) -> pd.DataFrame:
        """Best-effort metadata: columns symbol, name, exchange, sector, currency."""

    def max_lookback(self, interval: str) -> Optional[timedelta]:
        """How far back this provider can serve the interval; None = unlimited."""
        return None
