"""Deprecated thin shim over ingestion.yfinance_provider.YFinanceProvider.

The LLM-agent wrapper, Alpha Vantage stub, and in-process scheduler that used
to live here were removed: data collection is deterministic and now belongs to
the `ingestion` package (see ingestion.service.IngestionService). This class
remains so existing wiring (`main.setup_agents`, api dependencies) keeps
working; new code should use IngestionService directly.
"""

from typing import Optional

import pandas as pd

from ingestion.yfinance_provider import YFinanceProvider


class DataCollectorAgent:
    def __init__(self, config, db_manager):
        self.config = config
        self.db_manager = db_manager
        ingestion_config = config.get("INGESTION_CONFIG", {})
        market_config = config.get("MARKET_CONFIG", {})
        self.provider = YFinanceProvider(
            batch_size=ingestion_config.get(
                "batch_size", market_config.get("download_batch_size", 25)
            ),
            inter_batch_delay_s=ingestion_config.get("inter_batch_delay_s", 1.0),
            max_retries=ingestion_config.get("max_retries", 3),
        )

    def fetch_yfinance_data(
        self,
        symbols,
        period="1d",
        interval="1h",
        start: Optional[str] = None,
        end: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """Deprecated: use IngestionService.collect / YFinanceProvider.fetch_bars."""
        if batch_size:
            self.provider.batch_size = max(int(batch_size), 1)
        result = self.provider.fetch_bars(
            symbols, interval=interval, start=start, end=end, period=period if not (start or end) else None
        )
        return result.frame

    def store_market_data(self, df: pd.DataFrame, exchange: Optional[str] = None):
        """Deprecated: exchange labels are no longer part of bar identity."""
        if df is None or df.empty:
            return "No data to store"
        stored = self.db_manager.insert_market_data(df, exchange)
        return f"Successfully stored {stored} records"
