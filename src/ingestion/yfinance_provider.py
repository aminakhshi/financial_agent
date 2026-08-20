"""yfinance implementation of MarketDataProvider.

Behavior notes:
- Prices are fetched with auto_adjust=True so stored history is split- AND
  dividend-adjusted. Switching an existing raw-price database over requires a
  full daily re-backfill (see docs/operations).
- Intraday intervals are capped upstream by Yahoo (~730 days); requests are
  clamped so the cap is explicit instead of a silent empty response.
"""

import random
import time
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional

import pandas as pd
import yfinance as yf

try:
    from loguru import logger
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

from ingestion.provider import (
    BAR_COLUMNS,
    FetchReport,
    FetchResult,
    MarketDataProvider,
    RateLimitError,
)

# Yahoo serves at most ~730 days of hourly data; stay one day inside the edge.
INTRADAY_MAX_LOOKBACK = timedelta(days=729)


def _is_rate_limit_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return "ratelimit" in name or "rate limit" in message or "429" in message or "too many requests" in message


class YFinanceProvider(MarketDataProvider):
    name = "yfinance"

    def __init__(
        self,
        batch_size: int = 25,
        inter_batch_delay_s: float = 1.0,
        max_retries: int = 3,
        backoff_base_s: float = 2.0,
    ):
        self.batch_size = max(int(batch_size), 1)
        self.inter_batch_delay_s = float(inter_batch_delay_s)
        self.max_retries = max(int(max_retries), 1)
        self.backoff_base_s = float(backoff_base_s)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_symbols(symbols: Iterable[str]) -> List[str]:
        seen = set()
        normalized: List[str] = []
        for symbol in symbols:
            cleaned = str(symbol).strip().upper()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            normalized.append(cleaned)
        return normalized

    def max_lookback(self, interval: str) -> Optional[timedelta]:
        if interval.endswith(("m", "h")):
            return INTRADAY_MAX_LOOKBACK
        return None

    def _clamp_start(self, interval: str, start: Optional[str]) -> Optional[str]:
        lookback = self.max_lookback(interval)
        if lookback is None:
            return start
        earliest = datetime.now(timezone.utc) - lookback
        if start is None:
            return earliest.strftime("%Y-%m-%d")
        requested = pd.to_datetime(start, utc=True)
        if requested < earliest:
            logger.warning(
                f"{self.name}: {interval} data older than {earliest.date()} is unavailable; "
                f"clamping requested start {start}."
            )
            return earliest.strftime("%Y-%m-%d")
        return start

    def _history_request_kwargs(
        self,
        period: Optional[str],
        interval: str,
        start: Optional[str],
        end: Optional[str],
    ) -> dict:
        kwargs = {
            "interval": interval,
            "auto_adjust": True,
            "actions": False,
        }
        if start or end:
            kwargs["start"] = self._clamp_start(interval, start)
            if end:
                kwargs["end"] = end
        else:
            kwargs["period"] = period or "1mo"
        return kwargs

    @staticmethod
    def _format_history_frame(data: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
        if data is None or data.empty:
            return pd.DataFrame()

        frame = data.copy().reset_index()
        timestamp_column = None
        for candidate in ("Datetime", "Date"):
            if candidate in frame.columns:
                timestamp_column = candidate
                break
        if timestamp_column is None:
            timestamp_column = frame.columns[0]

        rename_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        frame = frame.rename(columns=rename_map)
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(frame.columns):
            return pd.DataFrame()

        frame["symbol"] = symbol
        frame["timestamp"] = pd.to_datetime(frame[timestamp_column], utc=True)
        frame["timeframe"] = interval
        frame = frame[BAR_COLUMNS]

        # yfinance often returns pre-listing rows as all-NaN when a wide start
        # date is requested for newer symbols; drop them so backfills start at
        # the first real session.
        frame = frame.dropna(subset=["open", "high", "low", "close"])
        frame = frame[frame["timestamp"].notna()]
        if frame.empty:
            return pd.DataFrame()

        frame["volume"] = frame["volume"].fillna(0.0)
        numeric_columns = ["open", "high", "low", "close", "volume"]
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=numeric_columns)
        return frame.reset_index(drop=True)

    def _with_retries(self, description: str, func):
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return func()
            except Exception as exc:  # noqa: BLE001 - provider errors are heterogeneous
                last_exc = exc
                rate_limited = _is_rate_limit_error(exc)
                if attempt == self.max_retries - 1:
                    break
                delay = self.backoff_base_s * (2 ** attempt) + random.uniform(0.0, 1.0)
                if rate_limited:
                    delay = max(delay, 30.0)
                logger.warning(
                    f"{self.name}: {description} failed (attempt {attempt + 1}/{self.max_retries}): "
                    f"{exc}. Retrying in {delay:.1f}s."
                )
                time.sleep(delay)
        if last_exc is not None and _is_rate_limit_error(last_exc):
            raise RateLimitError(str(last_exc)) from last_exc
        raise last_exc  # type: ignore[misc]

    def _download_batch(
        self,
        batch: List[str],
        period: Optional[str],
        interval: str,
        start: Optional[str],
        end: Optional[str],
    ) -> pd.DataFrame:
        kwargs = self._history_request_kwargs(period=period, interval=interval, start=start, end=end)
        return yf.download(
            tickers=batch,
            group_by="ticker",
            progress=False,
            threads=True,
            **kwargs,
        )

    def _fetch_single_symbol(
        self,
        symbol: str,
        period: Optional[str],
        interval: str,
        start: Optional[str],
        end: Optional[str],
    ) -> pd.DataFrame:
        kwargs = self._history_request_kwargs(period=period, interval=interval, start=start, end=end)
        data = yf.Ticker(symbol).history(**kwargs)
        return self._format_history_frame(data, symbol, interval)

    # ------------------------------------------------------------------
    # MarketDataProvider interface
    # ------------------------------------------------------------------

    def fetch_bars(
        self,
        symbols,
        interval: str = "1h",
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: Optional[str] = None,
    ) -> FetchResult:
        normalized_symbols = self._normalize_symbols(symbols)
        if not normalized_symbols:
            return FetchResult.empty_result()

        report = FetchReport()
        all_data: List[pd.DataFrame] = []

        for index in range(0, len(normalized_symbols), self.batch_size):
            batch = normalized_symbols[index:index + self.batch_size]
            batch_frames: dict = {}
            try:
                data = self._with_retries(
                    f"batch download {batch[0]}..{batch[-1]}",
                    lambda: self._download_batch(batch, period=period, interval=interval, start=start, end=end),
                )
                if isinstance(data.columns, pd.MultiIndex):
                    available_symbols = set(data.columns.get_level_values(0))
                    for symbol in batch:
                        if symbol not in available_symbols:
                            continue
                        formatted = self._format_history_frame(data[symbol], symbol, interval)
                        if not formatted.empty:
                            batch_frames[symbol] = formatted
                elif len(batch) >= 1:
                    formatted = self._format_history_frame(data, batch[0], interval)
                    if not formatted.empty:
                        batch_frames[batch[0]] = formatted
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"{self.name}: batch download failed for {', '.join(batch)}: {exc}")

            # Per-symbol fallback for anything the batch call did not produce.
            for symbol in batch:
                if symbol in batch_frames:
                    continue
                try:
                    formatted = self._with_retries(
                        f"single download {symbol}",
                        lambda s=symbol: self._fetch_single_symbol(
                            s, period=period, interval=interval, start=start, end=end
                        ),
                    )
                    if formatted.empty:
                        report.empty.append(symbol)
                    else:
                        batch_frames[symbol] = formatted
                except Exception as exc:  # noqa: BLE001
                    report.failed[symbol] = str(exc)

            for symbol, frame in batch_frames.items():
                report.ok.append(symbol)
                all_data.append(frame)

            if index + self.batch_size < len(normalized_symbols):
                time.sleep(self.inter_batch_delay_s)

        frame = (
            pd.concat(all_data, ignore_index=True)
            if all_data
            else pd.DataFrame(columns=BAR_COLUMNS)
        )
        return FetchResult(frame=frame, report=report)

    def fetch_instrument_info(self, symbols) -> pd.DataFrame:
        normalized_symbols = self._normalize_symbols(symbols)
        records = []
        for symbol in normalized_symbols:
            info = {}
            try:
                info = yf.Ticker(symbol).get_info() or {}
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"{self.name}: instrument info failed for {symbol}: {exc}")
            records.append(
                {
                    "symbol": symbol,
                    "name": info.get("longName") or info.get("shortName"),
                    "exchange": info.get("exchange"),
                    "sector": info.get("sector"),
                    "currency": info.get("currency"),
                }
            )
            time.sleep(min(self.inter_batch_delay_s, 0.5))
        return pd.DataFrame(records, columns=["symbol", "name", "exchange", "sector", "currency"])
