"""IngestionService: deterministic orchestration of market-data collection.

Every public method returns a structured run report (dict) and never swallows
per-symbol failures — they are collected in the report instead of failing or
hiding the whole run.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional

import pandas as pd

try:
    from loguru import logger
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

from ingestion.aggregates import SectorAggregateService
from ingestion.market_calendar import MarketCalendar
from ingestion.membership import MembershipService
from ingestion.provider import FetchReport, MarketDataProvider
from ingestion.universe import UniverseResolver
from ingestion.watermark import plan_incremental_fetch
from ingestion.yfinance_provider import YFinanceProvider


def _default_config() -> dict:
    from config.settings import INGESTION_CONFIG, MARKET_CONFIG

    return {"INGESTION_CONFIG": INGESTION_CONFIG, "MARKET_CONFIG": MARKET_CONFIG}


def _infer_kind(symbol: str, etf_symbols: Iterable[str]) -> str:
    if symbol.startswith("^"):
        return "index"
    if symbol.startswith("SECT_"):
        return "synthetic"
    if symbol in set(etf_symbols):
        return "etf"
    return "equity"


class IngestionService:
    def __init__(
        self,
        config: Optional[dict] = None,
        db_manager=None,
        provider: Optional[MarketDataProvider] = None,
        calendar: Optional[MarketCalendar] = None,
    ):
        self.config = config or _default_config()
        if "INGESTION_CONFIG" not in self.config:
            self.config = {**self.config, **_default_config()}
        self.ingestion_config = self.config["INGESTION_CONFIG"]
        self.db = db_manager
        self.provider = provider or YFinanceProvider(
            batch_size=self.ingestion_config.get("batch_size", 25),
            inter_batch_delay_s=self.ingestion_config.get("inter_batch_delay_s", 1.0),
            max_retries=self.ingestion_config.get("max_retries", 3),
        )
        self.calendar = calendar or MarketCalendar(self.ingestion_config.get("exchange_calendar", "XNYS"))
        self.universes = UniverseResolver(self.config, db_manager)
        self.membership = MembershipService(
            db_manager, index_symbol=self.ingestion_config.get("constituents_index", "^GSPC")
        )
        self.aggregates = SectorAggregateService(
            db_manager,
            index_symbol=self.ingestion_config.get("constituents_index", "^GSPC"),
            recompute_days=self.ingestion_config.get("aggregate_recompute_days", 7),
        )

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def _fetchable(self, symbols: List[str]) -> List[str]:
        """Synthetic sector series are computed, never fetched from a provider."""
        return [symbol for symbol in symbols if not symbol.startswith("SECT_")]

    def _store(self, frame: pd.DataFrame) -> Dict[str, int]:
        if frame.empty:
            return {}
        self.db.insert_market_data(frame)
        return {
            symbol: int(count)
            for symbol, count in frame.groupby("symbol").size().to_dict().items()
        }

    def _touch_instruments(self, symbols: Iterable[str]) -> None:
        now = datetime.now(timezone.utc)
        etfs = self.ingestion_config.get("sector_etf_symbols", [])
        records = [
            {"symbol": symbol, "kind": _infer_kind(symbol, etfs), "last_seen": now}
            for symbol in symbols
        ]
        if records:
            try:
                self.db.upsert_instruments(records)
            except Exception as exc:  # noqa: BLE001 - metadata must not fail collection
                logger.warning(f"Instrument touch failed: {exc}")

    def collect(
        self,
        symbols: Optional[Iterable[str]] = None,
        universe: Optional[str] = None,
        timeframe: str = "1h",
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: Optional[str] = None,
        incremental: Optional[bool] = None,
        batch_size: Optional[int] = None,
    ) -> dict:
        """Collect bars for a universe or explicit symbols.

        Default mode is incremental: only the gap since each symbol's stored
        watermark is fetched. Passing start/end/period switches to an explicit
        window (used by backfills and legacy API calls).
        """
        started_at = datetime.now(timezone.utc)
        resolved = self.universes.resolve(universe or "default", symbols)
        fetchable = self._fetchable(resolved)
        if batch_size:
            self.provider.batch_size = max(int(batch_size), 1)

        explicit_window = bool(start or end or period)
        if incremental is None:
            incremental = not explicit_window

        report = FetchReport()
        rows_by_symbol: Dict[str, int] = {}
        skipped_fresh: List[str] = []

        if incremental and not explicit_window:
            watermarks = self.db.get_watermarks(symbols=fetchable, timeframe=timeframe)
            last_complete = self.calendar.last_session_close(started_at)
            plans = plan_incremental_fetch(
                watermarks,
                fetchable,
                timeframe,
                now=started_at,
                max_lookback=self.provider.max_lookback(timeframe),
                backfill_start=self.ingestion_config.get("backfill_start", "1991-01-01"),
                last_complete_bar=last_complete,
            )
            planned = {symbol for plan in plans for symbol in plan.symbols}
            skipped_fresh = sorted(set(fetchable) - planned)
            for plan in plans:
                result = self.provider.fetch_bars(
                    plan.symbols, interval=timeframe, start=plan.start, end=plan.end
                )
                report.merge(result.report)
                for symbol, count in self._store(result.frame).items():
                    rows_by_symbol[symbol] = rows_by_symbol.get(symbol, 0) + count
        else:
            result = self.provider.fetch_bars(
                fetchable, interval=timeframe, start=start, end=end, period=period
            )
            report.merge(result.report)
            rows_by_symbol = self._store(result.frame)

        self._touch_instruments(rows_by_symbol)
        try:
            actuals_updated = self.db.sync_prediction_actuals(resolved, timeframe=timeframe)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Prediction actuals sync failed: {exc}")
            actuals_updated = 0

        rows_collected = int(sum(rows_by_symbol.values()))
        duration_s = (datetime.now(timezone.utc) - started_at).total_seconds()
        status = "ok" if rows_collected or skipped_fresh else ("no_data" if not report.failed else "failed")
        run_report = {
            "status": status,
            "job": "collect",
            "universe": (universe or ("custom" if symbols else "default")),
            "timeframe": timeframe,
            "interval": timeframe,
            "incremental": bool(incremental and not explicit_window),
            "start": start,
            "end": end,
            "period": period,
            "symbols": resolved,
            "requested_symbol_count": len(resolved),
            "stored_symbol_count": len(rows_by_symbol),
            "rows_collected": rows_collected,
            "rows_by_symbol": rows_by_symbol,
            "skipped_up_to_date": skipped_fresh,
            "fetch_report": report.to_dict(),
            "failures": report.failed,
            "actuals_updated": int(actuals_updated),
            "duration_s": round(duration_s, 2),
            "timestamp": started_at.isoformat(),
            "message": (
                f"Collected {rows_collected} {timeframe} rows for {len(rows_by_symbol)} of "
                f"{len(resolved)} symbols ({len(skipped_fresh)} already up to date, "
                f"{len(report.failed)} failed). Updated {actuals_updated} prediction actuals."
            ),
        }
        logger.info(run_report["message"])
        return run_report

    # ------------------------------------------------------------------
    # Backfills
    # ------------------------------------------------------------------

    def backfill_daily(
        self,
        symbols: Optional[Iterable[str]] = None,
        universe: str = "all",
        start: Optional[str] = None,
        end: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> dict:
        start = start or self.ingestion_config.get("backfill_start", "1991-01-01")
        report = self.collect(
            symbols=symbols,
            universe=universe,
            timeframe="1d",
            start=start,
            end=end,
            incremental=False,
            batch_size=batch_size,
        )
        report["job"] = "backfill_daily"
        report["message"] = (
            f"Daily backfill from {start} through {end or 'latest'}: stored "
            f"{report['rows_collected']} rows for {report['stored_symbol_count']} symbols."
        )
        return report

    def backfill_hourly(
        self,
        symbols: Optional[Iterable[str]] = None,
        universe: str = "all",
        end: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> dict:
        lookback = self.provider.max_lookback("1h") or timedelta(days=729)
        start = (datetime.now(timezone.utc) - lookback).strftime("%Y-%m-%d")
        report = self.collect(
            symbols=symbols,
            universe=universe,
            timeframe="1h",
            start=start,
            end=end,
            incremental=False,
            batch_size=batch_size,
        )
        report["job"] = "backfill_hourly"
        report["message"] = (
            f"Hourly backfill from {start} (provider lookback limit): stored "
            f"{report['rows_collected']} rows for {report['stored_symbol_count']} symbols."
        )
        return report

    # ------------------------------------------------------------------
    # Membership, aggregates, instruments, gap repair
    # ------------------------------------------------------------------

    def seed_membership(self, effective_from=None) -> dict:
        return self.membership.seed(effective_from=effective_from)

    def refresh_membership(self, backfill_new: bool = True) -> dict:
        report = self.membership.refresh()
        report["job"] = "membership_refresh"
        added = report.get("added", [])
        if added and backfill_new:
            logger.info(f"Backfilling {len(added)} new index members: {', '.join(added)}.")
            report["daily_backfill"] = self.backfill_daily(symbols=added)
            report["hourly_backfill"] = self.backfill_hourly(symbols=added)
            report["instrument_sync"] = self.sync_instruments(added)
        return report

    def recompute_aggregates(self, timeframe: str = "1d", full: bool = False) -> dict:
        report = self.aggregates.recompute(timeframe=timeframe, full=full)
        report["job"] = "recompute_aggregates"
        return report

    def sync_instruments(self, symbols: Optional[Iterable[str]] = None) -> dict:
        if symbols is None:
            instruments = self.db.get_instruments()
            symbols = [] if instruments.empty else list(instruments["symbol"])
        symbols = self._fetchable([str(symbol).upper() for symbol in symbols])
        if not symbols:
            return {"status": "no_data", "job": "sync_instruments", "updated": 0,
                    "message": "No symbols to sync."}
        info = self.provider.fetch_instrument_info(symbols)
        records = [
            {key: (None if pd.isna(value) else value) for key, value in record.items()}
            for record in info.to_dict(orient="records")
        ]
        etfs = self.ingestion_config.get("sector_etf_symbols", [])
        for record in records:
            record["kind"] = _infer_kind(record["symbol"], etfs)
        updated = self.db.upsert_instruments(records)
        return {
            "status": "ok",
            "job": "sync_instruments",
            "updated": int(updated),
            "message": f"Refreshed metadata for {updated} instruments.",
        }

    def repair_gaps(self, timeframe: str = "1d", lookback_days: int = 30, universe: str = "all") -> dict:
        """Refetch symbols whose stored bar counts fall short of the trading calendar."""
        from sqlalchemy import func, select

        started_at = datetime.now(timezone.utc)
        window_start = started_at - timedelta(days=int(lookback_days))
        resolved = self._fetchable(self.universes.resolve(universe))

        sessions = self.calendar.sessions_between(window_start.date(), started_at.date())
        expected = sessions if timeframe.endswith("d") else sessions * 7  # ~7 hourly bars/session
        if expected <= 0:
            return {"status": "no_data", "job": "repair_gaps",
                    "message": "No trading sessions in the window.", "repaired": []}

        MarketData = self.db.MarketData
        with self.db.engine.connect() as conn:
            stmt = (
                select(MarketData.symbol, func.count().label("bar_count"))
                .where(
                    MarketData.timeframe == timeframe,
                    MarketData.timestamp >= window_start,
                    MarketData.symbol.in_(resolved),
                )
                .group_by(MarketData.symbol)
            )
            counts = dict(conn.execute(stmt).fetchall())

        threshold = int(expected * 0.9)
        gappy = sorted(
            symbol for symbol in resolved if counts.get(symbol, 0) < threshold
        )
        collect_report = None
        if gappy:
            collect_report = self.collect(
                symbols=gappy,
                timeframe=timeframe,
                start=window_start.strftime("%Y-%m-%d"),
                incremental=False,
            )
        return {
            "status": "ok",
            "job": "repair_gaps",
            "timeframe": timeframe,
            "lookback_days": int(lookback_days),
            "expected_bars": int(expected),
            "repaired": gappy,
            "collect_report": collect_report,
            "timestamp": started_at.isoformat(),
            "message": f"Gap repair refetched {len(gappy)} symbols over the last {lookback_days} days.",
        }
