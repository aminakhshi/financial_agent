"""One-shot CLI and scheduler entry point for the ingestion layer.

Usage (run with PYTHONPATH=src):
    python -m ingestion.cli serve
    python -m ingestion.cli collect --timeframe 1d --symbols AAPL MSFT
    python -m ingestion.cli collect --timeframe 1h --universe all
    python -m ingestion.cli backfill --timeframe 1d --start 1991-01-01
    python -m ingestion.cli membership seed|refresh|show
    python -m ingestion.cli aggregates recompute --timeframe 1d [--full]
    python -m ingestion.cli instruments sync [--symbols ...]
    python -m ingestion.cli repair --timeframe 1d --days 30
    python -m ingestion.cli migrate

Every one-shot command prints a JSON run report to stdout, so external cron or
EventBridge can drive these instead of the built-in scheduler.
"""

import argparse
import json
import sys


def _build_service():
    from config.settings import (
        DATABASE_CONFIG,
        INGESTION_CONFIG,
        MARKET_CONFIG,
        should_use_database_config,
        sqlite_fallback_enabled,
    )
    from data.database import DatabaseManager
    from ingestion.service import IngestionService

    db_config = DATABASE_CONFIG if should_use_database_config() else None
    db_manager = DatabaseManager(db_config, use_sqlite_fallback=sqlite_fallback_enabled(default=True))
    db_manager.create_tables()
    config = {"INGESTION_CONFIG": INGESTION_CONFIG, "MARKET_CONFIG": MARKET_CONFIG}
    return IngestionService(config, db_manager)


def _emit(report: dict) -> int:
    print(json.dumps(report, indent=2, default=str))
    status = report.get("status")
    if status in ("failed", "aborted"):
        return 1
    if report.get("failures") and not report.get("rows_collected"):
        return 1
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ingestion", description="Market data ingestion jobs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("serve", help="Run the APScheduler background service (blocking).")
    subparsers.add_parser("migrate", help="Upgrade the database schema to the latest revision.")

    collect = subparsers.add_parser("collect", help="Incremental (or explicit-window) collection.")
    collect.add_argument("--timeframe", default="1h", choices=["1h", "1d"])
    collect.add_argument("--universe", default=None,
                         help="Named universe (all, indices, sector_etfs, constituents, default, ...).")
    collect.add_argument("--symbols", nargs="*", default=None)
    collect.add_argument("--start", default=None)
    collect.add_argument("--end", default=None)
    collect.add_argument("--period", default=None)
    collect.add_argument("--batch-size", type=int, default=None)

    backfill = subparsers.add_parser("backfill", help="Historical backfill (chunked, resumable).")
    backfill.add_argument("--timeframe", default="1d", choices=["1h", "1d"])
    backfill.add_argument("--universe", default="all")
    backfill.add_argument("--symbols", nargs="*", default=None)
    backfill.add_argument("--start", default=None, help="Daily backfill start date (default from config).")
    backfill.add_argument("--end", default=None)
    backfill.add_argument("--batch-size", type=int, default=None)

    membership = subparsers.add_parser("membership", help="Point-in-time index membership.")
    membership.add_argument("action", choices=["seed", "refresh", "show"])
    membership.add_argument("--effective-from", default=None, help="Seed effective date (YYYY-MM-DD).")

    aggregates = subparsers.add_parser("aggregates", help="Sector aggregate series.")
    aggregates.add_argument("action", choices=["recompute"])
    aggregates.add_argument("--timeframe", default="1d", choices=["1h", "1d"])
    aggregates.add_argument("--full", action="store_true", help="Full recompute instead of incremental.")

    instruments = subparsers.add_parser("instruments", help="Instrument metadata.")
    instruments.add_argument("action", choices=["sync"])
    instruments.add_argument("--symbols", nargs="*", default=None)

    repair = subparsers.add_parser("repair", help="Refetch symbols with calendar gaps.")
    repair.add_argument("--timeframe", default="1d", choices=["1h", "1d"])
    repair.add_argument("--days", type=int, default=30)
    repair.add_argument("--universe", default="all")

    args = parser.parse_args(argv)

    if args.command == "migrate":
        from config.settings import DATABASE_CONFIG, should_use_database_config, sqlite_fallback_enabled
        from data.database import DatabaseManager

        db_config = DATABASE_CONFIG if should_use_database_config() else None
        db_manager = DatabaseManager(db_config, use_sqlite_fallback=sqlite_fallback_enabled(default=True))
        db_manager.create_tables()
        return _emit({"status": "ok", "message": "Database migrated to the latest revision."})

    service = _build_service()

    if args.command == "serve":
        from ingestion.scheduler import run_scheduler

        run_scheduler(service)
        return 0

    if args.command == "collect":
        return _emit(
            service.collect(
                symbols=args.symbols or None,
                universe=args.universe,
                timeframe=args.timeframe,
                start=args.start,
                end=args.end,
                period=args.period,
                batch_size=args.batch_size,
            )
        )

    if args.command == "backfill":
        if args.timeframe == "1d":
            return _emit(
                service.backfill_daily(
                    symbols=args.symbols or None,
                    universe=args.universe,
                    start=args.start,
                    end=args.end,
                    batch_size=args.batch_size,
                )
            )
        return _emit(
            service.backfill_hourly(
                symbols=args.symbols or None,
                universe=args.universe,
                end=args.end,
                batch_size=args.batch_size,
            )
        )

    if args.command == "membership":
        if args.action == "seed":
            return _emit(service.seed_membership(effective_from=args.effective_from))
        if args.action == "refresh":
            return _emit(service.refresh_membership())
        members = service.membership.current_members()
        return _emit(
            {
                "status": "ok",
                "index_symbol": service.membership.index_symbol,
                "member_count": len(members),
                "members": members,
                "message": f"{len(members)} current members.",
            }
        )

    if args.command == "aggregates":
        return _emit(service.recompute_aggregates(timeframe=args.timeframe, full=args.full))

    if args.command == "instruments":
        return _emit(service.sync_instruments(symbols=args.symbols or None))

    if args.command == "repair":
        return _emit(
            service.repair_gaps(timeframe=args.timeframe, lookback_days=args.days, universe=args.universe)
        )

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
