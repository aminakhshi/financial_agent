"""APScheduler-based background collection service.

Runs as its own process (see ingestion.cli `serve`), locally or as a container
on AWS ECS/EC2 — it only needs database env vars. All jobs are session-guarded
by the exchange calendar and are also invocable one-shot through the CLI, so an
external cron/EventBridge can drive them instead if preferred.
"""

from datetime import datetime, timezone

try:
    from loguru import logger
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

JOB_DEFAULTS = {
    "coalesce": True,
    "max_instances": 1,
    "misfire_grace_time": 3600,
}


def build_scheduler(service) -> BlockingScheduler:
    cfg = service.ingestion_config
    tz = cfg.get("scheduler_timezone", "America/New_York")
    scheduler = BlockingScheduler(job_defaults=JOB_DEFAULTS, timezone=tz)

    def _session_guard(job_name: str) -> bool:
        if not service.calendar.is_trading_day(datetime.now(timezone.utc).date()):
            logger.info(f"{job_name}: not a trading session; skipping.")
            return False
        return True

    def hourly_collect():
        if not _session_guard("hourly_collect"):
            return
        service.collect(universe=cfg.get("hourly_universe", "all"), timeframe="1h")
        service.recompute_aggregates(timeframe="1h")

    def daily_collect():
        if not _session_guard("daily_collect"):
            return
        service.collect(universe=cfg.get("daily_universe", "all"), timeframe="1d")
        service.recompute_aggregates(timeframe="1d")

    def membership_refresh():
        service.refresh_membership()

    def gap_repair():
        service.repair_gaps(timeframe="1d", lookback_days=30)
        service.repair_gaps(timeframe="1h", lookback_days=7)

    minute = int(cfg.get("hourly_collect_minute", 5))
    scheduler.add_job(
        hourly_collect,
        CronTrigger(day_of_week="mon-fri", hour="10-17", minute=minute, timezone=tz),
        id="hourly_collect",
        name="Hourly incremental collection (full universe) + 1h sector aggregates",
    )

    daily_hour, daily_minute = (cfg.get("daily_collect_time", "17:30").split(":") + ["0"])[:2]
    scheduler.add_job(
        daily_collect,
        CronTrigger(day_of_week="mon-fri", hour=int(daily_hour), minute=int(daily_minute), timezone=tz),
        id="daily_collect",
        name="Daily incremental collection (full universe) + 1d sector aggregates",
    )

    refresh_hour, refresh_minute = (cfg.get("membership_refresh_time", "08:00").split(":") + ["0"])[:2]
    scheduler.add_job(
        membership_refresh,
        CronTrigger(
            day_of_week=cfg.get("membership_refresh_day", "sat"),
            hour=int(refresh_hour),
            minute=int(refresh_minute),
            timezone=tz,
        ),
        id="membership_refresh",
        name="Weekly S&P 500 membership refresh (+ backfill of new members)",
    )

    if cfg.get("gap_repair_enabled", True):
        scheduler.add_job(
            gap_repair,
            CronTrigger(day_of_week="sun", hour=9, minute=0, timezone=tz),
            id="gap_repair",
            name="Weekly gap repair (1d/30d and 1h/7d windows)",
        )

    return scheduler


def run_scheduler(service) -> None:
    scheduler = build_scheduler(service)
    jobs = scheduler.get_jobs()
    logger.info(
        "Ingestion scheduler starting with jobs:\n"
        + "\n".join(f"  - {job.id}: {job.name} [{job.trigger}]" for job in jobs)
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Ingestion scheduler stopped.")
