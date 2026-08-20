import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional

import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .dependencies import get_automation_agent, get_ingestion_service


# ---------------------------------------------------------------------------
# Background ingestion run registry (in-memory, last N runs)
# ---------------------------------------------------------------------------

_MAX_TRACKED_RUNS = 50
_runs: "OrderedDict[str, dict]" = OrderedDict()
_runs_lock = threading.Lock()


def _register_run(job: str) -> str:
    job_id = uuid.uuid4().hex
    with _runs_lock:
        _runs[job_id] = {
            "job_id": job_id,
            "job": job,
            "status": "running",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "result": None,
        }
        while len(_runs) > _MAX_TRACKED_RUNS:
            _runs.popitem(last=False)
    return job_id


def _run_job(job_id: str, func: Callable[[], dict]) -> None:
    try:
        result = func()
        status = "completed"
    except Exception as exc:  # noqa: BLE001 - report, never crash the worker
        result = {"error": str(exc)}
        status = "failed"
    with _runs_lock:
        if job_id in _runs:
            _runs[job_id]["status"] = status
            _runs[job_id]["result"] = result
            _runs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()


def _submit_background(background_tasks: BackgroundTasks, job: str, func: Callable[[], dict]) -> dict:
    job_id = _register_run(job)
    background_tasks.add_task(_run_job, job_id, func)
    return {
        "status": "accepted",
        "job_id": job_id,
        "job": job,
        "message": f"{job} started in the background. Poll GET /ingestion/runs for the result.",
    }


class CollectRequest(BaseModel):
    symbols: List[str] = Field(default_factory=list)
    period: str = "5d"
    interval: str = "1h"
    start: Optional[str] = None
    end: Optional[str] = None
    batch_size: Optional[int] = Field(default=None, ge=1, le=250)


class UniverseCollectRequest(BaseModel):
    universe: str = "all"
    symbols: List[str] = Field(default_factory=list)
    period: str = "1mo"
    interval: str = "1h"
    start: Optional[str] = None
    end: Optional[str] = None
    batch_size: Optional[int] = Field(default=None, ge=1, le=250)


class TrainRequest(BaseModel):
    symbols: List[str] = Field(default_factory=list)
    history_period: str = "6mo"
    interval: str = "1h"
    force_refresh: bool = False


class PredictionRequest(BaseModel):
    symbols: List[str] = Field(default_factory=list)
    interval: str = "1h"
    refresh_period: str = "5d"
    force_refresh: bool = False
    auto_train: bool = True


class PipelineRequest(BaseModel):
    symbols: List[str] = Field(default_factory=list)
    history_period: str = "6mo"
    interval: str = "1h"


class DailyBackfillRequest(BaseModel):
    universe: str = "sp500"
    symbols: List[str] = Field(default_factory=list)
    start: str = "1991-01-01"
    end: Optional[str] = None
    batch_size: Optional[int] = Field(default=None, ge=1, le=250)


class HourlyBackfillRequest(BaseModel):
    universe: str = "sp500"
    symbols: List[str] = Field(default_factory=list)
    period: str = "6mo"
    end: Optional[str] = None
    batch_size: Optional[int] = Field(default=None, ge=1, le=250)


class Sp500BackfillRequest(BaseModel):
    daily_start: str = "1991-01-01"
    daily_end: Optional[str] = None
    hourly_period: str = "6mo"
    hourly_end: Optional[str] = None
    batch_size: Optional[int] = Field(default=None, ge=1, le=250)


class MonitorRequest(BaseModel):
    symbols: List[str] = Field(default_factory=list)
    interval: str = "1d"
    auto_fine_tune: bool = True


app = FastAPI(
    title="Financial Market Service",
    description=(
        "API services for market data collection, short-term forecasting, reporting, and operational logs."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _handle_service_error(exc: Exception):
    if isinstance(exc, FileNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    raise HTTPException(status_code=400, detail=str(exc)) from exc


def _read_recent_logs(lines: int = 100):
    log_dir = Path("logs")
    if not log_dir.exists():
        return {
            "log_file": None,
            "lines": [],
            "message": "No log directory is available yet.",
        }

    candidates = sorted(log_dir.glob("financial_agent_*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        return {
            "log_file": None,
            "lines": [],
            "message": "No log file is available yet.",
        }

    latest_log = candidates[0]
    log_lines = latest_log.read_text(encoding="utf-8", errors="ignore").splitlines()[-lines:]
    return {
        "log_file": str(latest_log),
        "lines": log_lines,
        "message": f"Retrieved {len(log_lines)} log lines from {latest_log.name}.",
    }


@app.get("/health")
def health(agent: Any = Depends(get_automation_agent)):
    try:
        with agent.db_manager.engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database check failed: {exc}") from exc

    return {
        "status": "ok",
        "message": "The financial market service is available.",
        "default_symbols": agent.default_symbols,
    }


@app.post("/market-data/collect")
def collect_market_data(request: CollectRequest, service: Any = Depends(get_ingestion_service)):
    try:
        return service.collect(
            symbols=request.symbols or None,
            timeframe=request.interval,
            start=request.start,
            end=request.end,
            period=None if (request.start or request.end) else request.period,
            batch_size=request.batch_size,
        )
    except Exception as exc:
        _handle_service_error(exc)


@app.post("/market-data/collect-universe")
def collect_market_universe(request: UniverseCollectRequest, service: Any = Depends(get_ingestion_service)):
    try:
        return service.collect(
            symbols=request.symbols or None,
            universe=request.universe,
            timeframe=request.interval,
            start=request.start,
            end=request.end,
            period=None if (request.start or request.end) else request.period,
            batch_size=request.batch_size,
        )
    except Exception as exc:
        _handle_service_error(exc)


@app.post("/backfill/daily")
def backfill_daily(
    request: DailyBackfillRequest,
    background_tasks: BackgroundTasks,
    service: Any = Depends(get_ingestion_service),
):
    return _submit_background(
        background_tasks,
        "backfill_daily",
        lambda: service.backfill_daily(
            universe=request.universe,
            symbols=request.symbols or None,
            start=request.start,
            end=request.end,
            batch_size=request.batch_size,
        ),
    )


@app.post("/backfill/hourly")
def backfill_hourly(
    request: HourlyBackfillRequest,
    background_tasks: BackgroundTasks,
    service: Any = Depends(get_ingestion_service),
):
    return _submit_background(
        background_tasks,
        "backfill_hourly",
        lambda: service.backfill_hourly(
            universe=request.universe,
            symbols=request.symbols or None,
            end=request.end,
            batch_size=request.batch_size,
        ),
    )


@app.post("/backfill/sp500")
def backfill_sp500(
    request: Sp500BackfillRequest,
    background_tasks: BackgroundTasks,
    service: Any = Depends(get_ingestion_service),
):
    def _run() -> dict:
        daily = service.backfill_daily(
            universe="sp500", start=request.daily_start, end=request.daily_end,
            batch_size=request.batch_size,
        )
        hourly = service.backfill_hourly(
            universe="sp500", end=request.hourly_end, batch_size=request.batch_size,
        )
        return {
            "universe": "sp500",
            "daily": daily,
            "hourly": hourly,
            "message": "Completed S&P 500 daily and hourly history backfills.",
        }

    return _submit_background(background_tasks, "backfill_sp500", _run)


@app.get("/ingestion/runs")
def ingestion_runs(job_id: Optional[str] = Query(None)):
    with _runs_lock:
        if job_id:
            run = _runs.get(job_id)
            if run is None:
                raise HTTPException(status_code=404, detail=f"Unknown job id: {job_id}")
            return run
        return {"runs": list(reversed(_runs.values()))}


@app.post("/ingestion/membership/refresh")
def refresh_membership(
    background_tasks: BackgroundTasks,
    service: Any = Depends(get_ingestion_service),
):
    return _submit_background(background_tasks, "membership_refresh", service.refresh_membership)


@app.get("/ingestion/membership")
def get_membership(
    asof: Optional[str] = Query(None, description="Point-in-time date (YYYY-MM-DD); default: current."),
    service: Any = Depends(get_ingestion_service),
):
    try:
        if asof:
            members = service.db.get_members_asof(service.membership.index_symbol, asof)
        else:
            members = service.membership.current_members()
        return {
            "index_symbol": service.membership.index_symbol,
            "asof": asof or "current",
            "member_count": len(members),
            "members": members,
        }
    except Exception as exc:
        _handle_service_error(exc)


@app.post("/ingestion/aggregates/recompute")
def recompute_aggregates(
    background_tasks: BackgroundTasks,
    timeframe: str = Query("1d"),
    full: bool = Query(False),
    service: Any = Depends(get_ingestion_service),
):
    return _submit_background(
        background_tasks,
        "recompute_aggregates",
        lambda: service.recompute_aggregates(timeframe=timeframe, full=full),
    )


@app.get("/instruments")
def list_instruments(
    kind: Optional[str] = Query(None),
    active: Optional[bool] = Query(None),
    service: Any = Depends(get_ingestion_service),
):
    try:
        instruments = service.db.get_instruments(kind=kind, active=active)
        rows = instruments.to_dict(orient="records")
        for row in rows:
            for key in ("first_seen", "last_seen", "created_at", "updated_at"):
                if row.get(key) is not None and not isinstance(row[key], str):
                    row[key] = pd.to_datetime(row[key], utc=True).isoformat()
        return {"count": len(rows), "instruments": rows}
    except Exception as exc:
        _handle_service_error(exc)


@app.get("/market-data/history")
def market_history(
    symbols: Optional[List[str]] = Query(None),
    universe: str = Query("configured"),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    exchange: Optional[str] = Query(None),
    timeframe: Optional[str] = Query(None),
    limit: int = Query(1000, ge=1, le=100000),
    ascending: bool = Query(False),
    agent: Any = Depends(get_automation_agent),
):
    try:
        return agent.get_market_history(
            symbols=symbols,
            universe=universe,
            start=start,
            end=end,
            exchange=exchange,
            timeframe=timeframe,
            limit_rows=limit,
            ascending=ascending,
        )
    except Exception as exc:
        _handle_service_error(exc)


@app.get("/coverage/summary")
def coverage_summary(
    symbols: Optional[List[str]] = Query(None),
    universe: str = Query("configured"),
    exchange: Optional[str] = Query(None),
    timeframe: Optional[str] = Query(None),
    agent: Any = Depends(get_automation_agent),
):
    try:
        return agent.get_data_coverage(
            symbols=symbols,
            universe=universe,
            exchange=exchange,
            timeframe=timeframe,
        )
    except Exception as exc:
        _handle_service_error(exc)


@app.post("/models/train")
def train_models(request: TrainRequest, agent: Any = Depends(get_automation_agent)):
    try:
        return agent.train_models(
            symbols=request.symbols or None,
            history_period=request.history_period,
            interval=request.interval,
            force_refresh=request.force_refresh,
        )
    except Exception as exc:
        _handle_service_error(exc)


@app.post("/predictions/generate")
def generate_predictions(request: PredictionRequest, agent: Any = Depends(get_automation_agent)):
    try:
        return agent.generate_predictions(
            symbols=request.symbols or None,
            interval=request.interval,
            refresh_period=request.refresh_period,
            force_refresh=request.force_refresh,
            auto_train=request.auto_train,
        )
    except Exception as exc:
        _handle_service_error(exc)


@app.get("/predictions/latest")
def latest_predictions(
    symbol: str = Query(..., min_length=1),
    interval: Optional[str] = Query(None),
    limit: int = Query(24, ge=1, le=500),
    agent: Any = Depends(get_automation_agent),
):
    try:
        return agent.get_latest_predictions(symbol, limit_rows=limit, interval=interval)
    except Exception as exc:
        _handle_service_error(exc)


@app.get("/predictions/evaluate")
def evaluate_predictions(
    symbols: Optional[List[str]] = Query(None),
    universe: str = Query("configured"),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    interval: Optional[str] = Query(None),
    limit: int = Query(1000, ge=1, le=100000),
    sync_actuals: bool = Query(True),
    agent: Any = Depends(get_automation_agent),
):
    try:
        return agent.evaluate_predictions(
            symbols=symbols,
            universe=universe,
            start=start,
            end=end,
            interval=interval,
            limit_rows=limit,
            sync_actuals=sync_actuals,
        )
    except Exception as exc:
        _handle_service_error(exc)


@app.post("/models/monitor")
def monitor_models(request: MonitorRequest, agent: Any = Depends(get_automation_agent)):
    try:
        symbols = request.symbols or agent.default_symbols
        completed = [
            agent.monitor_model_health(symbol, interval=request.interval, auto_fine_tune=request.auto_fine_tune)
            for symbol in symbols
        ]
        return {
            "symbols": symbols,
            "interval": request.interval,
            "completed": completed,
            "message": f"Monitored {len(symbols)} models for interval {request.interval}.",
        }
    except Exception as exc:
        _handle_service_error(exc)


@app.get("/models/monitor-history")
def monitor_history(
    symbol: Optional[str] = Query(None),
    interval: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    agent: Any = Depends(get_automation_agent),
):
    try:
        history = agent.db_manager.get_monitor_history(symbol=symbol, timeframe=interval, limit_rows=limit)
        rows = history.to_dict(orient="records")
        for row in rows:
            if row.get("prediction_timestamp") is not None:
                row["prediction_timestamp"] = pd.to_datetime(row["prediction_timestamp"], utc=True).isoformat()
            if row.get("created_at") is not None:
                row["created_at"] = pd.to_datetime(row["created_at"], utc=True).isoformat()
        return {
            "symbol": symbol,
            "interval": interval,
            "rows": rows,
            "message": f"Retrieved {len(rows)} monitor events.",
        }
    except Exception as exc:
        _handle_service_error(exc)


@app.get("/reports/market-summary")
def market_summary(
    symbols: Optional[List[str]] = Query(None),
    interval: str = Query("1h"),
    refresh_period: str = Query("5d"),
    force_refresh: bool = Query(False),
    auto_predict: bool = Query(True),
    auto_train: bool = Query(True),
    agent: Any = Depends(get_automation_agent),
):
    try:
        return agent.build_market_report(
            symbols=symbols,
            interval=interval,
            refresh_period=refresh_period,
            force_refresh=force_refresh,
            auto_predict=auto_predict,
            auto_train=auto_train,
        )
    except Exception as exc:
        _handle_service_error(exc)


@app.post("/pipeline/full-run")
async def run_full_pipeline(request: PipelineRequest, agent: Any = Depends(get_automation_agent)):
    try:
        return await agent.run_full_pipeline(
            symbols=request.symbols or None,
            history_period=request.history_period,
            interval=request.interval,
        )
    except Exception as exc:
        _handle_service_error(exc)


@app.post("/pipeline/hourly-update")
def run_hourly_update(request: PredictionRequest, agent: Any = Depends(get_automation_agent)):
    try:
        return agent.run_hourly_update(
            symbols=request.symbols or None,
            refresh_period=request.refresh_period,
            interval=request.interval,
        )
    except Exception as exc:
        _handle_service_error(exc)


@app.get("/logs/recent")
def recent_logs(lines: int = Query(100, ge=1, le=1000)):
    return _read_recent_logs(lines)
