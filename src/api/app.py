from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .dependencies import get_automation_agent


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
def collect_market_data(request: CollectRequest, agent: Any = Depends(get_automation_agent)):
    try:
        return agent.collect_market_data(
            symbols=request.symbols or None,
            period=request.period,
            interval=request.interval,
            start=request.start,
            end=request.end,
            batch_size=request.batch_size,
        )
    except Exception as exc:
        _handle_service_error(exc)


@app.post("/market-data/collect-universe")
def collect_market_universe(request: UniverseCollectRequest, agent: Any = Depends(get_automation_agent)):
    try:
        return agent.collect_market_universe(
            universe=request.universe,
            symbols=request.symbols or None,
            period=request.period,
            interval=request.interval,
            start=request.start,
            end=request.end,
            batch_size=request.batch_size,
        )
    except Exception as exc:
        _handle_service_error(exc)


@app.post("/backfill/daily")
def backfill_daily(request: DailyBackfillRequest, agent: Any = Depends(get_automation_agent)):
    try:
        return agent.backfill_daily_history(
            universe=request.universe,
            symbols=request.symbols or None,
            start=request.start,
            end=request.end,
            batch_size=request.batch_size,
        )
    except Exception as exc:
        _handle_service_error(exc)


@app.post("/backfill/hourly")
def backfill_hourly(request: HourlyBackfillRequest, agent: Any = Depends(get_automation_agent)):
    try:
        return agent.backfill_hourly_history(
            universe=request.universe,
            symbols=request.symbols or None,
            period=request.period,
            end=request.end,
            batch_size=request.batch_size,
        )
    except Exception as exc:
        _handle_service_error(exc)


@app.post("/backfill/sp500")
def backfill_sp500(request: Sp500BackfillRequest, agent: Any = Depends(get_automation_agent)):
    try:
        return agent.backfill_sp500_history(
            daily_start=request.daily_start,
            daily_end=request.daily_end,
            hourly_period=request.hourly_period,
            hourly_end=request.hourly_end,
            batch_size=request.batch_size,
        )
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
