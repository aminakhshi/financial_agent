from pathlib import Path
from typing import Any, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .dependencies import get_automation_agent


class CollectRequest(BaseModel):
    symbols: List[str] = Field(default_factory=list)
    period: str = "5d"
    interval: str = "1h"


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
    limit: int = Query(24, ge=1, le=500),
    agent: Any = Depends(get_automation_agent),
):
    try:
        return agent.get_latest_predictions(symbol, limit_rows=limit)
    except Exception as exc:
        _handle_service_error(exc)


@app.get("/reports/market-summary")
def market_summary(
    symbols: Optional[List[str]] = Query(None),
    agent: Any = Depends(get_automation_agent),
):
    try:
        return agent.build_market_report(symbols=symbols)
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
