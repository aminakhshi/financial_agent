# Multi agent-based LLM financial market prediction pipeline

This pipeline uses LLM agents to collect financial market data, train DNN models, and make predictions of market trends. It integrates various open-source frameworks and tools to create a modular and scalable architecture and provides a dashboard for visualization of market trends in a user-friendly way.

## Features

- **Multi agent-based architecture**: Integrated specialized agents for different tasks
- **Flexible DNN architecture**: Modular DNN models with technical indicators
- **Data management**: Optimized time-series database schema for financial data
- **Dashboard & visualization**: Real-time charts and performance metrics for market trends
- **Framework integration**: LangChain, CrewAI, AutoGen with open-source LLMs

## Quick Start

1. Clone the repository
2. Copy `.env.example` to a local env file such as `.env` or `.env.docker-test`
3. Keep local env files out of source control. The repo is configured to ignore `.env*` files except `.env.example`.
4. Start with Docker: `docker-compose up -d`
5. Access the dashboard: http://localhost:8501
6. Access the API docs: http://localhost:8000/docs

## Prerequisites

- Python 3.11+
- Docker & Docker Compose
- API Keys:
  - Alpha Vantage (free): https://www.alphavantage.co/support/#api-key
  - Optional: OpenAI API key for enhanced LLM capabilities

## Documentation

Structured ReadTheDocs-style documentation is available under `docs/`.

- Main index: `docs/index.rst`
- API usage: `docs/api.rst`
- Runtime modes: `docs/run-modes.rst`
- Dashboard and visualization: `docs/dashboard.rst`
- Data storage and inspection: `docs/data-storage.rst`
- OpenClaw integration: `docs/openclaw.rst`
- Read the Docs publishing: `docs/readthedocs.rst`

## API Services

The project now includes a FastAPI service for external automation and assistant integrations.

- Start locally: `uvicorn api.app:app --host 0.0.0.0 --port 8000`
- Health check: `GET /health`
- Collect market data: `POST /market-data/collect`
- Train models: `POST /models/train`
- Generate predictions: `POST /predictions/generate`
- Get latest predictions: `GET /predictions/latest?symbol=AAPL`
- Build a market report: `GET /reports/market-summary`
- Read recent logs: `GET /logs/recent`

## Data Ingestion & Background Scheduler

Deterministic ingestion lives in `src/ingestion/` (no LLM involved) and is driven three ways:

- **One-shot CLI** (cron/EventBridge friendly, prints JSON run reports):

  ```bash
  export PYTHONPATH=src
  python -m ingestion.cli migrate                    # apply schema migrations
  python -m ingestion.cli membership seed            # seed S&P 500 membership
  python -m ingestion.cli membership refresh         # diff vs live list + backfill new members
  python -m ingestion.cli collect --timeframe 1h --universe all   # incremental collection
  python -m ingestion.cli backfill --timeframe 1d --start 1991-01-01
  python -m ingestion.cli aggregates recompute --timeframe 1d --full
  ```

- **Background scheduler** (APScheduler, market-calendar aware): `python -m ingestion.cli serve`, or `docker compose up -d financial_scheduler`, or set `SCHEDULER_ENABLED=true` for `python -m main`. It runs hourly collection during market hours, daily collection after the close, a weekly S&P 500 membership refresh, weekly gap repair, and sector-aggregate recomputes. The container works as-is on AWS ECS/EC2 (only DB env vars needed).

- **API**: collection routes call the same service; backfills run as background jobs (`POST /backfill/*` returns a `job_id`; poll `GET /ingestion/runs`).

Collection is **incremental**: each run fetches only the gap since the latest stored bar per `(symbol, timeframe)`. The universe covers index tickers (`^GSPC`, `^IXIC`, `^DJI`, `^RUT`), the 11 GICS sector ETFs, and all S&P 500 constituents at both `1h` and `1d`. Hourly history is capped at ~730 days by yfinance.

## SQL Storage Behavior

Historical data is stored incrementally in SQL; the schema is managed by Alembic migrations (applied automatically on startup, including in-place upgrades of pre-existing databases).

- Market bars are stored in `market_data`, upserted in chunks by `symbol + timeframe + timestamp`, so daily and hourly bars coexist and re-collection is idempotent. Prices are split/dividend-adjusted (`auto_adjust=True`).
- `instruments` is the symbol dimension (name, exchange, GICS sector, kind: equity/index/etf/synthetic, active flag).
- `index_membership` tracks point-in-time S&P 500 membership: symbols that leave the index are closed out (`effective_to`) but their price history is **never deleted**, so training data horizons stay stable. A weekly refresh diffs against the live constituent list and backfills new members automatically.
- Synthetic sector series (`SECT_ENRG`, `SECT_INFT`, ...) are chained equal-weighted return indices over point-in-time sector members — stable training targets that survive membership churn — stored at both `1d` and `1h`.
- Predictions are stored in `prediction_results`; `actual_price` is backfilled when matching bars arrive so accuracy can be measured over time.

## Recommended Periods And Intervals

The collection layer passes `period` and `interval` through to `yfinance`.

- Common `period` values: `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `max`
- Common `interval` values for this project: `1h`, `1d`

Use `1h` or `1d` for the prediction pipeline. The prediction timestamp logic currently assumes hour-based or day-based intervals.

## Python Runbook

Start the API only:

```bash
cd /path/to/financial_agent
export PYTHONPATH=src
uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Start the dashboard plus initial pipeline (add `SCHEDULER_ENABLED=true` for background updates):

```bash
cd /path/to/financial_agent
export PYTHONPATH=src
python -m main
```

Start only the background ingestion scheduler:

```bash
export PYTHONPATH=src
python -m ingestion.cli serve
```

Collect one stock into SQL:

```bash
curl -X POST http://127.0.0.1:8000/market-data/collect \
  -H 'Content-Type: application/json' \
  -d '{"symbols":["AAPL"],"period":"1mo","interval":"1h"}'
```

Collect the full configured market universe into SQL:

```bash
curl -X POST http://127.0.0.1:8000/market-data/collect-universe \
  -H 'Content-Type: application/json' \
  -d '{"universe":"configured","period":"1mo","interval":"1h"}'
```

Supported `universe` values:

- `default` — the small watchlist
- `indices` — `^GSPC`, `^IXIC`, `^DJI`, `^RUT`
- `sector_etfs` — the 11 GICS sector ETFs
- `sp500` / `constituents` — current S&P 500 members (from the membership table)
- `all` — indices + sector ETFs + constituents
- `nasdaq`, `configured` — legacy configured lists

Train one symbol:

```bash
curl -X POST http://127.0.0.1:8000/models/train \
  -H 'Content-Type: application/json' \
  -d '{"symbols":["AAPL"],"history_period":"6mo","interval":"1h"}'
```

Generate predictions for one symbol:

```bash
curl -X POST http://127.0.0.1:8000/predictions/generate \
  -H 'Content-Type: application/json' \
  -d '{"symbols":["AAPL"],"refresh_period":"5d","interval":"1h","auto_train":true}'
```

Read historical SQL data back out:

```bash
curl 'http://127.0.0.1:8000/market-data/history?symbols=AAPL&limit=500&ascending=false'
curl 'http://127.0.0.1:8000/market-data/history?universe=configured&limit=5000'
```

Evaluate stored predictions against real prices:

```bash
curl 'http://127.0.0.1:8000/predictions/evaluate?symbols=AAPL&limit=500&sync_actuals=true'
curl 'http://127.0.0.1:8000/predictions/evaluate?universe=configured&limit=5000&sync_actuals=true'
```

If you want to query every symbol already stored in SQL, not just the configured watchlists:

```bash
curl 'http://127.0.0.1:8000/market-data/history?universe=database&limit=5000'
curl 'http://127.0.0.1:8000/predictions/evaluate?universe=database&limit=5000'
```
