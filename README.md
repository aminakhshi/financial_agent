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

See the implementation guide for detailed setup and usage instructions.

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

## Background Behavior

There are two different ways to run this project, and they behave differently:

- `PYTHONPATH=src python -m main`
  This starts the integrated Python service in `src/main.py`. It runs an initial full pipeline, then keeps running in the foreground with:
  - hourly incremental updates
  - a daily full pipeline at `02:00`
  - the Streamlit dashboard on port `8501`
- `PYTHONPATH=src uvicorn api.app:app --host 0.0.0.0 --port 8000`
  This starts only the FastAPI service. It does not schedule background collection by itself.

If you want continuous background updates, keep `python -m main` running under `systemd`, Docker, `tmux`, or another process manager.

## SQL Storage Behavior

Historical data is stored incrementally in SQL.

- Market bars are stored in `market_data`.
- Existing bars are upserted by `symbol + exchange + timestamp`, so old history stays in the database and only matching bars are refreshed.
- Predictions are stored in `prediction_results`.
- When later market bars arrive, the code backfills `actual_price` for matching prediction timestamps so you can measure model accuracy over time.

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

Start the full background scheduler plus dashboard:

```bash
cd /path/to/financial_agent
export PYTHONPATH=src
python -m main
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

- `default`
- `sp500`
- `nasdaq`
- `all`
- `configured`

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
