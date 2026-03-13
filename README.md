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
