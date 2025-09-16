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
2. Set up your environment variables in `.env`
3. Start with Docker: `docker-compose up -d`
4. Access the dashboard: http://localhost:8501

## Prerequisites

- Python 3.11+
- Docker & Docker Compose
- API Keys:
  - Alpha Vantage (free): https://www.alphavantage.co/support/#api-key
  - Optional: OpenAI API key for enhanced LLM capabilities

## Documentation

See the implementation guide for detailed setup and usage instructions.