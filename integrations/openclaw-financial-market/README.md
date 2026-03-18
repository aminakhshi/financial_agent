# OpenClaw Financial Market Plugin

This plugin connects OpenClaw to the local financial market FastAPI service in this repository.

## Install

```bash
openclaw plugins install /absolute/path/to/financial_agent/integrations/openclaw-financial-market
```

For isolated local testing, prefer a dedicated profile:

```bash
openclaw --profile financial-agent-test plugins install /absolute/path/to/financial_agent/integrations/openclaw-financial-market
```

## Plugin config

Set the plugin config in your OpenClaw config file under `plugins.entries.financial-market.config`:

```json
{
  "plugins": {
    "entries": {
      "financial-market": {
        "enabled": true,
        "config": {
          "baseUrl": "http://127.0.0.1:8000",
          "timeoutMs": 180000,
          "defaultSymbols": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
          "defaultPeriod": "5d",
          "defaultInterval": "1h",
          "defaultReportRefresh": false,
          "delivery": {
            "every": "1h",
            "tz": "America/Toronto",
            "channel": "slack",
            "to": "C0123456789",
            "session": "isolated",
            "bestEffortDeliver": true
          }
        }
      }
    }
  }
}
```

## CLI commands

```bash
openclaw financial-market health
openclaw financial-market collect --symbols AAPL,MSFT
openclaw financial-market train --symbols AAPL,MSFT
openclaw financial-market predict --symbols AAPL,MSFT
openclaw financial-market report --symbols AAPL,MSFT --refresh
openclaw financial-market report --symbols AAPL,MSFT --train --history-period 6mo
openclaw financial-market logs --lines 50
```

## What Runs In The Background

The OpenClaw plugin does not keep a collector running by itself. It calls the FastAPI service when you invoke a command.

- For always-on background collection inside Python, run `PYTHONPATH=src python -m main`
- For scheduled report delivery through OpenClaw, use `openclaw financial-market schedule-report`
- For API-only mode, run `uvicorn api.app:app ...` and invoke commands on demand

## OpenClaw Runbook

Check the API:

```bash
openclaw financial-market health
```

Collect one symbol:

```bash
openclaw financial-market collect --symbols AAPL --period 1mo --interval 1h
```

Collect multiple symbols:

```bash
openclaw financial-market collect --symbols AAPL,MSFT,NVDA --period 1mo --interval 1h
```

Train one symbol:

```bash
openclaw financial-market train --symbols AAPL --period 6mo --interval 1h
```

Generate predictions for one symbol:

```bash
openclaw financial-market predict --symbols AAPL --period 5d --interval 1h
```

Build a report for one symbol:

```bash
openclaw financial-market report --symbols AAPL --refresh
```

Build a report and retrain first:

```bash
openclaw financial-market report --symbols AAPL --refresh --train --history-period 6mo --interval 1h
```

Run the full CLI sequence for one or more symbols:

```bash
openclaw financial-market collect --symbols AAPL,MSFT --period 1mo --interval 1h
openclaw financial-market train --symbols AAPL,MSFT --period 6mo --interval 1h
openclaw financial-market predict --symbols AAPL,MSFT --period 5d --interval 1h
openclaw financial-market report --symbols AAPL,MSFT
```

Show recent logs:

```bash
openclaw financial-market logs --lines 100
```

### Duration And Interval Flags

- `--period`
  Used for market data collection or refresh windows such as `5d`, `1mo`, `3mo`, `6mo`, `1y`
- `--history-period`
  Used for model training history, typically `6mo` or `1y`
- `--interval`
  Use `1h` or `1d` for the forecasting workflow

### Chat Commands

OpenClaw chat commands accept the same timing flags:

- `/marketreport --symbols AAPL --refresh`
- `/marketreport --symbols AAPL --refresh --train --history-period 6mo --interval 1h`
- `/marketrun --symbols AAPL,MSFT --history-period 6mo --interval 1h`
- `/marketlogs --lines 50`

## Chat commands

- `/marketreport --symbols AAPL,MSFT --refresh`
- `/marketreport --symbols AAPL,MSFT --train --history-period 6mo`
- `/marketrun --symbols AAPL,MSFT`
- `/marketlogs --lines 50`

## Scheduled delivery

Create a cron job that runs `/marketreport` and delivers the result through OpenClaw:

```bash
openclaw financial-market schedule-report \
  --every 1h \
  --channel slack \
  --to C0123456789 \
  --symbols AAPL,MSFT \
  --refresh
```

Scheduled reports run in an isolated OpenClaw cron session so the command can execute and the result can be delivered back to the target channel.

Preview the generated `openclaw cron add` command without applying it:

```bash
openclaw financial-market schedule-report --every 1h --channel slack --to C0123456789 --dry-run
```

If you also want the schedule to retrain before each report, add `--train --history-period 6mo`.

Example hourly Telegram or Slack delivery:

```bash
openclaw financial-market schedule-report \
  --every 1h \
  --channel telegram \
  --to <CHAT_ID> \
  --symbols AAPL,MSFT \
  --refresh
```

This schedules report generation and delivery. It does not replace the Python background scheduler for broad continuous SQL ingestion.
