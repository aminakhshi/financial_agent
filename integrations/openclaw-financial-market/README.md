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
