Operations and Workflows
========================

Ingestion CLI (one-shot jobs)
-----------------------------

All ingestion jobs run without the API, printing a JSON run report — suitable
for external cron or AWS EventBridge:

.. code-block:: bash

   export PYTHONPATH=src
   python -m ingestion.cli migrate                       # apply schema migrations
   python -m ingestion.cli membership seed               # seed S&P 500 membership
   python -m ingestion.cli membership refresh            # diff vs live list, backfill new members
   python -m ingestion.cli membership show
   python -m ingestion.cli collect --timeframe 1h --universe all     # incremental
   python -m ingestion.cli collect --timeframe 1d --symbols AAPL MSFT
   python -m ingestion.cli backfill --timeframe 1d --start 1991-01-01
   python -m ingestion.cli backfill --timeframe 1h                   # ~730d provider cap
   python -m ingestion.cli aggregates recompute --timeframe 1d --full
   python -m ingestion.cli repair --timeframe 1d --days 30
   python -m ingestion.cli serve                         # blocking APScheduler service

Background scheduler
--------------------

Enable continuous collection either with the dedicated docker-compose service:

.. code-block:: bash

   docker compose up -d financial_scheduler

or in-process with ``python -m main`` by setting ``SCHEDULER_ENABLED=true``.
Schedule times, universes, and batch sizes are configured through environment
variables (see ``.env.example``). On AWS, run the same container image with
``python -m ingestion.cli serve`` as the command (ECS/EC2), or trigger the
one-shot CLI commands from EventBridge.

Migrating an existing database
------------------------------

Databases created before this ingestion layer are migrated in place the first
time any entry point starts (or via ``python -m ingestion.cli migrate``):
duplicate bars from legacy exchange labels are deduplicated (last write wins),
the unique key becomes ``(symbol, timeframe, timestamp)``, and the instrument +
membership tables are created and seeded. Back up PostgreSQL first
(``pg_dump``) and stop the API/scheduler during the upgrade. Because prices
are now stored dividend/split-adjusted, finish by re-running the daily
backfill once: ``python -m ingestion.cli backfill --timeframe 1d``.

Run one symbol hourly
---------------------

Collect:

.. code-block:: bash

   curl -X POST http://127.0.0.1:8000/market-data/collect \
     -H 'Content-Type: application/json' \
     -d '{"symbols":["AAPL"],"period":"1mo","interval":"1h"}'

Train:

.. code-block:: bash

   curl -X POST http://127.0.0.1:8000/models/train \
     -H 'Content-Type: application/json' \
     -d '{"symbols":["AAPL"],"history_period":"6mo","interval":"1h"}'

Predict:

.. code-block:: bash

   curl -X POST http://127.0.0.1:8000/predictions/generate \
     -H 'Content-Type: application/json' \
     -d '{"symbols":["AAPL"],"interval":"1h","refresh_period":"5d","auto_train":true}'

Run one symbol daily
--------------------

Backfill daily data:

.. code-block:: bash

   curl -X POST http://127.0.0.1:8000/backfill/daily \
     -H 'Content-Type: application/json' \
     -d '{"symbols":["AAPL"],"start":"1991-01-01","batch_size":1}'

Train daily model:

.. code-block:: bash

   curl -X POST http://127.0.0.1:8000/models/train \
     -H 'Content-Type: application/json' \
     -d '{"symbols":["AAPL"],"history_period":"10y","interval":"1d"}'

Generate a daily prediction:

.. code-block:: bash

   curl -X POST http://127.0.0.1:8000/predictions/generate \
     -H 'Content-Type: application/json' \
     -d '{"symbols":["AAPL"],"interval":"1d","refresh_period":"1y","auto_train":true}'

Run a full daily S&P 500 backfill
---------------------------------

Backfill routes now return immediately with a job id and run in the
background; poll ``/ingestion/runs`` for the result:

.. code-block:: bash

   curl -X POST http://127.0.0.1:8000/backfill/daily \
     -H 'Content-Type: application/json' \
     -d '{"universe":"sp500","start":"1991-01-01","batch_size":25}'
   # -> {"status": "accepted", "job_id": "..."}

   curl 'http://127.0.0.1:8000/ingestion/runs?job_id=<job_id>' | python -m json.tool

Run a full hourly S&P 500 backfill
----------------------------------

Hourly backfills always cover the provider's full intraday lookback (~730
days); the legacy ``period`` field is ignored:

.. code-block:: bash

   curl -X POST http://127.0.0.1:8000/backfill/hourly \
     -H 'Content-Type: application/json' \
     -d '{"universe":"sp500","batch_size":25}'

Membership, instruments, and aggregates over the API
----------------------------------------------------

.. code-block:: bash

   curl -X POST http://127.0.0.1:8000/ingestion/membership/refresh
   curl 'http://127.0.0.1:8000/ingestion/membership?asof=2024-06-01' | python -m json.tool
   curl -X POST 'http://127.0.0.1:8000/ingestion/aggregates/recompute?timeframe=1d&full=true'
   curl 'http://127.0.0.1:8000/instruments?kind=synthetic' | python -m json.tool
   curl 'http://127.0.0.1:8000/ingestion/runs' | python -m json.tool

Check coverage after a backfill
-------------------------------

.. code-block:: bash

   curl 'http://127.0.0.1:8000/coverage/summary?universe=sp500&timeframe=1d' | python -m json.tool

.. code-block:: bash

   curl 'http://127.0.0.1:8000/coverage/summary?universe=sp500&timeframe=1h' | python -m json.tool

Evaluate prediction quality
---------------------------

.. code-block:: bash

   curl 'http://127.0.0.1:8000/predictions/evaluate?symbols=AAPL&interval=1d&limit=500&sync_actuals=true' | python -m json.tool

Monitor and fine-tune
---------------------

.. code-block:: bash

   curl -X POST http://127.0.0.1:8000/models/monitor \
     -H 'Content-Type: application/json' \
     -d '{"symbols":["AAPL"],"interval":"1d","auto_fine_tune":true}' | python -m json.tool

Read monitor history:

.. code-block:: bash

   curl 'http://127.0.0.1:8000/models/monitor-history?symbol=AAPL&interval=1d&limit=20' | python -m json.tool

When to run ``python -m main``
------------------------------

Use ``python -m main`` when you want a single long-running process to:

* launch the dashboard
* run the initial pipeline
* optionally run the ingestion scheduler (``SCHEDULER_ENABLED=true``)

Do not use it as a replacement for explicit long historical backfills if you
need precise control over time windows, batches, or direct API access — use
``python -m ingestion.cli`` for those.

