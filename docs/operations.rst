Operations and Workflows
========================

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

.. code-block:: bash

   time curl -X POST http://127.0.0.1:8000/backfill/daily \
     -H 'Content-Type: application/json' \
     -d '{"universe":"sp500","start":"1991-01-01","batch_size":25}' \
     -o sp500_daily_backfill.json

Run a full hourly S&P 500 backfill
----------------------------------

.. code-block:: bash

   time curl -X POST http://127.0.0.1:8000/backfill/hourly \
     -H 'Content-Type: application/json' \
     -d '{"universe":"sp500","period":"6mo","batch_size":25}' \
     -o sp500_hourly_backfill.json

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
* schedule hourly updates
* schedule the daily full run

Do not use it as a replacement for explicit long historical backfills if you
need precise control over time windows, batches, or direct API access.

Automatic bootstrap behavior
----------------------------

If you want ``python -m main`` or the ``financial_app`` Docker service to
perform a long historical bootstrap automatically, enable the startup backfill
environment variables:

.. code-block:: bash

   export STARTUP_DAILY_BACKFILL_ENABLED=true
   export STARTUP_DAILY_BACKFILL_UNIVERSE=sp500
   export STARTUP_DAILY_BACKFILL_START=1991-01-01
   export STARTUP_DAILY_BACKFILL_BATCH_SIZE=25

For ongoing daily ``1d`` market maintenance after the bootstrap:

.. code-block:: bash

   export ENABLE_DAILY_MARKET_UPDATE=true
   export DAILY_MARKET_UPDATE_UNIVERSE=sp500
   export DAILY_MARKET_UPDATE_PERIOD=7d
   export DAILY_MARKET_UPDATE_TIME=18:30
