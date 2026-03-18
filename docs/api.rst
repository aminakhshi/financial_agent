API
===

FastAPI surface
---------------

Start the API from the repository root:

.. code-block:: bash

   export PYTHONPATH=src
   uvicorn api.app:app --host 127.0.0.1 --port 8000

Interactive docs:

* ``http://127.0.0.1:8000/docs``
* ``http://127.0.0.1:8000/redoc``

Core endpoints
--------------

Service and logs:

* ``GET /health``
* ``GET /logs/recent``

Collection and backfill:

* ``POST /market-data/collect``
* ``POST /market-data/collect-universe``
* ``POST /backfill/daily``
* ``POST /backfill/hourly``
* ``POST /backfill/sp500``

Stored data inspection:

* ``GET /market-data/history``
* ``GET /coverage/summary``

Model training and prediction:

* ``POST /models/train``
* ``POST /predictions/generate``
* ``GET /predictions/latest``
* ``GET /predictions/evaluate``

Monitoring:

* ``POST /models/monitor``
* ``GET /models/monitor-history``

Reports and orchestration:

* ``GET /reports/market-summary``
* ``POST /pipeline/full-run``
* ``POST /pipeline/hourly-update``

Example workflows
-----------------

Collect one symbol:

.. code-block:: bash

   curl -X POST http://127.0.0.1:8000/market-data/collect \
     -H 'Content-Type: application/json' \
     -d '{"symbols":["AAPL"],"period":"1mo","interval":"1h"}'

Collect a universe:

.. code-block:: bash

   curl -X POST http://127.0.0.1:8000/market-data/collect-universe \
     -H 'Content-Type: application/json' \
     -d '{"universe":"sp500","period":"1mo","interval":"1h"}'

Backfill daily S&P 500 history:

.. code-block:: bash

   curl -X POST http://127.0.0.1:8000/backfill/daily \
     -H 'Content-Type: application/json' \
     -d '{"universe":"sp500","start":"1991-01-01","batch_size":25}'

Backfill hourly S&P 500 history:

.. code-block:: bash

   curl -X POST http://127.0.0.1:8000/backfill/hourly \
     -H 'Content-Type: application/json' \
     -d '{"universe":"sp500","period":"6mo","batch_size":25}'

Train a daily model:

.. code-block:: bash

   curl -X POST http://127.0.0.1:8000/models/train \
     -H 'Content-Type: application/json' \
     -d '{"symbols":["AAPL"],"history_period":"10y","interval":"1d"}'

Generate predictions:

.. code-block:: bash

   curl -X POST http://127.0.0.1:8000/predictions/generate \
     -H 'Content-Type: application/json' \
     -d '{"symbols":["AAPL"],"interval":"1d","refresh_period":"1y","auto_train":true}'

Evaluate predictions:

.. code-block:: bash

   curl 'http://127.0.0.1:8000/predictions/evaluate?symbols=AAPL&interval=1d&limit=500&sync_actuals=true'

Trigger model monitoring:

.. code-block:: bash

   curl -X POST http://127.0.0.1:8000/models/monitor \
     -H 'Content-Type: application/json' \
     -d '{"symbols":["AAPL"],"interval":"1d","auto_fine_tune":true}'

Build a market report:

.. code-block:: bash

   curl 'http://127.0.0.1:8000/reports/market-summary?symbols=AAPL&interval=1d&auto_predict=true&auto_train=true'

Query parameters and universes
------------------------------

Common universe values:

* ``default``
* ``sp500``
* ``nasdaq``
* ``all``
* ``configured``
* ``database`` or ``stored`` for SQL-only inspection routes

Common interval values:

* ``1h``
* ``1d``

Common history windows:

* ``5d``
* ``1mo``
* ``3mo``
* ``6mo``
* ``1y``
* explicit ``start`` and ``end`` dates for backfills

