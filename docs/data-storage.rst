Data Storage and Inspection
===========================

Storage overview
----------------

The project stores state in SQL and on disk.

SQL tables:

* ``market_data``
* ``prediction_results``
* ``model_monitor_events``

File-based artifacts:

* SQLite DB file such as ``financial_data.db``
* model artifacts under ``models/saved/``
* log files under ``logs/financial_agent_*.log``

Market data behavior
--------------------

Historical market bars are stored incrementally.

* Existing history is preserved.
* Matching bars are refreshed on upsert.
* Daily and hourly bars are separated by the ``timeframe`` column.

Prediction behavior
-------------------

Predictions are stored in ``prediction_results`` with:

* ``symbol``
* ``timeframe``
* ``prediction_timestamp``
* ``predicted_price``
* ``confidence_score``
* ``model_version``
* ``actual_price`` once realized market data becomes available

Monitoring behavior
-------------------

Model health events are stored in ``model_monitor_events``. These rows let you
audit when the system decided to monitor, skip, or fine-tune a model.

Inspecting SQLite directly
--------------------------

List stored symbols:

.. code-block:: bash

   sqlite3 financial_data.db "SELECT DISTINCT symbol FROM market_data ORDER BY symbol;"

Summarize stored bars:

.. code-block:: bash

   sqlite3 financial_data.db "SELECT timeframe, COUNT(*), COUNT(DISTINCT symbol), MIN(timestamp), MAX(timestamp) FROM market_data GROUP BY timeframe ORDER BY timeframe;"

Inspect daily coverage per symbol:

.. code-block:: bash

   sqlite3 financial_data.db "SELECT symbol, COUNT(*), MIN(timestamp), MAX(timestamp) FROM market_data WHERE timeframe='1d' GROUP BY symbol ORDER BY symbol;"

Inspect prediction coverage:

.. code-block:: bash

   sqlite3 financial_data.db "SELECT symbol, timeframe, COUNT(*), SUM(CASE WHEN actual_price IS NOT NULL THEN 1 ELSE 0 END) FROM prediction_results GROUP BY symbol, timeframe ORDER BY symbol, timeframe;"

Export daily prices to CSV:

.. code-block:: bash

   sqlite3 -header -csv financial_data.db "
   SELECT symbol, exchange, timeframe, timestamp, open_price, high_price, low_price, close_price, volume
   FROM market_data
   WHERE timeframe='1d'
   ORDER BY symbol, timestamp;
   " > daily_prices.csv

Useful API inspection routes
----------------------------

* ``GET /market-data/history``
* ``GET /coverage/summary``
* ``GET /predictions/latest``
* ``GET /predictions/evaluate``
* ``GET /models/monitor-history``
* ``GET /logs/recent``

