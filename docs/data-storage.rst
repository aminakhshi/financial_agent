Data Storage and Inspection
===========================

Storage overview
----------------

The project stores state in SQL and on disk.

SQL tables:

* ``market_data`` — one bar per ``(symbol, timeframe, timestamp)``
* ``instruments`` — symbol dimension (name, real exchange, GICS sector,
  currency, kind: equity/index/etf/synthetic, active flag)
* ``index_membership`` — point-in-time index membership
  (``effective_from``/``effective_to``; ``effective_to IS NULL`` = current member)
* ``prediction_results``
* ``model_monitor_events``

The schema is managed by Alembic migrations under ``src/data/migrations``.
Every entry point runs ``upgrade head`` on startup; databases created before
Alembic are stamped at the baseline automatically and migrated in place
(existing daily history is preserved).

File-based artifacts:

* SQLite DB file such as ``financial_data.db``
* model artifacts under ``models/saved/``
* log files under ``logs/financial_agent_*.log``

Market data behavior
--------------------

Historical market bars are stored incrementally.

* Existing history is preserved; delisted symbols keep their bars forever.
* Bars are upserted in chunks keyed by ``(symbol, timeframe, timestamp)``,
  so daily and hourly bars for the same moment coexist and re-collection is
  idempotent.
* Collection is watermark-driven: each run fetches only the gap since the
  latest stored bar per symbol (plus a two-bar overlap to repair partially
  formed bars).
* Prices are fetched split- and dividend-adjusted (``auto_adjust=True``). If
  your database predates this change, run a full daily re-backfill once:
  ``python -m ingestion.cli backfill --timeframe 1d``.
* Hourly history is limited to roughly the last 730 days by the data provider.
* The legacy ``exchange`` column is deprecated (kept nullable for API
  compatibility); real listing venues live on ``instruments.exchange``.

Index membership behavior
-------------------------

* The initial S&P 500 membership is seeded from
  ``src/data/static/sp500_symbols.txt`` effective from the backfill horizon
  (current members are assumed to have been members through stored history — a
  documented approximation until real historical changes are imported).
* The weekly refresh diffs the live constituent list (Wikipedia) against open
  memberships: removed symbols get ``effective_to`` set and their instrument
  marked inactive — price rows are never deleted; added symbols get an open
  row and are backfilled automatically.
* Point-in-time queries: ``GET /ingestion/membership?asof=YYYY-MM-DD``.

Sector aggregate series
-----------------------

Synthetic sector indices (``SECT_ENRG``, ``SECT_INFT``, ...) are chained
equal-weighted return indices over the point-in-time S&P 500 members of each
GICS sector, stored through the normal upsert at both ``1d`` and ``1h``.
Chaining means membership changes never cause level jumps, so these series are
stable model-training targets that survive index churn.

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

