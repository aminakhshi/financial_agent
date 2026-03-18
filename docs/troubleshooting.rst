Troubleshooting
===============

Port already in use
-------------------

Symptoms:

* ``[Errno 98] address already in use``

Cause:

* another ``uvicorn`` or Streamlit process is already bound to the port

Fix:

* stop the old process or start on a different port

Missing Python modules
----------------------

Symptoms:

* ``ModuleNotFoundError: No module named 'sqlalchemy'``
* ``ModuleNotFoundError: No module named 'loguru'``

Cause:

* the project is being run with the wrong Python interpreter or an incomplete
  environment

Fix:

* activate the intended environment
* install dependencies with ``pip install -r requirements.txt``

PostgreSQL authentication failure
---------------------------------

Symptoms:

* ``password authentication failed``

Cause:

* ``DATABASE_URL`` or ``DB_*`` points to an unavailable or misconfigured
  PostgreSQL instance

Fix:

* switch to SQLite locally with:

  .. code-block:: bash

     export DATABASE_URL=sqlite:///./financial_data.db
     export ENABLE_SQLITE_FALLBACK=true

No data in dashboard
--------------------

Symptoms:

* the dashboard starts but charts are empty

Cause:

* no stored market rows exist for the selected timeframe or symbol

Fix:

* run ``/backfill/daily`` or ``/market-data/collect`` first
* verify coverage through ``/coverage/summary``

Backfill stopped and the DB did not change
------------------------------------------

Common causes:

* the API process is still running stale code from before a patch
* the request failed and wrote an error response to the output file instead of
  a success summary

Fix:

* restart ``uvicorn`` after code changes
* inspect the saved response file
* inspect ``financial_data.db`` directly with ``sqlite3``

No predictions for a symbol
---------------------------

Possible reasons:

* no model has been trained for the symbol and timeframe
* no sufficient history exists yet for the selected interval
* the dashboard is showing stored market data but prediction generation has not
  been run

Fix:

* run ``/models/train``
* run ``/predictions/generate``
* check ``/predictions/latest`` and ``/predictions/evaluate``

OpenClaw plugin warnings
------------------------

If OpenClaw reports ``plugin not found`` or continues to use stale plugin code:

* relink the plugin path
* restart the OpenClaw gateway
* confirm the plugin config points to the correct API base URL

