Dashboard and Visualization
===========================

Streamlit dashboard
-------------------

Start the dashboard directly:

.. code-block:: bash

   export PYTHONPATH=src
   streamlit run src/dashboard/app.py --server.address 127.0.0.1 --server.port 8501

Open:

* ``http://127.0.0.1:8501``

What the dashboard shows
------------------------

The dashboard reads directly from SQL storage. It does not fetch new market
data by itself.

Main views:

* price history chart
* prediction overlay
* technical indicators
* market coverage
* prediction coverage
* recent market rows
* recent prediction rows

Sidebar controls
----------------

Available filters include:

* stored data timeframe: ``1h`` or ``1d``
* universe selector
* only show symbols with stored data
* symbol filter
* rows to plot
* prediction rows
* auto refresh

Typical visualization workflow
------------------------------

1. Run a backfill or collection job through the API.
2. Start the dashboard.
3. Choose ``1d`` for daily history or ``1h`` for hourly history.
4. Set the universe to ``Stored symbols`` or a configured universe.
5. Inspect coverage and prediction availability for the selected symbol.

If the dashboard looks empty
----------------------------

Common causes:

* no stored bars exist for the selected timeframe
* the selected universe is filtered to stored symbols and the DB is still empty
* predictions have not been generated yet for the selected symbol
* the dashboard is pointing at a different database configuration than the API

Other visualization surfaces
----------------------------

FastAPI also provides two useful web UIs:

* Swagger UI at ``/docs``
* ReDoc at ``/redoc``

These are useful for visual inspection of request and response schemas.

