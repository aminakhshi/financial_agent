Run Modes
=========

Overview
--------

The project supports multiple runtime modes. Choose the one that matches the
task you are trying to perform.

API-only mode with uvicorn
--------------------------

Use this mode when you want:

* FastAPI endpoints
* OpenClaw integration
* manual ``curl`` workflows
* backfills and model jobs triggered on demand

Command:

.. code-block:: bash

   export PYTHONPATH=src
   uvicorn api.app:app --host 127.0.0.1 --port 8000

Behavior:

* Starts the FastAPI app only
* Does not start the Streamlit dashboard
* Does not schedule background collection by itself

Integrated service mode
-----------------------

Use this mode when you want:

* the initial pipeline run
* Streamlit started automatically
* optionally, the background ingestion scheduler in the same process

Command:

.. code-block:: bash

   export PYTHONPATH=src
   python -m main

Behavior:

* Creates the database connection and applies schema migrations
* checks local LLM connectivity
* starts the dashboard
* runs an initial full pipeline
* if ``SCHEDULER_ENABLED=true``, runs the ingestion scheduler (see below)

Scheduler mode
--------------

A dedicated APScheduler service keeps the database current in the background.
It is optional and can run locally, in Docker, or as an AWS ECS/EC2 container
(it only needs database environment variables).

.. code-block:: bash

   export PYTHONPATH=src
   python -m ingestion.cli serve

Jobs (all timed in ``SCHEDULER_TIMEZONE``, default America/New_York, and
guarded by the exchange trading calendar):

* hourly incremental collection of the full universe (indices, sector ETFs,
  S&P 500 constituents) during market hours, plus 1h sector aggregates
* daily incremental collection after the close, plus 1d sector aggregates
* weekly S&P 500 membership refresh (new members are backfilled automatically)
* weekly gap repair (optional)

Every job is also available as a one-shot CLI command (``collect``,
``backfill``, ``membership``, ``aggregates``, ``repair``), printing a JSON run
report — so an external cron or AWS EventBridge can drive collection instead
of the built-in scheduler.

Dashboard-only mode
-------------------

Use this mode when you only want to inspect already-stored SQL data.

.. code-block:: bash

   export PYTHONPATH=src
   streamlit run src/dashboard/app.py --server.address 127.0.0.1 --server.port 8501

Docker mode
-----------

Docker Compose defines separate services:

* ``financial_app``
  Runs ``python src/main.py``
* ``financial_api``
  Runs ``uvicorn api.app:app --host 0.0.0.0 --port 8000``
* ``financial_scheduler``
  Runs ``python -m ingestion.cli serve`` (background collection; optional)
* ``postgres``
  PostgreSQL storage
* ``redis``
  Redis sidecar
* ``ollama``
  local model server

Common examples:

.. code-block:: bash

   docker compose --env-file .env.docker-local up --build financial_api

.. code-block:: bash

   docker compose --env-file .env.docker-local up --build financial_app

.. code-block:: bash

   docker compose --env-file .env.docker-local up --build

Which mode to use
-----------------

* Use ``uvicorn`` for API clients, OpenClaw, and manual backfill commands.
* Use ``python -m ingestion.cli serve`` (or the ``financial_scheduler``
  compose service) for long-running background collection.
* Use ``python -m main`` for the dashboard plus initial pipeline (add
  ``SCHEDULER_ENABLED=true`` to also schedule in-process).
* Use ``streamlit run`` when you only need visualization.
* Use Docker when you want a more reproducible local stack.

