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
* hourly scheduled updates
* the daily scheduled full run
* Streamlit started automatically

Command:

.. code-block:: bash

   export PYTHONPATH=src
   python -m main

Behavior:

* Creates the database connection
* checks local LLM connectivity
* starts the dashboard
* runs an initial full pipeline
* schedules hourly updates
* schedules a daily full pipeline at ``02:00``

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
* Use ``python -m main`` for long-running local scheduling.
* Use ``streamlit run`` when you only need visualization.
* Use Docker when you want a more reproducible local stack.

