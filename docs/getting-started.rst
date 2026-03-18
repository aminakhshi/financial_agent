Getting Started
===============

Prerequisites
-------------

The project is designed for Python 3.11+ and Docker-based local deployment.

Minimum local prerequisites:

* Python 3.11+
* ``pip`` or another Python package installer
* Optional: Docker and Docker Compose
* Optional: Ollama if you want local LLM-backed summaries
* Optional: PostgreSQL if you do not want SQLite

Repository setup
----------------

Clone the repository and create a local environment file from
``.env.example``.

.. code-block:: bash

   git clone <repo-url>
   cd financial_agent
   cp .env.example .env.local

Keep local environment files out of source control. The repository is already
configured so example files remain tracked while local env files stay private.

Local Python install
--------------------

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

For source runs, export ``PYTHONPATH=src`` from the repository root:

.. code-block:: bash

   export PYTHONPATH=src

Recommended local SQLite profile
--------------------------------

For the lowest-friction local setup, use SQLite:

.. code-block:: bash

   export DATABASE_URL=sqlite:///./financial_data.db
   export ENABLE_SQLITE_FALLBACK=true

This stores data in ``financial_data.db`` in the repository root.

Quick health check
------------------

Start the API:

.. code-block:: bash

   export PYTHONPATH=src
   uvicorn api.app:app --host 127.0.0.1 --port 8000

In another shell:

.. code-block:: bash

   curl http://127.0.0.1:8000/health

You can also open the generated OpenAPI docs in the browser:

* ``http://127.0.0.1:8000/docs``
* ``http://127.0.0.1:8000/redoc``

Docker quick start
------------------

Create a Docker-specific env file from the example and edit values locally:

.. code-block:: bash

   cp .env.example .env.docker-local

Then start the stack:

.. code-block:: bash

   docker compose --env-file .env.docker-local up --build

Default exposed ports:

* Dashboard: ``http://127.0.0.1:8501``
* API docs: ``http://127.0.0.1:8000/docs``

