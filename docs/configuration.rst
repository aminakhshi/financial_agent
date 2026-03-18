Configuration
=============

Configuration sources
---------------------

The project reads configuration from environment variables. It loads local
environment files through ``python-dotenv`` during startup.

Database selection rules
------------------------

Database resolution works in this order:

1. ``DB_URL`` or ``DATABASE_URL`` if present
2. Derived PostgreSQL URL from ``DB_HOST``, ``DB_PORT``, ``DB_NAME``,
   ``DB_USER``, and ``DB_PASSWORD``
3. SQLite fallback if ``ENABLE_SQLITE_FALLBACK=true`` and PostgreSQL is not
   available

Useful database variables
-------------------------

* ``DATABASE_URL``
  Full database URL. Example: ``sqlite:///./financial_data.db``
* ``DB_URL``
  Alternative full URL name supported by the database layer
* ``DB_HOST``, ``DB_PORT``, ``DB_NAME``, ``DB_USER``, ``DB_PASSWORD``
  PostgreSQL configuration
* ``ENABLE_SQLITE_FALLBACK``
  Allows fallback to SQLite when PostgreSQL cannot be used
* ``SQLITE_DB_PATH``
  Overrides the default fallback SQLite path

Recommended profiles
--------------------

Local SQLite:

.. code-block:: bash

   export DATABASE_URL=sqlite:///./financial_data.db
   export ENABLE_SQLITE_FALLBACK=true

Local PostgreSQL:

.. code-block:: bash

   unset DATABASE_URL
   export DB_HOST=127.0.0.1
   export DB_PORT=5432
   export DB_NAME=financial_data
   export DB_USER=postgres
   export DB_PASSWORD=change_me

LLM configuration
-----------------

Important variables:

* ``LLM_PROVIDER``
* ``LLM_MODEL_NAME``
* ``LLM_BASE_URL``
* ``LLM_REQUEST_TIMEOUT``
* ``OPENAI_API_KEY``

The code prefers an OpenAI-compatible chat client against the configured
``LLM_BASE_URL`` and falls back to Ollama-specific integration if needed. If no
LLM is reachable, the pipeline still runs with simplified non-LLM summaries.

Market and model configuration
------------------------------

Important runtime knobs from ``src/config/settings.py``:

* ``DOWNLOAD_BATCH_SIZE``
* ``DAILY_MODEL_TRAINING_PERIOD``
* ``DAILY_PREDICTION_REFRESH_PERIOD``
* ``HOURLY_MODEL_TRAINING_PERIOD``
* ``HOURLY_PREDICTION_REFRESH_PERIOD``
* ``DISABLE_CREWAI``

Startup and scheduled backfill knobs
------------------------------------

These variables control long-running bootstrap and maintenance behavior for
``python -m main`` and the ``financial_app`` Docker service:

* ``STARTUP_DAILY_BACKFILL_ENABLED``
* ``STARTUP_DAILY_BACKFILL_UNIVERSE``
* ``STARTUP_DAILY_BACKFILL_START``
* ``STARTUP_DAILY_BACKFILL_END``
* ``STARTUP_DAILY_BACKFILL_BATCH_SIZE``
* ``STARTUP_HOURLY_BACKFILL_ENABLED``
* ``STARTUP_HOURLY_BACKFILL_UNIVERSE``
* ``STARTUP_HOURLY_BACKFILL_PERIOD``
* ``STARTUP_HOURLY_BACKFILL_END``
* ``STARTUP_HOURLY_BACKFILL_BATCH_SIZE``
* ``ENABLE_DAILY_MARKET_UPDATE``
* ``DAILY_MARKET_UPDATE_UNIVERSE``
* ``DAILY_MARKET_UPDATE_PERIOD``
* ``DAILY_MARKET_UPDATE_TIME``
* ``DAILY_MARKET_UPDATE_BATCH_SIZE``

Without these flags, the integrated service only runs the default watchlist
pipeline. It does not automatically perform the full S&P 500 daily bootstrap.

Default universes:

* default watchlist
* S&P 500
* NASDAQ
* combined configured universe

Port configuration in Docker
----------------------------

These variables change published ports in Docker Compose:

* ``APP_HOST_PORT`` for the dashboard
* ``API_HOST_PORT`` for FastAPI
* ``DB_HOST_PORT`` for PostgreSQL
* ``REDIS_HOST_PORT`` for Redis
* ``OLLAMA_HOST_PORT`` for Ollama

Security notes
--------------

Do not commit local tokens, database passwords, Telegram bot tokens, or other
private credentials. Keep them in local env files or external secret stores.
