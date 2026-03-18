Financial Market Agent
======================

This documentation describes how to run the project in source mode, API-only
mode, scheduler mode, Docker mode, and OpenClaw mode. It also explains where
the data is stored, how to inspect it, and how to use the dashboard and API
for different workflows.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting-started
   run-modes
   configuration
   api
   dashboard
   data-storage
   openclaw
   operations
   readthedocs
   troubleshooting

Publishing
----------

The repository includes a Read the Docs configuration in
``.readthedocs.yaml``. Push the docs source files to GitHub and let Read the
Docs build and host the HTML. Do not commit generated files from
``docs/_build/``.

Key runtime entry points
------------------------

The project has three main runtime surfaces:

* ``python -m main``
  Starts the integrated service in ``src/main.py``. This launches the initial
  pipeline, keeps scheduled jobs running, and starts the Streamlit dashboard.
* ``uvicorn api.app:app --host 127.0.0.1 --port 8000``
  Starts only the FastAPI service. This is the right mode for API clients,
  OpenClaw, and manual ``curl`` workflows.
* ``streamlit run src/dashboard/app.py --server.address 127.0.0.1 --server.port 8501``
  Starts only the dashboard.

What this project stores
------------------------

* Market bars in SQL under ``market_data``
* Model predictions in SQL under ``prediction_results``
* Model monitoring events in SQL under ``model_monitor_events``
* Saved model artifacts under ``models/saved/``
* Log files under ``logs/``
