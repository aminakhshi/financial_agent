OpenClaw Integration
====================

Overview
--------

The repository includes an OpenClaw plugin under
``integrations/openclaw-financial-market``. The plugin talks to the FastAPI
service. It does not replace the API.

Prerequisite
------------

Start the FastAPI service first:

.. code-block:: bash

   export PYTHONPATH=src
   uvicorn api.app:app --host 127.0.0.1 --port 8000

Install the plugin
------------------

.. code-block:: bash

   openclaw plugins install --link /absolute/path/to/financial_agent/integrations/openclaw-financial-market

Enable the plugin in OpenClaw config
------------------------------------

Add an entry under ``plugins.entries.financial-market`` with a config similar
to this:

.. code-block:: json

   {
     "enabled": true,
     "config": {
       "baseUrl": "http://127.0.0.1:8000",
       "timeoutMs": 180000,
       "defaultSymbols": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
       "defaultPeriod": "5d",
       "defaultInterval": "1h"
     }
   }

Common CLI commands
-------------------

.. code-block:: bash

   openclaw financial-market health
   openclaw financial-market collect --symbols AAPL,MSFT
   openclaw financial-market train --symbols AAPL,MSFT
   openclaw financial-market predict --symbols AAPL,MSFT
   openclaw financial-market report --symbols AAPL,MSFT --refresh
   openclaw financial-market logs --lines 50

Chat commands
-------------

.. code-block:: text

   /marketreport --symbols AAPL,MSFT --refresh
   /marketreport --symbols AAPL --refresh --train --history-period 6mo --interval 1h
   /marketrun --symbols AAPL,MSFT --history-period 6mo --interval 1h
   /marketlogs --lines 50

Scheduled delivery
------------------

The plugin can register scheduled report delivery through OpenClaw:

.. code-block:: bash

   openclaw financial-market schedule-report \
     --every 1h \
     --channel telegram \
     --to <CHAT_ID> \
     --symbols AAPL,MSFT \
     --refresh

Operational model
-----------------

* OpenClaw invokes the FastAPI service on demand.
* The plugin does not keep its own collector running.
* For automatic ongoing SQL updates, keep either ``python -m main`` running or
  schedule the required API jobs externally.

