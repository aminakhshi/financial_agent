import asyncio
import os
import threading
import traceback
import subprocess

import requests
from loguru import logger

from config.settings import DATABASE_CONFIG, API_KEYS, LLM_CONFIG, MARKET_CONFIG, MODEL_CONFIG
from data.database import DatabaseManager
from agents.data_collector_agent import DataCollectorAgent
from agents.automation_agent import AutomationAgent
from models.lstm_model import LSTMPredictor


def _sqlite_fallback_enabled():
    return os.getenv("ENABLE_SQLITE_FALLBACK", "false").strip().lower() == "true"


def configure_logging():
    log_file_path = os.path.join("logs", "financial_agent_{time}.log")
    logger.remove()
    logger.add(
        log_file_path,
        rotation="10 MB",
        retention="7 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )
    logger.add(
        lambda msg: print(msg, end=""),
        colorize=True,
        level="INFO",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        ),
    )

def setup_database():
    """This function creates the database and necessary tables"""
    db_config = None if os.getenv("DB_URL") or os.getenv("DATABASE_URL") else DATABASE_CONFIG
    db_manager = DatabaseManager(
        db_config,
        use_sqlite_fallback=_sqlite_fallback_enabled(),
    )
    db_manager.create_tables()
    logger.success("Database setup successful!")
    return db_manager


def setup_agents(db_manager):
    """Starting all the agents."""
    config = {
        'LLM_CONFIG': LLM_CONFIG,
        'API_KEYS': API_KEYS,
        'MARKET_CONFIG': MARKET_CONFIG,
        'MODEL_CONFIG': MODEL_CONFIG
    }

    data_collector = DataCollectorAgent(config, db_manager)
    ml_predictor = LSTMPredictor(config)
    orchestrator = AutomationAgent(config, db_manager, data_collector, ml_predictor)

    logger.info("All agents are initialized and ready for action.")
    return orchestrator


def run_dashboard():
    """Generates Streamlit dashboard in a separate, non-blocking process."""
    dashboard_path = os.path.join(os.path.dirname(__file__), 'dashboard', 'app.py')
    # Use Popen for non-blocking execution
    command = ['streamlit', 'run', dashboard_path, '--server.port', '8501', '--server.headless', 'true']
    
    # Using Popen to run the dashboard as a background process
    proc = subprocess.Popen(command)
    logger.info(f"Dashboard process started with PID: {proc.pid}")
    return proc

async def main():
    """The main entry point for our financial agent."""
    configure_logging()

    logger.info("=" * 80)
    logger.info("Starting the financial market service.")
    logger.info("=" * 80 + "\n")

    try:
        db_manager = setup_database()

        # Let's check if Ollama is awake
        model_name = LLM_CONFIG.get('model_name', 'gpt-oss:20b')
        base_url = LLM_CONFIG.get('base_url', 'http://localhost:11434')
        try:
            requests.get(base_url, timeout=2)
            logger.success(f"Connected to Ollama at {base_url}.")

            model_list_response = requests.get(f"{base_url}/api/tags", timeout=2)
            models = model_list_response.json().get("models", [])
            if any(m.get("name") == model_name for m in models):
                logger.success(f"Model '{model_name}' is available in Ollama.")
            else:
                logger.warning(f"Model '{model_name}' is not available in Ollama.")
                logger.info(f"Run `ollama pull {model_name}` to download it.")
        except Exception as e:
            logger.warning(f"Could not connect to the Ollama server: {e}")
            logger.info("Start Ollama with `ollama serve` to enable local model access.")

        orchestrator = setup_agents(db_manager)

        # Launch the dashboard
        dashboard_thread = threading.Thread(target=run_dashboard)
        dashboard_thread.daemon = True
        dashboard_thread.start()
        logger.info("Dashboard is available at http://localhost:8501.")

        logger.info("Starting the initial market pipeline run.")
        await orchestrator.run_full_pipeline()
        logger.success("Initial pipeline run finished successfully.")

        logger.info("Starting scheduled updates.")
        orchestrator.schedule_operations()

    except KeyboardInterrupt:
        logger.info("Shutting down the service.")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        logger.error(traceback.format_exc())
        logger.info("\n" + "=" * 80)
        logger.info("Troubleshooting guide:")
        logger.info("=" * 80)
        logger.info("1. Database: If PostgreSQL is unavailable, enable SQLite fallback or create the database manually.")
        logger.info("   Example: sudo -u postgres psql -c \"CREATE DATABASE financial_data;\"")
        logger.info("2. LLM: Start Ollama with `ollama serve` and pull the required model.")
        logger.info("3. API Keys: Verify the values in your `.env` file.")
        logger.info("=" * 80)

if __name__ == "__main__":
    os.makedirs("models/saved", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    asyncio.run(main())
