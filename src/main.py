import asyncio
import os
import threading
import traceback
import requests
from loguru import logger

from config.settings import DATABASE_CONFIG, API_KEYS, LLM_CONFIG, MARKET_CONFIG, MODEL_CONFIG
from data.database import DatabaseManager
from agents.data_collector_agent import DataCollectorAgent
from agents.automation_agent import AutomationAgent
from models.lstm_model import LSTMPredictor
from dashboard.app import main as dashboard_main
import subprocess

def setup_database():
    """This function creates the database and necessary tables"""
    db_manager = DatabaseManager(DATABASE_CONFIG, use_sqlite_fallback=True)
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
    # --- Friendly Logging Setup ---
    log_file_path = os.path.join("logs", "financial_agent_{time}.log")
    logger.add(log_file_path, rotation="10 MB", retention="7 days", level="INFO",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")
    logger.remove() # Removes the default handler
    logger.add(lambda msg: print(msg, end=""), colorize=True, level="INFO",
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    # --- End Logging Setup ---

    logger.info("=" * 80)
    logger.info("Starting the financial agent system!")
    logger.info("=" * 80 + "\n")

    try:
        db_manager = setup_database()

        # Let's check if Ollama is awake
        model_name = LLM_CONFIG.get('model_name', 'gpt-oss:20b')
        base_url = LLM_CONFIG.get('base_url', 'http://localhost:11434')
        try:
            requests.get(base_url, timeout=2)
            logger.success(f"Connected to Ollama at {base_url}")

            # Now, let's see if the model we need is available
            model_list_response = requests.get(f"{base_url}/api/tags", timeout=2)
            models = model_list_response.json().get("models", [])
            if any(m.get("name") == model_name for m in models):
                logger.success(f"Found our model '{model_name}' in Ollama!")
            else:
                logger.warning(f"Model '{model_name}' not found in Ollama.")
                logger.info("To get it, run: ollama pull {model_name}")
        except Exception as e:
            logger.warning(f"Couldn't connect to the Ollama server: {e}")
            logger.info("Make sure Ollama is running to use all the features: ollama serve")

        orchestrator = setup_agents(db_manager)

        # Launch the dashboard
        dashboard_thread = threading.Thread(target=run_dashboard)
        dashboard_thread.daemon = True
        dashboard_thread.start()
        logger.info("🌐 Dashboard is live at: http://localhost:8501")

        # first pipeline run
        logger.info("Calling agents to start the task.")
        await orchestrator.run_full_pipeline()
        logger.success("✅ Initial pipeline run finished successfully!")

        logger.info("Starting the scheduled operations to update the data...")
        orchestrator.schedule_operations()

    except KeyboardInterrupt:
        logger.info("👋 Shutting down the system. See you next time!")
    except Exception as e:
        logger.error(f"\n❌ A critical error occurred: {e}")
        logger.error(traceback.format_exc())
        logger.info("\n" + "=" * 80)
        logger.info("📋 Having trouble? Here's a quick troubleshooting guide:")
        logger.info("=" * 80)
        logger.info("1. Database: If PostgreSQL isn't running, we're using SQLite as a backup.")
        logger.info("   To set up PostgreSQL: sudo -u postgres psql -c \"CREATE DATABASE financial_data;\"")
        logger.info("2. LLM: Make sure Ollama is running (`ollama serve`) and you have the model (`ollama pull llama3:8b`).")
        logger.info("3. API Keys: Double-check that your `.env` file has your API keys.")
        logger.info("=" * 80)

if __name__ == "__main__":
    os.makedirs("models/saved", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    asyncio.run(main())