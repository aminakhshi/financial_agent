import asyncio
import os
from config.settings import DATABASE_CONFIG, API_KEYS, LLM_CONFIG, MARKET_CONFIG, MODEL_CONFIG
from data.database import DatabaseManager
from agents.data_collector_agent import DataCollectorAgent
from agents.automation_agent import AutomationAgent
from models.lstm_model import LSTMPredictor
from dashboard.app import main as dashboard_main
import subprocess
import threading

def setup_database():
    """Initialize database and create tables"""
    db_manager = DatabaseManager(DATABASE_CONFIG)
    db_manager.create_tables()
    print("Database initialized")
    return db_manager

def setup_agents(db_manager):
    """Initialize all agents"""
    config = {
        'LLM_CONFIG': LLM_CONFIG,
        'API_KEYS': API_KEYS,
        'MARKET_CONFIG': MARKET_CONFIG,
        'MODEL_CONFIG': MODEL_CONFIG
    }
    
    # Initialize components
    data_collector = DataCollectorAgent(config, db_manager)
    ml_predictor = LSTMPredictor(config)
    orchestrator = AutomationAgent(config, db_manager, data_collector, ml_predictor)
    
    print("All agents initialized")
    return orchestrator

def run_dashboard():
    """Run the dashboard separately"""
    subprocess.run(['streamlit', 'run', 'dashboard/app.py', '--server.port', '8501'])

async def main():
    """Main application function"""
    print("Starting pipeline")
    print("=" * 60)
    
    # Setup
    db_manager = setup_database()
    orchestrator = setup_agents(db_manager)
    
    # Start dashboard in background
    dashboard_thread = threading.Thread(target=run_dashboard)
    dashboard_thread.daemon = True
    dashboard_thread.start()
    
    print("Dashboard started at: http://localhost:8501")
    
    # Initial full pipeline run
    print("Running initial pipeline...")
    initial_results = await orchestrator.run_full_pipeline()
    print("Initial pipeline completed")
    
    # Start scheduled operations
    print(" Starting scheduled operations...")
    
    # Run continuously
    try:
        orchestrator.schedule_operations()
    except KeyboardInterrupt:
        print("\n Pipeline stopped by user")
    except Exception as e:
        print(f" Error in pipeline: {e}")

if __name__ == "__main__":
    # Create directories 
    os.makedirs("models/saved", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Run the application
    asyncio.run(main())