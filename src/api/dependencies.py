import os
from functools import lru_cache


def _sqlite_fallback_enabled() -> bool:
    from config.settings import sqlite_fallback_enabled

    return sqlite_fallback_enabled(default=True)


@lru_cache
def get_automation_agent():
    from agents.automation_agent import AutomationAgent
    from agents.data_collector_agent import DataCollectorAgent
    from config.settings import (
        API_KEYS,
        DATABASE_CONFIG,
        LLM_CONFIG,
        MARKET_CONFIG,
        MODEL_CONFIG,
        should_use_database_config,
    )
    from data.database import DatabaseManager
    from models.lstm_model import LSTMPredictor

    config = {
        "LLM_CONFIG": LLM_CONFIG,
        "API_KEYS": API_KEYS,
        "MARKET_CONFIG": MARKET_CONFIG,
        "MODEL_CONFIG": MODEL_CONFIG,
    }

    db_config = DATABASE_CONFIG if should_use_database_config() else None
    db_manager = DatabaseManager(
        db_config,
        use_sqlite_fallback=_sqlite_fallback_enabled(),
    )
    db_manager.create_tables()

    data_collector = DataCollectorAgent(config, db_manager)
    ml_predictor = LSTMPredictor(config)
    return AutomationAgent(config, db_manager, data_collector, ml_predictor)
