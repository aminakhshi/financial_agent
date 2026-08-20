import os
from dotenv import load_dotenv
from pathlib import Path

from data.symbol_universe import load_sp500_symbols

load_dotenv()


def get_required_env(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def has_explicit_database_url() -> bool:
    return bool(os.environ.get("DB_URL", "").strip() or os.environ.get("DATABASE_URL", "").strip())


def has_explicit_db_config() -> bool:
    return any(
        os.environ.get(name, "").strip()
        for name in ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD")
    )


def should_use_database_config() -> bool:
    return not has_explicit_database_url() and has_explicit_db_config()


def sqlite_fallback_enabled(default: bool = True) -> bool:
    return env_flag("ENABLE_SQLITE_FALLBACK", default=default)

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
FULL_SP500_SYMBOLS = load_sp500_symbols(BASE_DIR)
DEFAULT_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
FALLBACK_SP500_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B',
    'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'ABBV',
    'PFE', 'AVGO', 'COST', 'DIS', 'KO', 'MRK', 'PEP', 'TMO', 'WMT',
    'ABT', 'BAC', 'CRM', 'CSCO', 'ACN', 'LIN', 'ADBE', 'MCD', 'VZ',
    'DHR', 'NFLX', 'CMCSA', 'NKE', 'TXN', 'NEE', 'AMD', 'PM', 'RTX',
    'UPS', 'T', 'LOW',
]

# API Keys - load from environment variables
API_KEY_ALPHAVANTAGE = os.environ.get("API_KEY_ALPHAVANTAGE", "")
API_KEY_FINANCIALMODELINGPREP = os.environ.get("API_KEY_FINANCIALMODELINGPREP", "")
API_KEY_NEWS = os.environ.get("API_KEY_NEWS", "")
API_KEY_POLYGON = os.environ.get("API_KEY_POLYGON", "")

# Database settings
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# Model settings
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "saved")

# Database configuration
DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'financial_data'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', '')
}

# API keys
API_KEYS = {
    'ALPHAVANTAGE': os.getenv('API_KEY_ALPHAVANTAGE'),
    'FINANCIALMODELINGPREP': os.getenv('API_KEY_FINANCIALMODELINGPREP'),
    'NEWS': os.getenv('API_KEY_NEWS'),
    'POLYGON': os.getenv('API_KEY_POLYGON'),
    'OPENAI': os.getenv('OPENAI_API_KEY'),
    'HUGGINGFACE': os.getenv('HUGGINGFACE_API_KEY')
}

# LLM configuration
LLM_CONFIG = {
    'model_name': os.getenv('LLM_MODEL_NAME', 'gpt-oss:20b'),
    'temperature': float(os.getenv('LLM_TEMPERATURE', '0.1')),
    'max_tokens': int(os.getenv('LLM_MAX_TOKENS', '2048')),
    'base_url': os.getenv('LLM_BASE_URL', 'http://localhost:11434'),
    'provider': os.getenv('LLM_PROVIDER', 'ollama')
}

# Market data configuration
MARKET_CONFIG = {
    'default_symbols': DEFAULT_SYMBOLS,
    'sp500_symbols': FULL_SP500_SYMBOLS or FALLBACK_SP500_SYMBOLS,
    'nasdaq_symbols': [
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META',
        'AVGO', 'COST', 'NFLX', 'ADBE', 'PEP', 'CSCO', 'TMUS', 'AMD',
        'INTC', 'CMCSA', 'TXN', 'QCOM', 'HON', 'INTU', 'AMAT', 'BKNG',
        'ISRG', 'ADP', 'SBUX', 'GILD', 'MU', 'LRCX', 'ADI', 'MDLZ',
        'REGN', 'KLAC', 'PYPL', 'ATVI', 'MRVL', 'ORLY', 'CSX', 'FTNT'
    ],
    'update_frequency': 'hourly',
    'lookback_days': 365,
    'sp500_daily_backfill_start': '1991-01-01',
    'sp500_hourly_backfill_period': '6mo',
    'daily_model_training_period': os.getenv('DAILY_MODEL_TRAINING_PERIOD', '10y'),
    'daily_prediction_refresh_period': os.getenv('DAILY_PREDICTION_REFRESH_PERIOD', '1y'),
    'hourly_model_training_period': os.getenv('HOURLY_MODEL_TRAINING_PERIOD', '6mo'),
    'hourly_prediction_refresh_period': os.getenv('HOURLY_PREDICTION_REFRESH_PERIOD', '5d'),
    'download_batch_size': int(os.getenv('DOWNLOAD_BATCH_SIZE', '25')),
}

# Data ingestion configuration (universes, incremental collection, scheduler)
INGESTION_CONFIG = {
    # Broad market index tickers collected at both resolutions.
    'index_symbols': ['^GSPC', '^IXIC', '^DJI', '^RUT'],
    # The 11 GICS sector ETFs (stable, cheap-to-fetch sector series).
    'sector_etf_symbols': ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC'],
    # Index whose constituents are tracked with point-in-time membership.
    'constituents_index': '^GSPC',
    # Universe names used by the scheduled jobs.
    'hourly_universe': os.getenv('INGESTION_HOURLY_UNIVERSE', 'all'),
    'daily_universe': os.getenv('INGESTION_DAILY_UNIVERSE', 'all'),
    # Sector aggregate series ('equal_weight' is the only implemented method).
    'aggregate_method': os.getenv('AGGREGATE_METHOD', 'equal_weight'),
    'aggregate_recompute_days': int(os.getenv('AGGREGATE_RECOMPUTE_DAYS', '7')),
    # Backfill horizon for daily history; hourly is capped by the provider (~730d).
    'backfill_start': os.getenv('BACKFILL_START', '1991-01-01'),
    # Fetching behavior.
    'batch_size': int(os.getenv('DOWNLOAD_BATCH_SIZE', '25')),
    'inter_batch_delay_s': float(os.getenv('INTER_BATCH_DELAY_S', '1.0')),
    'max_retries': int(os.getenv('PROVIDER_MAX_RETRIES', '3')),
    # Background scheduler (dedicated service; disabled unless explicitly enabled).
    'scheduler_enabled': env_flag('SCHEDULER_ENABLED', default=False),
    'scheduler_timezone': os.getenv('SCHEDULER_TIMEZONE', 'America/New_York'),
    'hourly_collect_minute': int(os.getenv('HOURLY_COLLECT_MINUTE', '5')),
    'daily_collect_time': os.getenv('DAILY_COLLECT_TIME', '17:30'),  # scheduler timezone
    'membership_refresh_day': os.getenv('MEMBERSHIP_REFRESH_DAY', 'sat'),
    'membership_refresh_time': os.getenv('MEMBERSHIP_REFRESH_TIME', '08:00'),
    'gap_repair_enabled': env_flag('GAP_REPAIR_ENABLED', default=True),
    'exchange_calendar': os.getenv('EXCHANGE_CALENDAR', 'XNYS'),
}

# Model training configuration
MODEL_CONFIG = {
    'sequence_length': 60,
    'batch_size': 32,
    'epochs': 80,
    'learning_rate': 0.0007,
    'train_test_split': 0.8,
    'features': [
        'open', 'high', 'low', 'close', 'volume',
        'sma_20', 'ema_12', 'rsi', 'macd', 'macd_signal',
        'bollinger_upper', 'bollinger_lower',
        'return_1', 'return_5', 'volatility_10', 'momentum_10',
        'price_to_sma_20', 'price_to_ema_12', 'volume_ratio_10',
        'high_low_range', 'open_close_range'
    ],
    'fine_tune_learning_rate': 0.00015,
    'fine_tune_epochs': 6,
    'monitoring': {
        'cooldown_predictions': 2,
        'lookback_evaluations': 12,
    },
    'interval_overrides': {
        '1h': {
            'sequence_length': 72,
            'epochs': 50,
            'learning_rate': 0.0007,
            'batch_size': 32,
            'min_training_rows': 140,
            'fine_tune_epochs': 4,
            'recent_tune_window': 240,
            'accuracy_floor_pct': 94.0,
            'allowed_accuracy_drop_pct': 1.5,
            'consecutive_drop_limit': 4,
        },
        '1d': {
            'sequence_length': 90,
            'epochs': 90,
            'learning_rate': 0.0005,
            'batch_size': 16,
            'min_training_rows': 220,
            'fine_tune_epochs': 8,
            'recent_tune_window': 504,
            'accuracy_floor_pct': 96.0,
            'allowed_accuracy_drop_pct': 1.0,
            'consecutive_drop_limit': 3,
        },
    },
}
