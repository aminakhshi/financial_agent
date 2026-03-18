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
    'download_batch_size': int(os.getenv('DOWNLOAD_BATCH_SIZE', '25')),
}

# Model training configuration
MODEL_CONFIG = {
    'sequence_length': 60,  # 60 hours of data
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'train_test_split': 0.8,
    'features': ['open', 'high', 'low', 'close', 'volume', 'sma_20', 'rsi', 'macd']
}
