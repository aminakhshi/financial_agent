import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# API Keys - load from environment variables
API_KEY_ALPHAVANTAGE = os.environ.get("API_KEY_ALPHAVANTAGE", "")
API_KEY_FINANCIALMODELINGPREP = os.environ.get("API_KEY_FINANCIALMODELINGPREP", "")
API_KEY_NEWS = os.environ.get("API_KEY_NEWS", "")
API_KEY_POLYGON = os.environ.get("API_KEY_POLYGON", "")

# Database settings
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///../financial_data.db")

# Model settings
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "saved")

# Database configuration
DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'financial_data'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'password')
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
    'model_name': 'ollama/gpt-oss:20b',  # Added ollama/ prefix
    'temperature': 0.1,
    'max_tokens': 2048,
    'base_url': 'http://localhost:11434',  # Ollama server
    'provider': 'ollama'  # Specify the provider
}

# Market data configuration
MARKET_CONFIG = {
    'sp500_symbols': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B',
        'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'ABBV',
        'PFE', 'AVGO', 'COST', 'DIS', 'KO', 'MRK', 'PEP', 'TMO', 'WMT',
        'ABT', 'BAC', 'CRM', 'CSCO', 'ACN', 'LIN', 'ADBE', 'MCD', 'VZ',
        'DHR', 'NFLX', 'CMCSA', 'NKE', 'TXN', 'NEE', 'AMD', 'PM', 'RTX',
        'UPS', 'T', 'LOW'  # Top 50 index symbols
    ],
    'nasdaq_symbols': [
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META',
        'AVGO', 'COST', 'NFLX', 'ADBE', 'PEP', 'CSCO', 'TMUS', 'AMD',
        'INTC', 'CMCSA', 'TXN', 'QCOM', 'HON', 'INTU', 'AMAT', 'BKNG',
        'ISRG', 'ADP', 'SBUX', 'GILD', 'MU', 'LRCX', 'ADI', 'MDLZ',
        'REGN', 'KLAC', 'PYPL', 'ATVI', 'MRVL', 'ORLY', 'CSX', 'FTNT'
    ],
    'update_frequency': 'hourly',
    'lookback_days': 365
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