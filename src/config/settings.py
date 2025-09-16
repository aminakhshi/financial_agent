import os
from dotenv import load_dotenv

load_dotenv()

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
    'alpha_vantage': os.getenv('ALPHA_VANTAGE_API_KEY'),
    'openai': os.getenv('OPENAI_API_KEY'),
    'huggingface': os.getenv('HUGGINGFACE_API_KEY')
}

# LLM configuration
LLM_CONFIG = {
    'model_name': 'llama3:8b',  # or 'mistral:7b'
    'temperature': 0.1,
    'max_tokens': 2048,
    'base_url': 'http://localhost:11434'  # Ollama server
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