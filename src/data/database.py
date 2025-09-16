from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
from datetime import datetime

Base = declarative_base()

class MarketData(Base):
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False)
    exchange = Column(String(10), nullable=False)  # 'SP500' or 'NASDAQ'
    timestamp = Column(DateTime, nullable=False)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    # Technical indicators (calculated during preprocessing)
    sma_20 = Column(Float)
    ema_12 = Column(Float)
    rsi = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    bollinger_upper = Column(Float)
    bollinger_lower = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)

# Optimized indexes for time-series queries
Index('idx_symbol_timestamp', MarketData.symbol, MarketData.timestamp)
Index('idx_exchange_timestamp', MarketData.exchange, MarketData.timestamp)
Index('idx_timestamp', MarketData.timestamp)

class PredictionResults(Base):
    __tablename__ = 'prediction_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False)
    prediction_timestamp = Column(DateTime, nullable=False)
    predicted_price = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=False)
    model_version = Column(String(50), nullable=False)
    actual_price = Column(Float)  # Filled later for evaluation
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    def __init__(self, config):
        self.engine = create_engine(
            f"postgresql://{config['user']}:{config['password']}@"
            f"{config['host']}:{config['port']}/{config['database']}"
        )
        self.Session = sessionmaker(bind=self.engine)
        
    def create_tables(self):
        Base.metadata.create_all(self.engine)
        
    def insert_market_data(self, df, exchange):
        session = self.Session()
        try:
            for _, row in df.iterrows():
                market_data = MarketData(
                    symbol=row['symbol'],
                    exchange=exchange,
                    timestamp=row['timestamp'],
                    open_price=row['open'],
                    high_price=row['high'],
                    low_price=row['low'],
                    close_price=row['close'],
                    volume=row['volume']
                )
                session.merge(market_data)  # Use merge to handle duplicates
            session.commit()
        finally:
            session.close()
            
    def get_latest_data(self, symbol, hours=24):
        session = self.Session()
        try:
            query = session.query(MarketData).filter(
                MarketData.symbol == symbol
            ).order_by(MarketData.timestamp.desc()).limit(hours)
            return pd.read_sql(query.statement, self.engine)
        finally:
            session.close()