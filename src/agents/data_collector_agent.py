import yfinance as yf
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from langchain.agents import initialize_agent, Tool
from langchain.llms import Ollama
from datetime import datetime, timedelta
import asyncio
import schedule
import time

class DataCollectorAgent:
    def __init__(self, config, db_manager):
        self.config = config
        self.db_manager = db_manager
        self.llm = Ollama(
            model=config['LLM_CONFIG']['model_name'],
            base_url=config['LLM_CONFIG']['base_url']
        )
        self.alpha_vantage = TimeSeries(
            key=config['API_KEYS']['alpha_vantage'],
            output_format='pandas'
        )
        
        # Define tools for the agent
        tools = [
            Tool(
                name="fetch_yfinance_data",
                func=self.fetch_yfinance_data,
                description="Fetch hourly market data using yfinance API"
            ),
            Tool(
                name="fetch_alpha_vantage_data", 
                func=self.fetch_alpha_vantage_data,
                description="Fetch intraday data using Alpha Vantage API"
            ),
            Tool(
                name="store_data",
                func=self.store_market_data,
                description="Store fetched market data in PostgreSQL database"
            )
        ]
        
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent_type="zero-shot-react-description",
            verbose=True
        )
    
    def fetch_yfinance_data(self, symbols, period="1d", interval="1h"):
        """Fetch hourly data using yfinance"""
        all_data = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval)
                
                if not data.empty:
                    data = data.reset_index()
                    data['symbol'] = symbol
                    data['timestamp'] = data['Datetime']
                    data = data[['symbol', 'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
                    data.columns = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
                    all_data.append(data)
                    
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    
    def fetch_alpha_vantage_data(self, symbol):
        """Fetch intraday data using Alpha Vantage"""
        try:
            data, meta_data = self.alpha_vantage.get_intraday(
                symbol=symbol, 
                interval='60min', 
                outputsize='compact'
            )
            
            data = data.reset_index()
            data['symbol'] = symbol
            data = data.rename(columns={
                'date': 'timestamp',
                '1. open': 'open',
                '2. high': 'high', 
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            })
            
            return data[['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            print(f"Error fetching Alpha Vantage data for {symbol}: {e}")
            return pd.DataFrame()
    
    def store_market_data(self, df, exchange):
        """Store market data in database"""
        if not df.empty:
            self.db_manager.insert_market_data(df, exchange)
            return f"Successfully stored {len(df)} records for {exchange}"
        return "No data to store"
    
    def collect_sp500_data(self):
        """Collect S&P 500 data"""
        symbols = self.config['MARKET_CONFIG']['sp500_symbols']
        data = self.fetch_yfinance_data(symbols)
        return self.store_market_data(data, 'SP500')
    
    def collect_nasdaq_data(self):
        """Collect NASDAQ data"""
        symbols = self.config['MARKET_CONFIG']['nasdaq_symbols']
        data = self.fetch_yfinance_data(symbols)
        return self.store_market_data(data, 'NASDAQ')
    
    def run_scheduled_collection(self):
        """Run data collection on schedule"""
        schedule.every().hour.do(self.collect_sp500_data)
        schedule.every().hour.do(self.collect_nasdaq_data)
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def execute_collection_task(self, task_description):
        """Use LLM agent to execute collection tasks"""
        prompt = f"""
        Task: {task_description}
        
        You are a financial data collection agent. Your job is to:
        1. Understand what type of market data is needed
        2. Use the appropriate API to fetch the data
        3. Store the data in the database
        
        Available tools: fetch_yfinance_data, fetch_alpha_vantage_data, store_data
        
        Execute the task and provide a summary of what was accomplished.
        """
        
        return self.agent.run(prompt)