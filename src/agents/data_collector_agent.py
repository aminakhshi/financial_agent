import yfinance as yf
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta
import asyncio
import schedule
import time
import os
from typing import Iterable, List, Optional

try:
    from loguru import logger
except Exception:  # pragma: no cover - fallback for minimal environments
    import logging

    logger = logging.getLogger(__name__)

try:
    from langchain.llms.base import LLM as BaseLLM
except ImportError:
    from langchain_core.language_models.llms import LLM as BaseLLM

try:
    from langchain.agents import initialize_agent, Tool
except ImportError:
    class Tool:  # pragma: no cover - compatibility shim
        def __init__(self, name, func, description):
            self.name = name
            self.func = func
            self.description = description

    class _FallbackToolRunner:
        def __init__(self, tools):
            self._tool_map = {t.name: t.func for t in tools}

        def run(self, prompt):
            return (
                "initialize_agent is unavailable in the installed langchain version. "
                "Use direct methods like fetch_yfinance_data/fetch_alpha_vantage_data/store_market_data. "
                f"Prompt was: {prompt[:180]}"
            )

    def initialize_agent(tools, llm, agent_type=None, verbose=False):
        return _FallbackToolRunner(tools)


class DataCollectorAgent:
    def __init__(self, config, db_manager):
        self.config = config
        self.db_manager = db_manager
        try:
            # replace deprecated Ollama class
            # from langchain_community.llms import Ollama
            from langchain_ollama import ChatOllama
            self.llm = ChatOllama(
                model=config['LLM_CONFIG']['model_name'],
                base_url=config['LLM_CONFIG']['base_url'],
                request_timeout=float(os.getenv('LLM_REQUEST_TIMEOUT', '120'))
                )
            # self.llm = Ollama(
            #     model=config['LLM_CONFIG']['model_name'],
            #     base_url=config['LLM_CONFIG']['base_url']
            # )
            print(f"Data collector is using model {config['LLM_CONFIG']['model_name']}.")
        except Exception as e:
            try:
                # fallback to using OpenAI
                from langchain_openai import ChatOpenAI
                openai_api_key = config['API_KEYS'].get('OPENAI_API_KEY') or config['API_KEYS'].get('OPENAI')
                if openai_api_key:
                    self.llm = ChatOpenAI(
                        temperature=config['LLM_CONFIG'].get('temperature', 0.1),
                        openai_api_key=openai_api_key
                    )
                    print(
                        f"Data collector is using the OpenAI-compatible model "
                        f"{config['LLM_CONFIG']['model_name']}."
                    )
            except Exception as e2:
                print("Data collector could not connect to the configured LLM backend.")
                print("Running in non-LLM mode with limited functionality.")

            # Create a custom LLM that will work with litellm
            from typing import Optional, List, Dict, Any
            
            class SimpleMockLLM(BaseLLM):
                """A very simple mock LLM implementation."""
                model_name: str = "fake-llm"
                provider: str = "fake-provider"
                
                def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
                    return "Mock response for testing"
                
                @property
                def _llm_type(self) -> str:
                    """Return type of LLM."""
                    return "simple_mock"
                    
                @property
                def _identifying_params(self) -> Dict[str, Any]:
                    return {"model_name": self.model_name, "provider": self.provider}
            
            self.llm = SimpleMockLLM()
            
        alpha_vantage_key = config['API_KEYS'].get('ALPHA_VANTAGE_API_KEY') or config['API_KEYS'].get('ALPHAVANTAGE') or 'demo'
        self.alpha_vantage = TimeSeries(
            key=alpha_vantage_key,
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
    
    def _normalize_symbols(self, symbols: Iterable[str]) -> List[str]:
        seen = set()
        normalized: List[str] = []
        for symbol in symbols:
            cleaned = str(symbol).strip().upper()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            normalized.append(cleaned)
        return normalized

    def _history_request_kwargs(
        self,
        period: Optional[str],
        interval: str,
        start: Optional[str],
        end: Optional[str],
    ) -> dict:
        kwargs = {
            "interval": interval,
            "auto_adjust": False,
            "actions": False,
        }
        if start or end:
            kwargs["start"] = start
            if end:
                kwargs["end"] = end
        else:
            kwargs["period"] = period or "1mo"
        return kwargs

    def _format_history_frame(self, data: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
        if data is None or data.empty:
            return pd.DataFrame()

        frame = data.copy().reset_index()
        timestamp_column = None
        for candidate in ("Datetime", "Date"):
            if candidate in frame.columns:
                timestamp_column = candidate
                break
        if timestamp_column is None:
            timestamp_column = frame.columns[0]

        rename_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        frame = frame.rename(columns=rename_map)
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(frame.columns):
            return pd.DataFrame()

        frame["symbol"] = symbol
        frame["timestamp"] = pd.to_datetime(frame[timestamp_column], utc=True)
        frame["timeframe"] = interval
        frame = frame[["symbol", "timestamp", "open", "high", "low", "close", "volume", "timeframe"]]

        # yfinance often returns pre-listing rows as all-NaN when a wide start date
        # is requested for newer symbols. Drop those rows so historical backfills
        # continue from the first real trading session instead of failing the batch.
        frame = frame.dropna(subset=["open", "high", "low", "close"])
        frame = frame[frame["timestamp"].notna()]
        if frame.empty:
            return pd.DataFrame()

        # Keep rows only when numeric fields are valid. Volume can be zero, but not null.
        frame["volume"] = frame["volume"].fillna(0.0)
        numeric_columns = ["open", "high", "low", "close", "volume"]
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=numeric_columns)

        return frame.reset_index(drop=True)

    def _download_batch(
        self,
        batch: List[str],
        period: Optional[str],
        interval: str,
        start: Optional[str],
        end: Optional[str],
    ) -> pd.DataFrame:
        kwargs = self._history_request_kwargs(period=period, interval=interval, start=start, end=end)
        return yf.download(
            tickers=batch,
            group_by="ticker",
            progress=False,
            threads=True,
            **kwargs,
        )

    def _fetch_single_symbol(
        self,
        symbol: str,
        period: Optional[str],
        interval: str,
        start: Optional[str],
        end: Optional[str],
    ) -> pd.DataFrame:
        kwargs = self._history_request_kwargs(period=period, interval=interval, start=start, end=end)
        data = yf.Ticker(symbol).history(**kwargs)
        return self._format_history_frame(data, symbol, interval)

    def _fetch_batch_frames(
        self,
        batch: List[str],
        period: Optional[str],
        interval: str,
        start: Optional[str],
        end: Optional[str],
    ) -> List[pd.DataFrame]:
        batch_frames: List[pd.DataFrame] = []
        try:
            data = self._download_batch(batch, period=period, interval=interval, start=start, end=end)
            if isinstance(data.columns, pd.MultiIndex):
                available_symbols = set(data.columns.get_level_values(0))
                for symbol in batch:
                    if symbol not in available_symbols:
                        continue
                    formatted = self._format_history_frame(data[symbol], symbol, interval)
                    if not formatted.empty:
                        batch_frames.append(formatted)
            else:
                symbol = batch[0]
                formatted = self._format_history_frame(data, symbol, interval)
                if not formatted.empty:
                    batch_frames.append(formatted)
        except Exception as exc:
            logger.warning("Error fetching batched data for {}: {}", ", ".join(batch), exc)

        fetched_symbols = {frame["symbol"].iloc[0] for frame in batch_frames if not frame.empty}
        for symbol in batch:
            if symbol in fetched_symbols:
                continue
            try:
                formatted = self._fetch_single_symbol(
                    symbol,
                    period=period,
                    interval=interval,
                    start=start,
                    end=end,
                )
                if not formatted.empty:
                    batch_frames.append(formatted)
            except Exception as exc:
                logger.warning("Error fetching data for {}: {}", symbol, exc)

        return batch_frames

    def iter_yfinance_data_batches(
        self,
        symbols,
        period="1d",
        interval="1h",
        start: Optional[str] = None,
        end: Optional[str] = None,
        batch_size: Optional[int] = None,
    ):
        normalized_symbols = self._normalize_symbols(symbols)
        if not normalized_symbols:
            return

        batch_size = max(int(batch_size or self.config["MARKET_CONFIG"].get("download_batch_size", 25)), 1)
        total_batches = max((len(normalized_symbols) + batch_size - 1) // batch_size, 1)
        range_label = f"start={start}, end={end or 'latest'}" if start or end else f"period={period}"

        for batch_number, index in enumerate(range(0, len(normalized_symbols), batch_size), start=1):
            batch = normalized_symbols[index:index + batch_size]
            logger.info(
                "Fetching provider batch {}/{} for {} symbols using {} {}.",
                batch_number,
                total_batches,
                len(batch),
                interval,
                range_label,
            )
            batch_frames = self._fetch_batch_frames(
                batch,
                period=period,
                interval=interval,
                start=start,
                end=end,
            )
            if not batch_frames:
                logger.warning(
                    "No provider data was returned for batch {}/{} ({}).",
                    batch_number,
                    total_batches,
                    ", ".join(batch),
                )
                time.sleep(0.1)
                continue

            batch_df = pd.concat(batch_frames, ignore_index=True)
            logger.info(
                "Fetched {} rows across {} symbols in batch {}/{}.",
                len(batch_df),
                batch_df["symbol"].nunique(),
                batch_number,
                total_batches,
            )
            yield batch_df
            time.sleep(0.1)

    def fetch_yfinance_data(
        self,
        symbols,
        period="1d",
        interval="1h",
        start: Optional[str] = None,
        end: Optional[str] = None,
        batch_size: Optional[int] = None,
    ):
        """Fetch market data using batched yfinance downloads."""
        all_data = list(
            self.iter_yfinance_data_batches(
                symbols,
                period=period,
                interval=interval,
                start=start,
                end=end,
                batch_size=batch_size,
            )
        )
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
