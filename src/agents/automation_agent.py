from langchain.agents import initialize_agent, Tool
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from crewai import Agent, Task, Crew
import asyncio
from datetime import datetime, timedelta

# Added imports for the custom LLM
from langchain.llms.base import LLM
from typing import Optional, List, Dict, Any

class AutomationAgent:
    def __init__(self, config, db_manager, data_collector, ml_predictor):
        self.config = config
        self.db_manager = db_manager
        self.data_collector = data_collector
        self.ml_predictor = ml_predictor

        try:
            # replace deprecated Ollama class
            from langchain_ollama import ChatOllama
            self.llm = ChatOllama(
                model=config['LLM_CONFIG']['model_name'],
                base_url=config['LLM_CONFIG']['base_url'],
                #request_timeout=1200.0  # 20-minute timeout
                )
            print(f"Connected the {config['LLM_CONFIG']['model_name']}")
            print(f"   at {config['LLM_CONFIG']['base_url']}")
        except Exception as e:
            try:
                # This fallback will now work with the updated requirements
                from langchain_openai import ChatOpenAI
                openai_api_key = config['API_KEYS'].get('OPENAI_API_KEY') or config['API_KEYS'].get('OPENAI')
                if openai_api_key:
                    self.llm = ChatOpenAI(
                        temperature=config['LLM_CONFIG'].get('temperature', 0.1),
                        openai_api_key=openai_api_key
                    )
                    print(f"Using {config['LLM_CONFIG']['model_name']}. Be careful of the charges!")
                else:
                    raise ValueError("No OpenAI API key found")
            except Exception as e2:
                print(f"⚠️ Cannot connect to Ollama: {e}")
                print(f"⚠️ Cannot fall back to OpenAI: {e2}")
                print("⚠️ Running in non-LLM mode with limited functionality")

                # Corrected implementation of the mock LLM
                class SimpleMockLLM(LLM):
                    """A very simple mock LLM implementation."""
                    model_name: str = "fake-llm"
                    provider: str = "fake-provider"

                    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
                        return "This is a mock response for testing without an LLM."

                    @property
                    def _llm_type(self) -> str:
                        """Return type of LLM."""
                        return "simple_mock"

                    @property
                    def _identifying_params(self) -> Dict[str, Any]:
                        return {"model_name": self.model_name, "provider": self.provider}

                self.llm = SimpleMockLLM()

        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.setup_crew_agents()

    def setup_crew_agents(self):
        """Setup CrewAI agents for different tasks"""

        # Data Collection Agent
        self.data_agent = Agent(
            role='Data Collection Specialist',
            goal='Collect and validate financial market data from various sources',
            backstory="""You are an expert in financial data collection with deep knowledge
            of market data APIs and data quality validation. You ensure data accuracy and completeness.""",
            llm=self.llm,
            verbose=True
        )

        # ML Training Agent
        self.ml_agent = Agent(
            role='Machine Learning Engineer',
            goal='Train and optimize predictive models for financial forecasting',
            backstory="""You are a skilled ML engineer specializing in time series forecasting
            and neural networks. You focus on model performance and accuracy.""",
            llm=self.llm,
            verbose=True
        )

        # Prediction Agent
        self.prediction_agent = Agent(
            role='Market Prediction Analyst',
            goal='Generate accurate market predictions and assess confidence levels',
            backstory="""You are a financial analyst expert in market prediction and risk assessment.
            You provide insights on market trends and prediction reliability.""",
            llm=self.llm,
            verbose=True
        )

        # Dashboard Agent
        self.dashboard_agent = Agent(
            role='Data Visualization Specialist',
            goal='Create insightful dashboards and visualizations for market data',
            backstory="""You are a data visualization expert who creates compelling and
            informative dashboards for financial professionals.""",
            llm=self.llm,
            verbose=True
        )

    def create_data_collection_task(self, symbols, exchange):
        """Create data collection task"""
        return Task(
            description=f"""
            Collect hourly market data for {exchange} symbols: {symbols}

            Steps:
            1. Fetch data from financial APIs
            2. Validate data quality and completeness
            3. Store data in PostgreSQL database
            4. Report collection status and any issues

            Success criteria:
            - All symbols have current data
            - No missing timestamps in the last 24 hours
            - Data passes quality validation checks
            """,
            expected_output=f"""
            A summary of collected data for {symbols} including:
            - Number of records collected
            - Timestamp range
            - Data quality metrics
            - Any issues encountered
            """,
            agent=self.data_agent
        )

    def create_ml_training_task(self, symbol):
        """Create ML training task"""
        return Task(
            description=f"""
            Train LSTM model for {symbol} price prediction

            Steps:
            1. Retrieve historical data from database
            2. Prepare features and technical indicators
            3. Train LSTM model with optimal parameters
            4. Validate model performance
            5. Save trained model for predictions

            Success criteria:
            - Model achieves RMSE < 5% of price range
            - Training completes without errors
            """,
            expected_output=f"""
            A training report for {symbol} model including:
            - Model architecture details
            - Training/validation loss curves
            - Performance metrics (RMSE, MAE, etc.)
            - Feature importance if available
            - Model is saved and ready for predictions
            """,
            agent=self.ml_agent
        )

    def create_prediction_task(self, symbols):
        """Create prediction task"""
        return Task(
            description=f"""
            Generate price predictions for symbols: {symbols}

            Steps:
            1. Load trained models for each symbol
            2. Fetch latest market data
            3. Generate price predictions for next hour
            4. Calculate confidence scores
            5. Store predictions in database

            Success criteria:
            - Predictions generated for all symbols
            - Confidence scores calculated
            - Results stored in database
            """,
            expected_output=f"""
            Prediction results for {symbols} including:
            - Predicted price movements (up/down)
            - Price targets with confidence intervals
            - Key factors influencing predictions
            - Recommended trading actions
            """,
            agent=self.prediction_agent
        )

    def create_dashboard_task(self):
        """Create dashboard update task"""
        return Task(
            description="""
            Update real-time dashboard with latest predictions

            Steps:
            1. Retrieve latest market data and predictions
            2. Update visualization charts and metrics
            3. Refresh dashboard display
            4. Ensure all components are working correctly

            Success criteria:
            - Dashboard displays current data
            - All visualizations are updated
            - No errors in dashboard components
            """,
            agent=self.dashboard_agent
        )

    async def run_full_pipeline(self):
        """Run the complete pipeline orchestration"""

        # Step 1: Data Collection
        print("🔄 Starting data collection phase...")

        sp500_task = self.create_data_collection_task(
            self.config['MARKET_CONFIG']['sp500_symbols'][:10],  # Limit for demo
            'SP500'
        )

        nasdaq_task = self.create_data_collection_task(
            self.config['MARKET_CONFIG']['nasdaq_symbols'][:10],  # Limit for demo
            'NASDAQ'
        )

        # Execute data collection
        data_crew = Crew(
            agents=[self.data_agent],
            tasks=[sp500_task, nasdaq_task],
            verbose=True
        )

        data_results = data_crew.kickoff()
        print("✅ Data collection completed")

        # Step 2: Model Training (for new symbols or periodic retraining)
        print("🔄 Starting model training phase...")

        training_tasks = []
        key_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']  # Focus on key symbols

        for symbol in key_symbols:
            training_tasks.append(self.create_ml_training_task(symbol))

        ml_crew = Crew(
            agents=[self.ml_agent],
            tasks=training_tasks,
            verbose=True
        )

        training_results = ml_crew.kickoff()
        print("✅ Model training completed")

        # Step 3: Generate Predictions
        print("🔄 Starting prediction phase...")

        prediction_task = self.create_prediction_task(key_symbols)

        prediction_crew = Crew(
            agents=[self.prediction_agent],
            tasks=[prediction_task],
            verbose=True
        )

        prediction_results = prediction_crew.kickoff()
        print("✅ Predictions generated")

        # Step 4: Update Dashboard
        print("🔄 Updating dashboard...")

        dashboard_task = self.create_dashboard_task()

        dashboard_crew = Crew(
            agents=[self.dashboard_agent],
            tasks=[dashboard_task],
            verbose=True
        )

        dashboard_results = dashboard_crew.kickoff()
        print("✅ Dashboard updated")

        return {
            'data_collection': data_results,
            'model_training': training_results,
            'predictions': prediction_results,
            'dashboard': dashboard_results,
            'timestamp': datetime.now()
        }

    def run_hourly_update(self):
        """Run hourly prediction updates"""
        print("🔄 Running hourly update...")

        # Quick data refresh and prediction update
        key_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

        # Collect latest data
        self.data_collector.execute_collection_task(
            f"Collect latest hourly data for symbols: {key_symbols}"
        )

        # Generate predictions
        prediction_task = self.create_prediction_task(key_symbols)
        prediction_crew = Crew(
            agents=[self.prediction_agent],
            tasks=[prediction_task],
            verbose=True
        )

        results = prediction_crew.kickoff()

        # Update dashboard
        dashboard_task = self.create_dashboard_task()
        dashboard_crew = Crew(
            agents=[self.dashboard_agent],
            tasks=[dashboard_task],
            verbose=True
        )

        dashboard_crew.kickoff()

        print("✅ Hourly update completed")
        return results

    def schedule_operations(self):
        """Schedule regular operations"""
        import schedule
        import time

        # Schedule hourly updates
        schedule.every().hour.do(self.run_hourly_update)

        # Schedule daily full pipeline (for retraining)
        schedule.every().day.at("02:00").do(lambda: asyncio.run(self.run_full_pipeline()))

        print("📅 Scheduler started. Running continuous operations...")

        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute