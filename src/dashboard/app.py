import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time
import os
import sys
import matplotlib

# non-interactive backend to prevent issues with dashboard when running in the background.
matplotlib.use('Agg')
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.database import DatabaseManager
from src.config.settings import DATABASE_CONFIG

class FinancialDashboard:
    """
    Main dashboard class to integrate all modules and functionalities of the dashboard.
    """
    def __init__(self, db_manager):
        """
        Initializes the dashboard with a database manager.
        """
        self.db_manager = db_manager

    def load_data(self, symbol, hours=168, pred_limit=24, use_predictions=True):  # Default to 7 days of data
        """
        Loads the latest market data and predictions for a given stock symbol.
        """
        market_data = self.db_manager.get_latest_data(symbol, limit_rows=hours)

        predictions = pd.DataFrame()  # Empty dataframe for predictions
        
        # Fetch predictions separately if use_predictions
        if use_predictions:
            with self.db_manager.Session() as session:
                predictions_query = session.query(
                    self.db_manager.PredictionResults
                ).filter(
                    self.db_manager.PredictionResults.symbol == symbol
                ).order_by(
                    self.db_manager.PredictionResults.prediction_timestamp.desc()
                ).limit(pred_limit)  # Get the predictions
                
                predictions = pd.read_sql(predictions_query.statement, self.db_manager.engine)
                
        return market_data, predictions

    def create_price_chart(self, market_data, predictions, symbol):
        """
        An interactive plot for stock price comparison betweem data and model.
        """
        fig = go.Figure()
        
        # Reference for comparison is the closing prices from market data.
        fig.add_trace(go.Scatter(
            x=market_data['timestamp'],
            y=market_data['close_price'],
            mode='lines',
            name='Tru Price',
            line=dict(color='dodgerblue', width=2)
        ))
        
        # Overlay predictions on the data, if available.
        if not predictions.empty:
            fig.add_trace(go.Scatter(
                x=predictions['prediction_timestamp'],
                y=predictions['predicted_price'],
                mode='markers+lines',
                name='Predicted Price',
                line=dict(color='teal', width=2, dash='dash'),
                marker=dict(size=8, symbol='x-thin')
            ))
        
        fig.update_layout(
            title=f'{symbol} stock price',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            hovermode='x unified',
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def create_technical_indicators_chart(self, market_data, symbol):
        """
        Additional plots for key technical indicators (e.g., moving averages and Bollinger bands).
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=market_data['timestamp'], y=market_data['close_price'], name='Close Price', line=dict(color='lightskyblue', width=1)))
        if 'sma_20' in market_data.columns and market_data['sma_20'].notna().any():
            fig.add_trace(go.Scatter(x=market_data['timestamp'], y=market_data['sma_20'], name='20-Day SMA', line=dict(color='orange')))
        
        if 'bollinger_upper' in market_data.columns and market_data['bollinger_upper'].notna().any():
            fig.add_trace(go.Scatter(x=market_data['timestamp'], y=market_data['bollinger_upper'], name='Bollinger Upper', line=dict(color='gray', dash='dash')))
            fig.add_trace(go.Scatter(x=market_data['timestamp'], y=market_data['bollinger_lower'], name='Bollinger Lower', line=dict(color='gray', dash='dash'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'))
        
        fig.update_layout(
            title=f'{symbol} - stock indicators',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig

    def calculate_metrics(self, market_data, predictions):
        """
        Calculates key performance metrics from the latest data.
        """
        if market_data.empty:
            return {'current_price': 0, 'price_change_24h': 0, 'avg_volume': 0, 'prediction_accuracy': 0}

        current_price = market_data['close_price'].iloc[-1]
        price_change_24h = ((current_price - market_data['close_price'].iloc[0]) / market_data['close_price'].iloc[0] * 100) if len(market_data) > 1 else 0
        avg_volume = market_data['volume'].mean()
        
        accuracy = 0
        if not predictions.empty and 'actual_price' in predictions.columns:
            predictions_with_actual = predictions.dropna(subset=['actual_price'])
            if not predictions_with_actual.empty:
                mape = np.mean(np.abs((predictions_with_actual['actual_price'] - predictions_with_actual['predicted_price']) / predictions_with_actual['actual_price'])) * 100
                accuracy = max(0, 100 - mape)
        
        return {
            'current_price': current_price,
            'price_change_24h': price_change_24h,
            'avg_volume': avg_volume,
            'prediction_accuracy': accuracy
        }

def main():
    st.set_page_config(
        page_title="Financial Agent Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Personal Financial AI Agent Assistant")
    st.markdown("### Personal automated stock market tracker.")
    st.markdown("#### This pipeline is powered by multi-agent LLMs to monitor stock market prices and make short-term predicctions for personal inverstors. Please use it at your own risk since some predictions may contain small errors.")
    st.markdown("#### If you use this app, please make sure to support the project on Github and credit the author.")

    # --- Flexible Database Connection ---
    @st.cache_resource
    def get_db_manager():
        """
        This function creates and caches the database manager.
        It works both with PostgreSQL and SQLite.
        """
        st.info("Connecting to the database...")
        db_manager = DatabaseManager(DATABASE_CONFIG, use_sqlite_fallback=True)
        st.success(f"Connected to {'SQLite' if db_manager.is_sqlite else 'PostgreSQL'}!")
        return db_manager

    db_manager = get_db_manager()
    dashboard = FinancialDashboard(db_manager)
    
    # Sidebar controls
    ## TODO: Add more settings here to control data range, model parameters, etc.
    st.sidebar.header("Dashboard Settings")
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA']
    selected_symbol = st.sidebar.selectbox("Select  Stock to Analyze", symbols)
    auto_refresh = st.sidebar.checkbox("Auto refresh 30s", value=True)
    
    # Data loading
    try:
        market_data, predictions = dashboard.load_data(selected_symbol)
        if market_data.empty:
            st.warning(f"Data for {selected_symbol} is not available yet. The data agent might still be warming up. Please check back in a moment.")
            return
        metrics = dashboard.calculate_metrics(market_data, predictions)
    except Exception as e:
        st.error(f"Failed to fetch the data: {e}")
        return

    # Main Content
    st.subheader(f"Live Analysis for {selected_symbol}")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    price_delta = f"{metrics['price_change_24h']:.2f}% (24h)"
    col1.metric("Current Price", f"${metrics['current_price']:.2f}", price_delta)
    col2.metric("Avg. Volume (24h)", f"{metrics['avg_volume']/1_000_000:.2f}M")
    col3.metric("Prediction Accuracy", f"{metrics['prediction_accuracy']:.2f}%")
    latest_pred = predictions['predicted_price'].iloc[0] if not predictions.empty else 0
    col4.metric("Latest Prediction", f"${latest_pred:.2f}")

    # Charts
    st.plotly_chart(dashboard.create_price_chart(market_data, predictions, selected_symbol), use_container_width=True)
    st.plotly_chart(dashboard.create_technical_indicators_chart(market_data, selected_symbol), use_container_width=True)
    
    # Agent Status (Static example, as real-time updates are complex) ---
    st.sidebar.header("👋 Meet Your AI Team")
    st.sidebar.info("**Data Collector:** I'm always fetching the latest market data to keep our analysis fresh.")
    st.sidebar.info("**ML Trainer:** I train our prediction models to be as sharp as possible.")
    st.sidebar.info("**Prediction Agent:** I use the trained models to forecast what's next.")

    # Auto refresh
    if auto_refresh:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()

