import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

class FinancialDashboard:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        
    def load_data(self, symbol, hours=168):  # 7 days
        """Load market data and predictions"""
        # Get market data
        market_data = self.db_manager.get_latest_data(symbol, hours)
        
        # Get predictions
        session = self.db_manager.Session()
        try:
            predictions_query = session.query(
                self.db_manager.PredictionResults
            ).filter(
                self.db_manager.PredictionResults.symbol == symbol
            ).order_by(
                self.db_manager.PredictionResults.prediction_timestamp.desc()
            ).limit(24)  # Last 24 predictions
            
            predictions = pd.read_sql(predictions_query.statement, self.db_manager.engine)
        finally:
            session.close()
            
        return market_data, predictions
    
    def create_price_chart(self, market_data, predictions, symbol):
        """Create interactive price chart with predictions"""
        fig = go.Figure()
        
        # Add actual prices
        fig.add_trace(go.Scatter(
            x=market_data['timestamp'],
            y=market_data['close_price'],
            mode='lines',
            name='Actual Price',
            line=dict(color='blue', width=2)
        ))
        
        # Add predictions
        if not predictions.empty:
            fig.add_trace(go.Scatter(
                x=predictions['prediction_timestamp'],
                y=predictions['predicted_price'],
                mode='markers+lines',
                name='Predictions',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title=f'{symbol} - Price vs Predictions',
            xaxis_title='Time',
            yaxis_title='Price ($)',
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def create_technical_indicators_chart(self, market_data, symbol):
        """Create technical indicators chart"""
        fig = go.Figure()
        
        # Price and moving averages
        fig.add_trace(go.Scatter(
            x=market_data['timestamp'],
            y=market_data['close_price'],
            name='Close Price',
            line=dict(color='blue')
        ))
        
        if 'sma_20' in market_data.columns:
            fig.add_trace(go.Scatter(
                x=market_data['timestamp'],
                y=market_data['sma_20'],
                name='SMA 20',
                line=dict(color='orange')
            ))
        
        if 'bollinger_upper' in market_data.columns:
            fig.add_trace(go.Scatter(
                x=market_data['timestamp'],
                y=market_data['bollinger_upper'],
                name='Bollinger Upper',
                line=dict(color='gray', dash='dot')
            ))
            
            fig.add_trace(go.Scatter(
                x=market_data['timestamp'],
                y=market_data['bollinger_lower'],
                name='Bollinger Lower',
                line=dict(color='gray', dash='dot')
            ))
        
        fig.update_layout(
            title=f'{symbol} - Technical Indicators',
            xaxis_title='Time',
            yaxis_title='Price ($)',
            height=400
        )
        
        return fig
    
    def create_volume_chart(self, market_data, symbol):
        """Create volume chart"""
        fig = px.bar(
            market_data,
            x='timestamp',
            y='volume',
            title=f'{symbol} - Trading Volume'
        )
        
        fig.update_layout(height=300)
        return fig
    
    def calculate_metrics(self, market_data, predictions):
        """Calculate performance metrics"""
        current_price = market_data['close_price'].iloc[-1] if not market_data.empty else 0
        price_change_24h = ((current_price - market_data['close_price'].iloc[-24]) / 
                           market_data['close_price'].iloc[-24] * 100) if len(market_data) >= 24 else 0
        
        avg_volume = market_data['volume'].mean() if not market_data.empty else 0
        
        # Prediction accuracy (if we have actual vs predicted)
        accuracy = 0
        if not predictions.empty:
            predictions_with_actual = predictions.dropna(subset=['actual_price'])
            if not predictions_with_actual.empty:
                mape = np.mean(np.abs((predictions_with_actual['actual_price'] - 
                                     predictions_with_actual['predicted_price']) / 
                                    predictions_with_actual['actual_price'])) * 100
                accuracy = max(0, 100 - mape)
        
        return {
            'current_price': current_price,
            'price_change_24h': price_change_24h,
            'avg_volume': avg_volume,
            'prediction_accuracy': accuracy
        }

def main():
    st.set_page_config(
        page_title="AI Financial Market Predictor",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("🤖 AI-Powered Financial Market Predictor")
    st.markdown("### Real-time market analysis with LLM agents and ML predictions")
    
    # Initialize dashboard (you'll need to pass actual db_manager)
    # dashboard = FinancialDashboard(db_manager)
    
    # Sidebar controls
    st.sidebar.header("Settings")
    
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA']
    selected_symbol = st.sidebar.selectbox("Select Symbol", symbols)
    
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=True)
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    # Sample metrics (replace with real data)
    with col1:
        st.metric("Current Price", "$150.25", "2.15%")
    
    with col2:
        st.metric("24h Change", "2.15%", "0.32%")
    
    with col3:
        st.metric("Volume", "1.2M", "15%")
    
    with col4:
        st.metric("Prediction Accuracy", "92.5%", "1.2%")
    
    # Charts
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Price Chart & Predictions")
        
        # Sample chart (replace with real data)
        dates = pd.date_range(start='2024-01-01', end='2024-01-07', freq='H')
        prices = np.random.normal(150, 5, len(dates))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=prices, name="Actual Price"))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Latest Predictions")
        
        # Sample predictions table
        pred_df = pd.DataFrame({
            'Time': ['10:00', '11:00', '12:00', '13:00'],
            'Predicted': [151.2, 152.1, 150.8, 151.5],
            'Confidence': [0.92, 0.89, 0.94, 0.91]
        })
        
        st.dataframe(pred_df, use_container_width=True)
        
        st.subheader("Model Performance")
        st.progress(0.925)
        st.write("Accuracy: 92.5%")
    
    # Technical indicators
    st.subheader("Technical Indicators")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # RSI gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = 65,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "RSI"},
            gauge = {'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 70], 'color': "gray"},
                        {'range': [70, 100], 'color': "lightgray"}],
                    'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 90}}))
        
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # MACD
        macd_dates = pd.date_range(start='2024-01-01', end='2024-01-07', freq='D')
        macd_values = np.random.normal(0, 0.5, len(macd_dates))
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=macd_dates, y=macd_values, name="MACD"))
        fig.update_layout(title="MACD", height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    # Agent Status
    st.subheader("🤖 Agent Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.success("✅ Data Collector Agent")
        st.write("Last update: 2 min ago")
    
    with col2:
        st.success("✅ ML Training Agent")
        st.write("Model accuracy: 92.5%")
    
    with col3:
        st.success("✅ Prediction Agent")
        st.write("Next prediction: 15 min")
    
    with col4:
        st.success("✅ Dashboard Agent")
        st.write("Status: Active")
    
    # Auto refresh
    if auto_refresh:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()